"""Capture actual Inductor scheduler stages for one concrete FX input setting.

The graph is copied before compilation; tensor inputs and module attributes keep
their original storage. Collection runs in ``torch.no_grad()`` and uses scoped
internal Inductor hooks. Do not run another compilation concurrently in the same
process while collecting a capture.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import operator
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch
from torch._guards import detect_fake_mode
from torch._dynamo.utils import GmWrapper, flatten_graph_inputs
from torch._inductor import config as inductor_config, debug, ir
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import Scheduler
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.virtualized import V
from torch.fx.traceback import NodeSource, NodeSourceAction
from torch.fx.experimental.symbolic_shapes import PropagateUnbackedSymInts
from torch.utils import _pytree as pytree

from .graph import input_geometry, json_value, normalize_fx_graph, value_geometry
from .execution import annotate_scheduler_execution


TRITON_MATMUL_CONFIG = {
    "max_autotune_gemm": True,
    "max_autotune_gemm_backends": "TRITON",
    "autotune_fallback_to_aten": False,
    "triton.native_matmul": False,
    "epilogue_fusion": True,
    "benchmark_template_fusion": True,
}


SCHEDULE_ONLY_CONFIG = {
    "deterministic": True,
    "benchmark_fusion": False,
    "benchmark_template_fusion": False,
    "pipeline_max_autotune_gemm": False,
    "distributed_max_autotune_gemm": False,
}


class _ScheduleCaptureComplete(BaseException):
    """Leave compilation after the real scheduler pass without error wrapping."""


def serialize_dependency(dep):
    result = {"type": type(dep).__name__, "name": getattr(dep, "name", None), "repr": repr(dep)}
    if dataclasses.is_dataclass(dep):
        result.update({f.name: json_value(getattr(dep, f.name)) for f in dataclasses.fields(dep)})
    result["weak"] = type(dep).__name__ == "WeakDep"
    return result


def flatten_scheduler_nodes(snode):
    children = list(snode.get_nodes())
    if len(children) == 1 and children[0] is snode:
        return [snode]
    return [leaf for child in children for leaf in flatten_scheduler_nodes(child)]


class CaptureSession:
    """Collect the actual IR snapshots without modifying compiler decisions."""

    def __init__(self, gm, name_to_original_id, outdir, flatten_outputs=False, schedule_only=False):
        self.gm = gm
        self.root_graph_id = id(gm.graph)
        self.name_to_original_id = dict(name_to_original_id)
        self.original_ids = set(self.name_to_original_id.values())
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.flatten_outputs = flatten_outputs
        self.schedule_only = schedule_only
        self.output_spec = None
        self._ir_results = {}
        self.data = {
            "original_graph_id": self.root_graph_id,
            "original_name_to_id": self.name_to_original_id,
            "stages": {},
            "stage_runs": [],
            "accepted_fusions": [],
            "lowering_returns": [],
            "provenance_errors": [],
            "instrumentation_errors": [],
            "provenance_execution": [],
            "original_value_observations": {},
            "flatten_outputs": flatten_outputs,
            "forbidden_compile_calls": [],
        }
        self._stack = contextlib.ExitStack()
        self._saved_meta = []

    def __enter__(self):
        for node in self.gm.graph.nodes:
            if node.name not in self.name_to_original_id:
                continue
            self._saved_meta.append((node, dict(node.meta)))
            source = NodeSource(node, "fx_visualizer_captured_fx", NodeSourceAction.CREATE)
            node.meta["from_node"] = [source]
            node.meta["fx_visualizer_original_id"] = self.name_to_original_id[node.name]
            node.meta["fx_visualizer_original_graph_id"] = self.root_graph_id

        old_pre = debug.log_ir_pre_fusion
        old_post = debug.log_ir_post_fusion
        old_run_node = GraphLowering.run_node
        old_fuse_two = Scheduler.fuse_two_nodes
        old_wrapper_forward = GmWrapper.forward
        capture = self

        def wrapper_forward(wrapper, *args):
            if wrapper.gm is not capture.gm:
                return old_wrapper_forward(wrapper, *args)
            interpreter_cls = PropagateUnbackedSymInts if detect_fake_mode() is not None else torch.fx.Interpreter
            capture.data["provenance_execution"].append({
                "graph_id": id(wrapper.gm.graph),
                "wrapper": "torch._dynamo.utils.GmWrapper",
                "execution": interpreter_cls.__name__ + " with preserved node metadata",
            })

            class ObserveValues(interpreter_cls):
                def run_node(self, node):
                    result = super().run_node(node)
                    original_id = capture.name_to_original_id.get(node.name)
                    if original_id is not None:
                        capture.data["original_value_observations"][original_id] = value_geometry(result)
                    return result

            with torch.fx.traceback.preserve_node_meta():
                outputs = ObserveValues(wrapper.gm).run(*wrapper.unflatten_fn(list(args)))
            if not capture.flatten_outputs:
                return outputs
            leaves_with_paths, spec = pytree.tree_flatten_with_path(outputs)
            if capture.output_spec is not None and capture.output_spec != spec:
                raise RuntimeError("Captured FX output TreeSpec changed during tracing")
            capture.output_spec = spec
            capture.data["output_adapter"] = {
                "boundary_only": True,
                "original_tree_spec_readable": str(spec),
                "flat_leaf_count": len(leaves_with_paths),
                "flat_leaf_paths": [pytree.keystr(path) for path, _ in leaves_with_paths],
            }
            try:
                capture.data["output_adapter"]["original_tree_spec"] = pytree.treespec_dumps(spec)
            except NotImplementedError:
                capture.data["output_adapter"]["serialized_tree_spec_available"] = False
            return tuple(value for _, value in leaves_with_paths)

        def pre(nodes):
            capture._snapshot("lowering", nodes)
            return old_pre(nodes)

        def post(nodes):
            capture._snapshot("fusion", nodes)
            return old_post(nodes)

        def run_node(graph_lowering, node):
            result = old_run_node(graph_lowering, node)
            graph_id = id(graph_lowering.module.graph)
            capture._ir_results.setdefault(graph_id, {})[node] = result
            capture.data["lowering_returns"].append({
                "graph_id": graph_id,
                "fx": capture._fx(node),
                "result": capture._ir_value(result),
            })
            return result

        def fuse_two(scheduler, node1, node2, fused_nodes):
            before = [list(node1.get_operation_names()), list(node2.get_operation_names())]
            result = old_fuse_two(scheduler, node1, node2, fused_nodes)
            capture.data["accepted_fusions"].append({
                "graph_id": id(V.graph.module.graph),
                "inputs": before,
                "result": result.get_name(),
                "member_operation_names": list(result.get_operation_names()),
                "class": type(result).__name__,
            })
            return result

        self._stack.enter_context(patch.object(debug, "log_ir_pre_fusion", pre))
        self._stack.enter_context(patch.object(debug, "log_ir_post_fusion", post))
        self._stack.enter_context(patch.object(GraphLowering, "run_node", run_node))
        self._stack.enter_context(patch.object(Scheduler, "fuse_two_nodes", fuse_two))
        self._stack.enter_context(patch.object(GmWrapper, "forward", wrapper_forward))
        if self.schedule_only:
            old_post_pass = inductor_config._post_fusion_custom_pass

            def stop_after_fusion(nodes):
                if old_post_pass is not None:
                    nodes = old_post_pass(nodes)
                capture._snapshot("fusion", nodes)
                capture.data["stages"]["fusion"]["observation_point"] = "after Scheduler.fuse_nodes and the configured post-fusion pass; before merge_loops, template finalization and codegen"
                old_post(nodes)
                raise _ScheduleCaptureComplete()

            def forbid(name):
                def guarded(*args, **kwargs):
                    capture.data["forbidden_compile_calls"].append(name)
                    raise RuntimeError(f"Scheduling-only capture reached forbidden compiler operation: {name}")
                return guarded

            self._stack.enter_context(inductor_config.patch({"_post_fusion_custom_pass": stop_after_fusion}))
            for owner, method in [
                (AlgorithmSelectorCache, "make_precompile_fn"),
                (AlgorithmSelectorCache, "do_autotuning"),
                (Scheduler, "compile_kernel"),
                (Scheduler, "generate_kernel_code_from_nodes"),
                (Scheduler, "benchmark_fused_nodes"),
                (Scheduler, "benchmark_codegened_module"),
                (Scheduler, "codegen"),
            ]:
                name = f"{owner.__name__}.{method}"
                self._stack.enter_context(patch.object(owner, method, forbid(name)))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stack.close()
        for node, meta in self._saved_meta:
            node.meta.clear()
            node.meta.update(meta)
        if exc_value is not None and not isinstance(exc_value, _ScheduleCaptureComplete):
            self.data["compile_exception"] = {"type": exc_type.__name__, "message": str(exc_value)}
        self.write()
        return False

    def _get(self, obj, method, default=None):
        try:
            return getattr(obj, method)()
        except (AttributeError, AssertionError, NotImplementedError, RuntimeError, TypeError) as exc:
            self.data["instrumentation_errors"].append({
                "object_type": type(obj).__name__, "method": method,
                "exception": type(exc).__name__, "message": str(exc),
            })
            return default

    def _fx(self, node):
        raw_sources = []
        for source in node.meta.get("from_node", ()):
            if isinstance(source, NodeSource):
                raw_sources.append(source.to_dict())
            elif isinstance(source, dict):
                raw_sources.append(json_value(source))
            else:
                raw_sources.append({"unrecognized_source": json_value(source)})
        mapped = set()
        matches = []
        candidates = [{"name": node.name, "graph_id": id(node.graph), "from_node": raw_sources}]
        while candidates:
            source = candidates.pop()
            name, graph_id = source.get("name"), source.get("graph_id")
            if graph_id == self.root_graph_id and name in self.name_to_original_id:
                original_id = self.name_to_original_id[name]
                mapped.add(original_id)
                matches.append({"original_id": original_id, "name": name, "graph_id": graph_id})
            candidates.extend(item for item in source.get("from_node", ()) if isinstance(item, dict))
        if node.meta.get("fx_visualizer_original_graph_id") == self.root_graph_id:
            original_id = node.meta.get("fx_visualizer_original_id")
            if original_id in self.original_ids:
                mapped.add(original_id)
                matches.append({"original_id": original_id, "method": "preserved_original_metadata"})
        result = {
            "name": node.name, "op": node.op, "target": str(node.target),
            "is_getitem": node.op == "call_function" and node.target is operator.getitem,
            "graph_id": id(node.graph), "origin_fx_ids": sorted(mapped, key=str),
            "from_node": raw_sources, "matches": matches,
        }
        if "val" in node.meta:
            result["value"] = json_value(node.meta["val"])
        if not mapped and node.op == "call_function":
            self.data["provenance_errors"].append({"kind": "unmapped_fx_origin", **result})
        return result

    def _ir_value(self, value):
        if isinstance(value, (list, tuple)):
            return {"type": type(value).__name__, "values": [self._ir_value(v) for v in value]}
        if isinstance(value, dict):
            return {"type": "dict", "values": {str(k): self._ir_value(v) for k, v in value.items()}}
        if not isinstance(value, ir.IRNode):
            return {"type": type(value).__name__, "value": json_value(value)}
        current = value
        chain = []
        seen = set()
        while isinstance(current, ir.IRNode) and id(current) not in seen:
            seen.add(id(current))
            item = {"type": type(current).__name__, "object_id": id(current)}
            name = getattr(current, "name", None)
            if name is not None:
                item["buffer_name"] = str(name)
            if isinstance(current, ir.Operation):
                item["operation_name"] = getattr(current, "operation_name", None)
            item["is_view"] = isinstance(current, ir.BaseView)
            item["is_realized_node"] = ir.IRNode.is_realized_node(current)
            item["is_input_buffer"] = isinstance(current, ir.InputBuffer)
            item["origins"] = [self._fx(origin) for origin in getattr(current, "origins", ()) if isinstance(origin, torch.fx.Node)]
            if getattr(current, "origin_node", None) is not None:
                item["origin_node"] = self._fx(current.origin_node)
            chain.append(item)
            current = getattr(current, "data", None)
        buffer_names = list(dict.fromkeys(item["buffer_name"] for item in chain if "buffer_name" in item))
        operation_names = list(dict.fromkeys(item["operation_name"] for item in chain if item.get("operation_name") is not None))
        return {
            "type": type(value).__name__, "ir_chain": chain,
            "ir_object_id": id(value),
            "buffer_names": buffer_names,
            "operation_names": operation_names,
            "materialized": bool(buffer_names),
            "contains_view": any(item["is_view"] for item in chain),
            "input_buffer_names": [item["buffer_name"] for item in chain if item["is_input_buffer"] and "buffer_name" in item],
            "terminal_ir_type": chain[-1]["type"] if chain else None,
        }

    def _buffer(self, buf):
        node = buf.node
        result = {
            "name": buf.get_name(), "ir_type": type(node).__name__,
            "layout": str(getattr(node, "layout", None)),
            "aliases": list(buf.get_aliases()), "mutations": list(buf.get_mutations()),
            "users": [],
        }
        for user in getattr(buf, "users", ()) or ():
            result["users"].append({
                "name": self._get(user.node, "get_name", str(user.node)),
                "type": type(user.node).__name__,
                "can_inplace": getattr(user, "can_inplace", None),
                "is_weak": getattr(user, "is_weak", None),
            })
        for method, key in [("get_size", "shape"), ("get_stride", "stride"), ("get_dtype", "dtype"), ("get_device", "device")]:
            value = self._get(node, method)
            if value is not None:
                result[key] = json_value(value)
        return result

    def _loop_ir(self, snode):
        body = getattr(snode, "_body", None)
        blocks = []
        candidates = [("root", getattr(body, "root_block", None))]
        candidates.extend(getattr(body, "subblocks", {}).items())
        for name, block in candidates:
            graph = getattr(block, "graph", None)
            if not isinstance(graph, torch.fx.Graph):
                continue
            blocks.append({
                "name": name, "graph_id": id(graph),
                "nodes": [{"name": node.name, "op": node.op, "target": str(node.target),
                           "args": json_value(node.args), "kwargs": json_value(node.kwargs)}
                          for node in graph.nodes],
            })
        return {"blocks": blocks, "op_counts": json_value(getattr(body, "op_counts", {}))}

    def _leaf(self, snode):
        node = snode.node
        origins = [self._fx(origin) for origin in node.get_origins() if isinstance(origin, torch.fx.Node)]
        non_fx_origins = [json_value(origin) for origin in node.get_origins() if not isinstance(origin, torch.fx.Node)]
        caller = getattr(node, "_render_caller", None)
        caller_info = None
        if caller is not None:
            caller_info = {"class": type(caller).__name__}
            for field in ("name", "description", "bmreq", "info", "kwargs"):
                value = getattr(caller, field, None)
                if value is not None and not callable(value):
                    caller_info[field] = json_value(value)
        if snode.is_template():
            if isinstance(node, ir.MultiTemplateBuffer):
                backend = getattr(node, "_render_kind", None) or "unresolved_template_choices"
            elif isinstance(node, ir.TritonTemplateBuffer):
                backend = "triton"
            else:
                backend = type(node).__name__
        elif snode.is_extern():
            backend = "extern"
        elif type(snode).__name__ == "NopKernelSchedulerNode":
            backend = "nop"
        else:
            backend = "generated_compute"
        return {
            "name": snode.get_name(), "class": type(snode).__name__,
            "scheduler_type": type(snode).__name__, "ir_type": type(node).__name__,
            "origin_fx_ids": sorted({item for origin in origins for item in origin["origin_fx_ids"]}, key=str),
            "origin_fx_names": sorted({origin["name"] for origin in origins}),
            "origins": origins, "non_fx_origins": non_fx_origins,
            "origin_node": self._fx(node.origin_node) if node.origin_node is not None else None,
            "is_template": snode.is_template(), "is_extern": snode.is_extern(),
            "is_native_matmul": isinstance(node, ir.ComputedBuffer) and snode.is_native_matmul(),
            "reduction_type": getattr(getattr(node, "data", None), "reduction_type", None),
            "backend": backend, "template_render_kind": getattr(node, "_render_kind", None),
            "selected_caller": caller_info,
            "python_kernel_name": getattr(node, "python_kernel_name", None),
            "reads": [serialize_dependency(dep) for dep in snode.read_writes.reads],
            "writes": [serialize_dependency(dep) for dep in snode.read_writes.writes],
            "unmet_dependencies": [serialize_dependency(dep) for dep in snode.unmet_dependencies],
            "buffer_names": list(snode.get_buffer_names()),
            "outputs": [self._buffer(buf) for buf in snode.get_outputs()],
            "loop_body": str(getattr(snode, "_body", "")),
            "loop_ir": self._loop_ir(snode),
            "ir_inputs": [self._ir_value(value) for value in (getattr(node, "inputs", ()) or ())],
            "ir_named_inputs": {name: self._ir_value(value) for name, value in (getattr(node, "_named_inputs", {}) or {}).items()},
            "data_origins": [self._fx(origin) for origin in getattr(getattr(node, "data", None), "origins", ()) if isinstance(origin, torch.fx.Node)],
            "data_origin_node": self._fx(node.data.origin_node) if getattr(getattr(node, "data", None), "origin_node", None) is not None else None,
        }

    def _node(self, snode):
        leaves = [self._leaf(leaf) for leaf in flatten_scheduler_nodes(snode)]
        backends = sorted({leaf["backend"] for leaf in leaves})
        return {
            "id": snode.get_name(), "name": snode.get_name(), "kind": "scheduler",
            "class": type(snode).__name__, "scheduler_type": type(snode).__name__,
            "origin_fx_ids": sorted({item for leaf in leaves for item in leaf["origin_fx_ids"]}, key=str),
            "origin_fx_names": sorted({item for leaf in leaves for item in leaf["origin_fx_names"]}),
            "member_operation_names": list(snode.get_operation_names()),
            "reads": [serialize_dependency(dep) for dep in snode.read_writes.reads],
            "writes": [serialize_dependency(dep) for dep in snode.read_writes.writes],
            "unmet_dependencies": [serialize_dependency(dep) for dep in snode.unmet_dependencies],
            "buffer_names": list(snode.get_buffer_names()), "leaf_members": leaves,
            "is_template": any(leaf["is_template"] for leaf in leaves),
            "backend": backends[0] if len(backends) == 1 else "+".join(backends),
            "details": {
                "device": str(snode.get_device()), "debug_str": snode.debug_str(),
                "outputs": [self._buffer(buf) for buf in snode.get_outputs()],
                "direct_children": [child.get_name() for child in getattr(snode, "snodes", ())],
            },
        }

    def _snapshot(self, stage, snodes):
        graph = V.graph
        graph_id = id(graph.module.graph)
        nodes = [self._node(snode) for snode in snodes]
        producer = {buffer: node["id"] for node in nodes for buffer in node["buffer_names"]}
        boundaries = {}
        edges = []
        for node in nodes:
            dep_candidates = [(dep, True) for dep in node["unmet_dependencies"]]
            dep_candidates.extend((dep, False) for dep in node["reads"] if dep["name"] not in producer)
            seen_deps = set()
            for dep, unmet in dep_candidates:
                dep_key = (dep["name"], dep["repr"])
                if dep_key in seen_deps:
                    continue
                seen_deps.add(dep_key)
                name = dep["name"]
                if name in producer:
                    source = producer[name]
                else:
                    if name in graph.graph_inputs:
                        kind = "input"
                    elif name in graph.constants:
                        kind = "constant"
                    else:
                        kind = "external"
                    source = f"{kind}:{name}"
                    if source not in boundaries:
                        value = graph.graph_inputs.get(name, graph.constants.get(name))
                        boundaries[source] = {
                            "id": source, "name": name, "kind": kind,
                            "class": type(value).__name__, "scheduler_type": "boundary",
                            "origin_fx_ids": [], "origin_fx_names": [],
                            "member_operation_names": [], "buffer_names": [name],
                            "leaf_members": [], "is_template": False, "backend": kind,
                            "details": {"value": self._ir_value(value)},
                        }
                        for origin in getattr(value, "origins", ()):
                            if isinstance(origin, torch.fx.Node):
                                fx = self._fx(origin)
                                boundaries[source]["origin_fx_ids"].extend(fx["origin_fx_ids"])
                                boundaries[source]["origin_fx_names"].append(fx["name"])
                if source != node["id"]:
                    edges.append({"source": source, "target": node["id"], "buffer": name, "dependency_type": dep["type"], "weak": dep["weak"], "unmet": unmet, "dependency": dep})
        output_values = []
        for index, value in enumerate(graph.graph_outputs):
            result = {"index": index, "value": self._ir_value(value)}
            name = self._get(value, "get_name") if isinstance(value, ir.IRNode) else None
            result["buffer_name"] = name
            result["producer"] = producer.get(name)
            output_values.append(result)
            if name in producer:
                output_id = f"output:{index}"
                boundaries[output_id] = {
                    "id": output_id, "name": f"output[{index}]", "kind": "output",
                    "class": type(value).__name__, "scheduler_type": "boundary",
                    "origin_fx_ids": [], "origin_fx_names": [],
                    "member_operation_names": [], "buffer_names": [name],
                    "leaf_members": [], "is_template": False, "backend": "output",
                    "details": result,
                }
                edges.append({"source": producer[name], "target": output_id, "buffer": name, "dependency_type": "output", "weak": False})
        transformed_fx = []
        final_ir_values = []
        retained_values = self._ir_results.get(graph_id, {})
        for node in graph.module.graph.nodes:
            fx = self._fx(node)
            fx["args"] = json_value(node.args)
            fx["kwargs"] = json_value(node.kwargs)
            transformed_fx.append(fx)
            in_actual_env = node in graph.env
            retained = node in retained_values
            final_value = graph.env[node] if in_actual_env else retained_values.get(node)
            final_ir_values.append({
                "fx": fx, "in_actual_env": in_actual_env,
                "retained_lowering_result": retained,
                "observation_source": "actual_graph_lowering_env" if in_actual_env else "retained_run_node_result" if retained else "unavailable",
                "value": self._ir_value(final_value) if in_actual_env or retained else None,
            })
        snapshot = {
            "stage": stage, "graph_id": graph_id,
            "nodes": nodes + list(boundaries.values()), "edges": edges,
            "scheduler_nodes": nodes, "boundaries": list(boundaries.values()),
            "scheduler_node_count": len(nodes),
            "kernel_node_count": sum(node["scheduler_type"] != "NopKernelSchedulerNode" for node in nodes),
            "transformed_fx": transformed_fx, "output_values": output_values,
            "final_ir_values": final_ir_values,
            "actual_env_node_names": [node.name for node in graph.env],
            "environment_gc_enabled": graph.garbage_collect_values,
            "graph_input_names": list(graph.graph_inputs),
            "graph_output_names": list(graph.get_output_names()),
        }
        self.data["stage_runs"].append(snapshot)
        self.data["stages"][stage] = snapshot
        run_index = len(self.data["stage_runs"]) - 1
        (self.outdir / f"{run_index:02d}_{stage}.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        (self.outdir / f"{run_index:02d}_{stage}_ir.txt").write_text("\n\n".join(snode.debug_str() for snode in snodes) + "\n")
        self.write()

    def write(self):
        path = self.outdir / "capture.json"
        path.write_text(json.dumps(self.data, indent=2) + "\n")
        return path

    finish = write


def capture_scheduler_stages(gm, example_inputs, output_dir, *, config=None, triton_matmul=False, schedule_only=True):
    """Capture actual lowering and fusion, optionally stopping before codegen.

    ``example_inputs`` are the positional inputs for this concrete setting. No
    inputs are regenerated, cast, reshaped or copied. ``triton_matmul=True``
    limits GEMM autotuning to Triton templates, disables ATen fallback and native
    matmul, and enables template epilogue fusion. Other compiler defaults remain
    unchanged; ``config`` can override those GEMM choices. IR tracing is always
    enabled, and both FX caches are disabled to observe this actual compilation.

    ``schedule_only=True`` stops after the real fusion pass and its configured
    custom pass, before loop merging, deferred template finalization and kernel
    codegen. It enforces deterministic, unbenchmarked algorithm selection and
    disables fusion benchmarking and pipelined/distributed GEMM autotuning.
    These settings describe a heuristic schedule preview, not the tuned runtime
    result. No compiled graph is executed. Hooks reject unexpected compiler
    precompilation, autotuning or benchmark codegen rather than silently wait.
    Template candidate construction may still render Python Triton source.

    The returned dictionary includes ``capture_path``, actual dump paths and
    conservative execution/indexing memberships. Nested input/output pytrees
    are adapted only at the compiler boundary. This function runs without
    autograd. Full mode executes effects already present in the
    supplied graph; schedule-only mode only performs the compiler's tracing.
    """
    if not isinstance(gm, torch.fx.GraphModule):
        raise TypeError(f"Expected torch.fx.GraphModule, got {type(gm).__name__}")
    inputs = tuple(example_inputs)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_code = gm.code
    compiled_gm = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
    original, source_ids = normalize_fx_graph(compiled_gm, inputs)
    original["name"] = output_dir.name
    (output_dir / "original_fx.py").write_text(original_code)
    trace_dir = Path(tempfile.mkdtemp(prefix="inductor_trace_", dir=output_dir))
    options = dict(TRITON_MATMUL_CONFIG) if triton_matmul else {}
    options.update(config or {})
    if schedule_only:
        options.update(SCHEDULE_ONLY_CONFIG)
    options.update({
        "triton.cudagraphs": False,
        "fx_graph_cache": False,
        "fx_graph_remote_cache": False,
        "trace.enabled": True,
        "trace.debug_dir": str(trace_dir),
        "trace.provenance_tracking_level": 1,
        "trace.save_real_tensors": False,
        "trace.log_autotuning_results": True,
    })
    geometries, input_aliases, storage_groups = input_geometry(inputs)
    compile_info = {
        "status": "running", "entrypoint": "torch._inductor.compile via standard input-pytree boundary wrapper",
        "mode": "schedule_only" if schedule_only else "compile_and_execute",
        "torch_version": torch.__version__, "torch_source": torch.__file__,
        "options": options, "grad_enabled": False, "inputs": geometries,
        "input_aliases": input_aliases, "input_storage_groups": storage_groups,
        "output_boundary_adapter": "pytree flatten for AOT, then restore the original output tree",
    }
    if schedule_only:
        compile_info.update({
            "schedule_policy": "deterministic unbenchmarked candidates and heuristic fusion",
            "algorithm_selection": "Inductor deterministic choice: first extern if present, otherwise first valid template",
            "enforced_schedule_only_options": SCHEDULE_ONLY_CONFIG,
            "stop_point": "after Scheduler.fuse_nodes and configured post-fusion pass; before merge_loops, template finalization and kernel codegen",
            "autotuning_performed": False, "fusion_benchmarking_performed": False,
            "final_kernel_codegen_performed": False, "execution_succeeded": False,
            "gpu_precompilation_performed": False,
            "template_candidate_source_rendering_may_occur": True,
            "selected_runtime_backend_proven": False,
        })
    capture = CaptureSession(compiled_gm, source_ids, output_dir, flatten_outputs=True, schedule_only=schedule_only)
    capture.data.update({
        "name": output_dir.name, "schema_version": 1, "original_graph": original,
        "provided_graph_id": id(gm.graph), "fx_mapping": source_ids,
        "capture_path": str(output_dir / "capture.json"), "compile": compile_info,
    })
    started = time.perf_counter()
    try:
        with torch.no_grad(), inductor_config.patch(options), capture:
            compile_started = time.perf_counter()
            try:
                try:
                    compiled = flatten_graph_inputs(compiled_gm, inputs, torch._inductor.compile)
                finally:
                    compile_info["compile_seconds"] = time.perf_counter() - compile_started
            except _ScheduleCaptureComplete:
                compile_info.update(status="success", scheduling_succeeded=True)
            else:
                if schedule_only:
                    raise RuntimeError("Compilation returned without reaching the scheduler-only stopping hook")
                actual_flat = compiled(*inputs)
                if capture.output_spec is None:
                    raise RuntimeError("The compiler did not observe the original output pytree")
                actual = pytree.tree_unflatten(list(actual_flat), capture.output_spec)
                for device in {value.device for value in pytree.tree_leaves(inputs) if isinstance(value, torch.Tensor)}:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                compile_info.update(status="success", execution_succeeded=True, output_geometry=value_geometry(actual))
        for node in original["nodes"]:
            geometry = capture.data["original_value_observations"].get(node["id"])
            if geometry is not None:
                node["geometry"] = geometry
        original_nodes = {node["id"]: node for node in original["nodes"]}
        for node in original["nodes"]:
            for port in node.get("output_ports", []):
                if "producer" in port:
                    port["geometry"] = original_nodes[port["producer"]].get("geometry")
    except Exception as error:
        compile_info.update(status="error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        compile_info["total_seconds"] = time.perf_counter() - started
        compile_info["provided_fx_code_unchanged"] = gm.code == original_code
        compile_info["artifacts"] = {
            kind: [str(path) for path in sorted(trace_dir.rglob(filename))]
            for kind, filename in [
                ("ir_pre_fusion", "ir_pre_fusion.txt"), ("ir_post_fusion", "ir_post_fusion.txt"),
                ("output_code", "output_code.py"), ("fx_graph_transformed", "fx_graph_transformed.py"),
            ]
        }
        capture.write()
    annotate_scheduler_execution(capture.data)
    capture.write()
    return capture.data

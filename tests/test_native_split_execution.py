"""Structured regressions for lazy work around a native matmul split."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fx_visualizer.execution import annotate_scheduler_execution
from fx_visualizer.lazy import inlined_pointwise_proofs
from fx_visualizer.projection import project_capture, replay_projection


_GRAPH = "native-split-fixture"
_BIAS_NAMES = {"cast_up", "negate", "exponential", "offset", "normalized", "cast_down"}


def _source(name):
    return f"fx:{name}"


def _fx_ref(name):
    return {"name": name, "graph_id": _GRAPH}


def _loop_ref(name, block):
    return {"name": name, "graph_id": block}


def _operation(block, name, target, *args):
    return {
        "name": name, "op": "call_method", "target": target,
        "args": [_loop_ref("ops", block), *args], "kwargs": {},
    }


def _binding(name, terminal, buffer=None, operation=None):
    chain = []
    if buffer is not None and terminal != "InputBuffer":
        chain.append({
            "type": "ComputedBuffer", "object_id": f"buffer:{buffer}",
            "buffer_name": buffer, "operation_name": operation,
            "origin_node": _fx_ref(name),
        })
    data = {
        "type": terminal, "object_id": f"data:{name}",
        "origin_node": _fx_ref(name),
    }
    if terminal == "InputBuffer":
        data["buffer_name"] = buffer
    chain.append(data)
    return {
        "fx": _fx_ref(name), "in_actual_env": False,
        "retained_lowering_result": True,
        "observation_source": "retained_run_node_result",
        "value": {
            "type": "TensorBox", "ir_chain": chain,
            "ir_object_id": f"box:{name}", "terminal_ir_type": terminal,
            "buffer_names": [buffer] if buffer is not None else [],
            "operation_names": [operation] if operation is not None else [],
            "contains_view": False, "materialized": buffer is not None,
        },
    }


def _bias_body(block):
    ref = lambda name: _loop_ref(name, block)
    op = lambda name, target, *args: _operation(block, name, target, *args)
    return [
        op("numerator_load", "load", "product_buffer", 0),
        op("numerator_cast", "to_dtype", ref("numerator_load"), "torch.float32"),
        op("denominator_load", "load", "product_buffer", 0),
        op("denominator_cast", "to_dtype", ref("denominator_load"), "torch.float32"),
        op("negative", "neg", ref("denominator_cast")),
        op("exponent", "exp", ref("negative")),
        op("one", "constant", 1.0, "torch.float32"),
        op("denominator", "add", ref("exponent"), ref("one")),
        op("ratio", "truediv", ref("numerator_cast"), ref("denominator")),
        op("bias_cast", "to_dtype", ref("ratio"), "torch.bfloat16"),
    ]


def native_split_capture(*, materialize_bias=False, beta=1):
    """Two dots, one lazy bias epilogue, and two final-result reductions."""
    definitions = [
        ("input", "placeholder", "input", [], {}),
        ("weights", "placeholder", "weights", [], {}),
        ("other_weights", "placeholder", "other_weights", [], {}),
        ("matrix_operand", "call_function", "aten.relu.default", [_fx_ref("input")], {}),
        ("product", "call_function", "aten.mm.default", [_fx_ref("input"), _fx_ref("weights")], {}),
        ("cast_up", "call_function", "prims.convert_element_type.default", [_fx_ref("product"), "torch.float32"], {}),
        ("negate", "call_function", "aten.neg.default", [_fx_ref("cast_up")], {}),
        ("exponential", "call_function", "aten.exp.default", [_fx_ref("negate")], {}),
        ("offset", "call_function", "aten.add.Tensor", [_fx_ref("exponential"), 1.0], {}),
        ("normalized", "call_function", "aten.div.Tensor", [_fx_ref("cast_up"), _fx_ref("offset")], {}),
        ("cast_down", "call_function", "prims.convert_element_type.default", [_fx_ref("normalized"), "torch.bfloat16"], {}),
        ("biased", "call_function", "aten.addmm.default", [_fx_ref("cast_down"), _fx_ref("matrix_operand"), _fx_ref("other_weights")], {"beta": beta}),
        ("total", "call_function", "aten.sum.dim_IntList", [_fx_ref("biased"), [-1], True], {}),
        ("square", "call_function", "aten.mul.Tensor", [_fx_ref("biased"), _fx_ref("biased")], {}),
        ("square_total", "call_function", "aten.sum.dim_IntList", [_fx_ref("square"), [-1], True], {}),
        ("output", "output", "output", [[_fx_ref(name) for name in ("product", "biased", "total", "square_total")]], {}),
    ]
    original = [
        {"id": _source(name), "fx_name": name, "fx_op": kind, "target": target,
         "is_matmul": name in {"product", "biased"}}
        for name, kind, target, _, _ in definitions
    ]
    transformed = [
        {"name": name, "op": kind, "target": target, "graph_id": _GRAPH,
         "origin_fx_ids": [_source(name)], "args": args, "kwargs": kwargs}
        for name, kind, target, args, kwargs in definitions
    ]
    values = [
        _binding("input", "InputBuffer", "input_buffer"),
        _binding("weights", "InputBuffer", "weights_buffer"),
        _binding("other_weights", "InputBuffer", "other_weights_buffer"),
        _binding("matrix_operand", "Pointwise"),
        _binding("product", "Reduction", "product_buffer", "product_dot"),
        *[_binding(name, "Pointwise") for name in sorted(_BIAS_NAMES)],
        _binding("biased", "Pointwise", "final_buffer", "biased_epilogue"),
        _binding("total", "Reduction", "total_buffer", "total_reduce"),
        _binding("square", "Pointwise"),
        _binding("square_total", "Reduction", "square_total_buffer", "square_reduce"),
    ]

    def leaf(name, root, buffer, reads, body, *, native=False, reduction=None, ancestry=()):
        return {
            "name": name, "class": "SchedulerNode", "scheduler_type": "SchedulerNode",
            "ir_type": "ComputedBuffer", "buffer_names": [buffer],
            "reads": [{"name": read, "weak": False} for read in reads],
            "origin_node": _fx_ref(root) if root is not None else None,
            "data_origin_node": _fx_ref(root) if root is not None else None,
            "data_origins": [_fx_ref(actual) for actual in ancestry],
            "origin_fx_ids": [_source(actual) for actual in ancestry],
            "is_native_matmul": native, "reduction_type": reduction,
            "is_template": False, "is_extern": False, "backend": "generated_compute",
            "loop_body": "structured fixture",
            "loop_ir": {"blocks": [{"graph_id": f"loop:{name}", "nodes": body}]},
        }

    def dot_body(name, output, right, *, prepare_left=False):
        block = f"loop:{name}"
        ref = lambda actual: _loop_ref(actual, block)
        op = lambda actual, target, *args: _operation(block, actual, target, *args)
        body = [
            op("left", "load", "input_buffer", 0),
            op("right", "load", right, 0),
        ]
        left = "left"
        if prepare_left:
            body.append(op("matrix_operand", "relu", ref("left")))
            left = "matrix_operand"
        body.extend([
            op("product", "dot", ref(left), ref("right")),
            op("reduce", "reduction", "torch.float32", "torch.float32", "dot", ref("product")),
            op("store", "store_reduction", output, 0, ref("reduce")),
        ])
        return body

    inherited = ["biased", "matrix_operand", *sorted(_BIAS_NAMES)]
    leaves = [
        leaf("product_dot", "product", "product_buffer", ["input_buffer", "weights_buffer"],
             dot_body("product_dot", "product_buffer", "weights_buffer"),
             native=True, reduction="dot", ancestry=["product"]),
        # Internal dot stages can inherit bias origins without executing the bias.
        leaf("biased_dot", None, "native_intermediate", ["input_buffer", "other_weights_buffer"],
             dot_body("biased_dot", "native_intermediate", "other_weights_buffer", prepare_left=True),
             native=True, reduction="dot", ancestry=inherited),
    ]
    block = "loop:biased_epilogue"
    ref = lambda actual: _loop_ref(actual, block)
    op = lambda actual, target, *args: _operation(block, actual, target, *args)
    epilogue = [op("accumulator", "load", "native_intermediate", 0)]
    reads = ["native_intermediate"]
    if beta != 0:
        if materialize_bias:
            binding = next(item for item in values if item["fx"]["name"] == "cast_down")
            binding.update(_binding("cast_down", "Pointwise", "bias_buffer", "bias_materialize"))
            bias_block = "loop:bias_materialize"
            bias_body = _bias_body(bias_block)
            bias_body.append(_operation(bias_block, "store", "store", "bias_buffer", 0,
                                        _loop_ref("bias_cast", bias_block)))
            leaves.append(leaf("bias_materialize", "cast_down", "bias_buffer", ["product_buffer"],
                               bias_body, ancestry=sorted(_BIAS_NAMES)))
            epilogue.append(op("bias_cast", "load", "bias_buffer", 0))
            reads.append("bias_buffer")
        else:
            epilogue.extend(_bias_body(block))
            reads.append("product_buffer")
        epilogue.append(op("add_bias", "add", ref("accumulator"), ref("bias_cast")))
        stored = "add_bias"
    else:
        epilogue.append(op("final_cast", "to_dtype", ref("accumulator"), "torch.bfloat16"))
        stored = "final_cast"
    epilogue.append(op("store", "store", "final_buffer", 0, ref(stored)))
    leaves.append(leaf("biased_epilogue", "biased", "final_buffer", reads, epilogue, ancestry=inherited))

    for name, root, output, square in (
        ("total_reduce", "total", "total_buffer", False),
        ("square_reduce", "square_total", "square_total_buffer", True),
    ):
        block = f"loop:{name}"
        ref = lambda actual: _loop_ref(actual, block)
        op = lambda actual, target, *args: _operation(block, actual, target, *args)
        body = [op("value", "load", "final_buffer", 0)]
        reduced = "value"
        if square:
            body.extend([op("other_value", "load", "final_buffer", 0),
                         op("squared", "mul", ref("value"), ref("other_value"))])
            reduced = "squared"
        body.extend([
            op("reduce", "reduction", "torch.bfloat16", "torch.bfloat16", "sum", ref(reduced)),
            op("store", "store_reduction", output, 0, ref("reduce")),
        ])
        leaves.append(leaf(name, root, output, ["final_buffer"], body,
                           reduction="sum", ancestry=[root, *(["square"] if square else []), *inherited]))

    lowering = [{"id": item["name"], "member_operation_names": [item["name"]],
                 "leaf_members": [item]} for item in leaves]
    lowering_edges = [
        {"source": "product_dot", "target": "bias_materialize" if materialize_bias else "biased_epilogue",
         "buffer": "product_buffer"},
        {"source": "biased_dot", "target": "biased_epilogue", "buffer": "native_intermediate"},
        {"source": "biased_epilogue", "target": "total_reduce", "buffer": "final_buffer"},
        {"source": "biased_epilogue", "target": "square_reduce", "buffer": "final_buffer"},
    ]
    if beta == 0:
        lowering_edges = [edge for edge in lowering_edges if edge["source"] != "product_dot"]
    elif materialize_bias:
        lowering_edges.append({"source": "bias_materialize", "target": "biased_epilogue", "buffer": "bias_buffer"})
    fusion = []
    for name, members in (
        ("matmul_scope", [item for item in leaves if item["name"] not in {"total_reduce", "square_reduce"}]),
        ("reduction_scope", [item for item in leaves if item["name"] in {"total_reduce", "square_reduce"}]),
    ):
        fusion.append({"id": name, "member_operation_names": [item["name"] for item in members],
                       "leaf_members": copy.deepcopy(members)})
    return {
        "name": "structured_native_split", "compile": {"mode": "schedule_only"},
        "original_graph": {"nodes": original, "edges": [
            {"source": _source(source), "target": _source(target)} for source, target in (
                ("input", "matrix_operand"), ("input", "product"), ("weights", "product"),
                ("product", "cast_up"), ("cast_up", "negate"), ("negate", "exponential"),
                ("exponential", "offset"), ("cast_up", "normalized"), ("offset", "normalized"),
                ("normalized", "cast_down"), ("cast_down", "biased"), ("matrix_operand", "biased"),
                ("other_weights", "biased"), ("biased", "total"), ("biased", "square"),
                ("square", "square_total"), ("product", "output"), ("biased", "output"),
                ("total", "output"), ("square_total", "output"),
            )
        ]},
        "stages": {
            "lowering": {"graph_id": _GRAPH, "transformed_fx": transformed, "final_ir_values": values,
                         "scheduler_nodes": lowering, "nodes": lowering, "edges": lowering_edges},
            "fusion": {"graph_id": _GRAPH, "scheduler_nodes": fusion, "nodes": fusion,
                       "edges": [{"source": "matmul_scope", "target": "reduction_scope", "buffer": "final_buffer"}]},
        },
    }


class TestNativeSplitExecution(unittest.TestCase):
    def annotated(self, **kwargs):
        capture = native_split_capture(**kwargs)
        annotate_scheduler_execution(capture)
        nodes = {node["id"]: node for node in capture["stages"]["lowering"]["scheduler_nodes"]}
        return capture, nodes

    def test_multistep_bias_and_square_have_their_actual_owners(self):
        capture, nodes = self.annotated()
        bias_ids = {_source(name) for name in _BIAS_NAMES}
        self.assertEqual(set(nodes["biased_epilogue"]["execution_fx_ids"]), bias_ids)
        self.assertEqual(set(nodes["product_dot"]["execution_fx_ids"]), {_source("product")})
        self.assertEqual(set(nodes["biased_dot"]["execution_fx_ids"]), {_source("biased"), _source("matrix_operand")})
        self.assertEqual(set(nodes["total_reduce"]["execution_fx_ids"]), {_source("total")})
        self.assertEqual(set(nodes["square_reduce"]["execution_fx_ids"]), {_source("square"), _source("square_total")})
        self.assertEqual(nodes["biased_epilogue"]["native_matmul_leaf_count"], 0)
        fused = capture["stages"]["fusion"]["scheduler_nodes"][0]
        self.assertEqual(fused["native_matmul_leaf_count"], 2)
        self.assertEqual(set(fused["native_original_mm_ids"]), {_source("product"), _source("biased")})
        for name in _BIAS_NAMES:
            self.assertEqual(capture["fx_status"][_source(name)]["lowering"], ["biased_epilogue"])
        self.assertEqual(capture["fx_status"][_source("square")]["lowering"], ["square_reduce"])

    def test_final_result_consumers_cut_at_the_final_buffer(self):
        _, nodes = self.annotated()
        for name in ("total_reduce", "square_reduce"):
            with self.subTest(consumer=name):
                evidence = nodes[name]["leaf_members"][0]["execution_evidence"]
                self.assertEqual(evidence["uncertainties"], [])
                self.assertEqual({cut["buffer"] for cut in evidence["materialized_input_cuts"]}, {"final_buffer"})
                self.assertTrue(all(cut["matched_read_dependency"] for cut in evidence["materialized_input_cuts"]))
                self.assertEqual(nodes[name]["native_matmul_leaf_count"], 0)

    def test_materialized_bias_stays_with_its_producer(self):
        capture, nodes = self.annotated(materialize_bias=True)
        bias_ids = {_source(name) for name in _BIAS_NAMES}
        self.assertEqual(set(nodes["bias_materialize"]["execution_fx_ids"]), bias_ids)
        self.assertEqual(nodes["biased_epilogue"]["execution_fx_ids"], [])
        self.assertEqual(nodes["bias_materialize"]["native_matmul_leaf_count"], 0)
        self.assertEqual(nodes["biased_epilogue"]["native_matmul_leaf_count"], 0)
        for name in _BIAS_NAMES:
            self.assertEqual(capture["fx_status"][_source(name)]["lowering"], ["bias_materialize"])
        for name in ("total_reduce", "square_reduce"):
            self.assertTrue(bias_ids.isdisjoint(nodes[name]["execution_fx_ids"]))

    def test_lazy_matmul_binding_retains_a_materialized_input_boundary(self):
        capture = native_split_capture()
        stage = capture["stages"]["lowering"]
        values = stage["final_ir_values"]
        capture["original_graph"]["nodes"].insert(3, {
            "id": _source("product_operand"), "fx_name": "product_operand",
            "fx_op": "call_function", "target": "aten.neg.default", "is_matmul": False,
        })
        stage["transformed_fx"].insert(3, {
            "name": "product_operand", "op": "call_function", "target": "aten.neg.default",
            "graph_id": _GRAPH, "origin_fx_ids": [_source("product_operand")],
            "args": [_fx_ref("input")], "kwargs": {},
        })
        product_fx = next(node for node in stage["transformed_fx"] if node["name"] == "product")
        product_fx["args"][0] = _fx_ref("product_operand")
        original_edges = capture["original_graph"]["edges"]
        original_edges.remove({"source": _source("input"), "target": _source("product")})
        original_edges.extend([
            {"source": _source("input"), "target": _source("product_operand")},
            {"source": _source("product_operand"), "target": _source("product")},
        ])
        values.append(_binding("product_operand", "Pointwise"))
        producer = stage["scheduler_nodes"][0]["leaf_members"][0]
        block = "loop:product_dot"
        body = producer["loop_ir"]["blocks"][0]["nodes"]
        body.insert(2, _operation(block, "operand", "neg", _loop_ref("left", block)))
        body[3]["args"][1] = _loop_ref("operand", block)
        producer["origin_fx_ids"].append(_source("product_operand"))
        producer["data_origins"].append(_fx_ref("product_operand"))
        product = next(item for item in values if item["fx"]["name"] == "product")
        product.update(_binding("product", "Pointwise"))
        # A direct reduction consumer must cut at the proven native result even
        # when the retained FX result itself is an unnamed lazy finalization.
        total_fx = next(node for node in stage["transformed_fx"] if node["name"] == "total")
        total_fx["args"][0] = _fx_ref("product")
        original_edges.remove({"source": _source("biased"), "target": _source("total")})
        original_edges.append({"source": _source("product"), "target": _source("total")})
        consumer = next(node["leaf_members"][0] for node in stage["scheduler_nodes"]
                        if node["id"] == "total_reduce")
        consumer["reads"] = [{"name": "product_buffer", "weak": False}]
        consumer["origin_fx_ids"] = [_source(name) for name in ("total", "product", "product_operand")]
        consumer["data_origins"] = [_fx_ref(name) for name in ("total", "product", "product_operand")]
        consumer["loop_ir"]["blocks"][0]["nodes"][0]["args"][1] = "product_buffer"
        for edge in stage["edges"]:
            if edge["target"] == "total_reduce":
                edge.update(source="product_dot", buffer="product_buffer")
        capture["stages"]["fusion"]["edges"].append({
            "source": "matmul_scope", "target": "reduction_scope", "buffer": "product_buffer",
        })
        annotate_scheduler_execution(capture)
        nodes = {node["id"]: node for node in capture["stages"]["lowering"]["scheduler_nodes"]}
        evidence = nodes["total_reduce"]["leaf_members"][0]["execution_evidence"]
        cuts = [cut for cut in evidence["materialized_input_cuts"] if cut["buffer"] == "product_buffer"]
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]["owner"], ["product_dot"])
        self.assertTrue(cuts[0]["matched_read_dependency"])
        self.assertEqual(nodes["total_reduce"]["execution_fx_ids"], [_source("total")])
        self.assertEqual(nodes["total_reduce"]["native_matmul_leaf_count"], 0)
        self.assertEqual(capture["fx_status"][_source("product")]["lowering"], ["product_dot"])
        self.assertEqual(capture["fx_status"][_source("product_operand")]["lowering"], ["product_dot"])

    def test_zero_beta_does_not_turn_inherited_bias_origins_into_execution(self):
        _, nodes = self.annotated(beta=0)
        bias_ids = {_source(name) for name in _BIAS_NAMES}
        for name, node in nodes.items():
            with self.subTest(scheduler=name):
                self.assertTrue(bias_ids.isdisjoint(node["execution_fx_ids"]))
        self.assertEqual(nodes["biased_epilogue"]["execution_fx_ids"], [])
        self.assertEqual(nodes["biased_dot"]["native_matmul_leaf_count"], 1)

    def test_annotation_preserves_observed_graph_and_ir_and_is_repeatable(self):
        capture = native_split_capture()
        original = copy.deepcopy(capture["original_graph"])
        transformed = copy.deepcopy(capture["stages"]["lowering"]["transformed_fx"])
        values = copy.deepcopy(capture["stages"]["lowering"]["final_ir_values"])
        annotate_scheduler_execution(capture)
        before = copy.deepcopy(capture)
        annotate_scheduler_execution(capture)
        self.assertEqual(capture, before)
        self.assertEqual(capture["original_graph"], original)
        self.assertEqual(capture["stages"]["lowering"]["transformed_fx"], transformed)
        self.assertEqual(capture["stages"]["lowering"]["final_ir_values"], values)

    def test_projection_refreshes_lazy_only_scope_and_remaps_core_ids(self):
        source = native_split_capture()
        annotate_scheduler_execution(source)
        full = copy.deepcopy(source)
        omitted = {_source(name) for name in _BIAS_NAMES - {"cast_up"}} | {_source("square")}
        source["execution_annotation"] = {"version": 3}
        for stage in source["stages"].values():
            for node in stage["scheduler_nodes"]:
                for item in [node, *node["leaf_members"]]:
                    item["execution_fx_ids"] = [key for key in item["execution_fx_ids"] if key not in omitted]
                    evidence = item["execution_evidence"]
                    evidence["unknown_fx_ids"] = [key for key in evidence["unknown_fx_ids"] if key not in omitted]
        for key in omitted:
            source["fx_status"][key]["lowering"] = []
            source["fx_status"][key]["fusion"] = []

        def recovered(mapping):
            nodes = []
            for node in reversed(source["original_graph"]["nodes"]):
                if node["id"] in mapping:
                    nodes.append({**copy.deepcopy(node), "id": mapping[node["id"]]})
            edges = [
                {**edge, "source": mapping[edge["source"]], "target": mapping[edge["target"]]}
                for edge in source["original_graph"]["edges"]
                if edge["source"] in mapping and edge["target"] in mapping
            ]
            return {"nodes": nodes, "edges": edges}

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.json"
            path.write_text(json.dumps(source))
            persisted = path.read_bytes()
            scopes = (
                ({_source("normalized"): "local:z91", _source("cast_down"): "local:a04"},
                 {"matmul_scope"}, {"product_dot", "biased_dot", "biased_epilogue"}),
                ({_source("square"): "local:q17"},
                 {"reduction_scope"}, {"total_reduce", "square_reduce"}),
            )
            for mapping, fused_ids, lowered_ids in scopes:
                with self.subTest(mapping=mapping):
                    graph = recovered(mapping)
                    projected = project_capture(source_capture=path, original_graph=graph, fx_mapping=mapping)
                    self.assertEqual(projected["original_graph"], graph)
                    self.assertEqual({node["id"] for node in projected["stages"]["fusion"]["scheduler_nodes"]}, fused_ids)
                    self.assertEqual({node["id"] for node in projected["stages"]["lowering"]["scheduler_nodes"]}, lowered_ids)
                    expected_edges = [edge for edge in full["stages"]["lowering"]["edges"]
                                      if edge["source"] in lowered_ids and edge["target"] in lowered_ids]
                    self.assertEqual(projected["stages"]["lowering"]["edges"], expected_edges)
                    scoped_execution = {key for node in projected["stages"]["fusion"]["scheduler_nodes"]
                                        for key in node["execution_fx_ids"]}
                    self.assertEqual(scoped_execution, set(mapping.values()))
                    for stage in projected["stages"].values():
                        for node in stage["scheduler_nodes"]:
                            self.assertEqual(node["matmul_original_mm_ids"], [])
                    self.assertEqual(replay_projection(projected), projected)

            mapping = {_source("product"): "local:x83", _source("biased"): "local:m02",
                       _source("normalized"): "local:f61"}
            projected = project_capture(source_capture=path, original_graph=recovered(mapping), fx_mapping=mapping)
            fused = projected["stages"]["fusion"]["scheduler_nodes"][0]
            self.assertEqual(set(fused["matmul_original_mm_ids"]), {"local:x83", "local:m02"})
            self.assertEqual(set(fused["matmul_fx_ids"]), {"local:x83", "local:m02"})
            self.assertEqual(replay_projection(projected), projected)
            self.assertEqual(path.read_bytes(), persisted)

    def test_bias_proof_requires_exact_constants_operands_and_a_live_store_path(self):
        def prove(capture):
            stage = capture["stages"]["lowering"]
            fx = {node["name"]: node for node in stage["transformed_fx"]}
            values = {entry["fx"]["name"]: entry for entry in stage["final_ir_values"]}
            parents = {name: [arg["name"] for arg in node["args"]
                              if isinstance(arg, dict) and arg.get("graph_id") == _GRAPH]
                       for name, node in fx.items()}
            leaf = next(node["leaf_members"][0] for node in stage["scheduler_nodes"]
                        if node["id"] == "biased_epilogue")
            proofs = inlined_pointwise_proofs(
                leaf=leaf, roots=["cast_down"], fx=fx, values=values, parents=parents,
                graph_id=_GRAPH, source_ids=lambda names: {_source(name) for name in names},
            )
            return {key for proof in proofs for key in proof["execution_fx_ids"]}

        bias_ids = {_source(name) for name in _BIAS_NAMES}
        self.assertEqual(prove(native_split_capture()), bias_ids)
        for case in ("wrong_constant", "reversed_division", "disconnected_store"):
            with self.subTest(case=case):
                capture = native_split_capture()
                leaf = next(node["leaf_members"][0] for node in capture["stages"]["lowering"]["scheduler_nodes"]
                            if node["id"] == "biased_epilogue")
                operations = {node["name"]: node for node in leaf["loop_ir"]["blocks"][0]["nodes"]}
                if case == "wrong_constant":
                    operations["one"]["args"][1] = 2.0
                elif case == "reversed_division":
                    args = operations["ratio"]["args"]
                    args[1], args[2] = args[2], args[1]
                else:
                    operations["store"]["args"][3] = _loop_ref("accumulator", "loop:biased_epilogue")
                proven = prove(capture)
                self.assertNotIn(_source("cast_down"), proven)
                self.assertNotEqual(proven, bias_ids)
                if case == "disconnected_store":
                    self.assertEqual(proven, set())

    def test_recorded_v3_projection_is_verified_before_upgrade(self):
        source = native_split_capture()
        source["execution_annotation"] = {"version": 3}
        recorded_members = {
            "product_dot": ["product"], "biased_dot": ["biased", "matrix_operand"],
            "biased_epilogue": ["cast_up"], "total_reduce": ["total"],
            "square_reduce": ["square_total"],
        }
        recorded_unknown = {"biased_epilogue": ["biased"], "square_reduce": ["square"]}
        recorded_cores = {"product_dot": ["product"], "biased_dot": ["biased"]}
        lower = {}
        for node in source["stages"]["lowering"]["scheduler_nodes"]:
            name = node["id"]
            cores = [_source(actual) for actual in recorded_cores.get(name, [])]
            unknown = [_source(actual) for actual in recorded_unknown.get(name, [])]
            annotation = {
                "execution_fx_ids": [_source(actual) for actual in recorded_members[name]],
                "indexing_fx_ids": [], "native_original_mm_ids": cores,
                "native_matmul_leaf_count": len(cores),
                "execution_evidence": {"certainty": "partial" if unknown else "observed IR boundaries",
                                       "unknown_fx_ids": unknown},
            }
            node.update(copy.deepcopy(annotation))
            node["class"] = node["scheduler_type"] = "SchedulerNode"
            node["origin_fx_ids"] = list(node["leaf_members"][0]["origin_fx_ids"])
            node["matmul_original_mm_ids"] = list(cores)
            node["leaf_members"][0].update(copy.deepcopy(annotation))
            lower[name] = node
        for node in source["stages"]["fusion"]["scheduler_nodes"]:
            members = [lower[name] for name in node["member_operation_names"]]
            node["class"] = node["scheduler_type"] = "FusedSchedulerNode"
            for field in ("origin_fx_ids", "execution_fx_ids", "native_original_mm_ids", "matmul_original_mm_ids"):
                node[field] = sorted({key for member in members for key in member[field]})
            node["native_matmul_leaf_count"] = sum(member["native_matmul_leaf_count"] for member in members)
            node["indexing_fx_ids"] = []
            node["execution_evidence"] = {"certainty": "partial", "unknown_fx_ids":
                                          sorted({key for member in members for key in member["execution_evidence"]["unknown_fx_ids"]})}
            node["leaf_members"] = [copy.deepcopy(member["leaf_members"][0]) for member in members]
        source["matmul_lowering"] = [
            {"fx_id": _source(name), "fx_name": name, "target": target,
             "native_dot_reduction_leaves": [leaf], "template_leaves": [], "extern_leaves": [],
             "unknown": False, "runtime_backend_proven": False}
            for name, target, leaf in (("product", "aten.mm.default", "product_dot"),
                                       ("biased", "aten.addmm.default", "biased_dot"))
        ]
        source["fx_status"] = {
            _source("product"): {"role": "compute", "lowering": ["product_dot"], "fusion": ["matmul_scope"]},
            _source("normalized"): {"role": "unknown", "lowering": [], "fusion": []},
        }
        core, ratio = "legacy:core", "legacy:ratio"
        mapping = {_source("product"): core, _source("normalized"): ratio}
        graph = {"nodes": [{**copy.deepcopy(node), "id": mapping[node["id"]]}
                           for node in source["original_graph"]["nodes"] if node["id"] in mapping], "edges": []}

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recorded_v3_source.json"
            path.write_text(json.dumps(source))
            persisted = path.read_bytes()
            # Explicit old-writer record: only FX references were projected;
            # matmul_original_mm_ids still contained the full-graph IDs.
            projected_lower = []
            local_fields = {
                "product_dot": ([core], [core], [core], []),
                "biased_dot": ([ratio], [], [], [_source("biased"), _source("matrix_operand")]),
                "biased_epilogue": ([ratio], [], [], [_source("cast_up")]),
            }
            for name, (origins, execution, native, outside) in local_fields.items():
                node = copy.deepcopy(lower[name])
                for item in (node, node["leaf_members"][0]):
                    item["origin_fx_ids"] = list(origins)
                    item["execution_fx_ids"] = list(execution)
                    item["native_original_mm_ids"] = list(native)
                    item["execution_evidence"]["unknown_fx_ids"] = []
                node["matmul_fx_ids"] = [core] if name == "product_dot" else []
                node["source_scope"] = {"outside_fx_ids": outside, "full_graph_scheduler": name}
                if outside:
                    node["execution_evidence"]["certainty"] = "partial"
                projected_lower.append(node)
            fused = copy.deepcopy(source["stages"]["fusion"]["scheduler_nodes"][0])
            fused.update(origin_fx_ids=[ratio, core], execution_fx_ids=[core],
                         native_original_mm_ids=[core], matmul_fx_ids=[core],
                         leaf_members=[copy.deepcopy(node["leaf_members"][0]) for node in projected_lower])
            fused["execution_evidence"]["unknown_fx_ids"] = []
            fused["source_scope"] = {"outside_fx_ids": [_source("biased"), _source("cast_up"), _source("matrix_operand")],
                                     "full_graph_scheduler": "matmul_scope"}
            legacy = {
                "name": "recorded_v3_projection", "schema_version": 1,
                "original_graph": graph, "original_graph_id": None,
                "compile": copy.deepcopy(source["compile"]),
                "execution_annotation": {"version": 3, "context": "verified_full_graph_projection"},
                "fx_status": {core: copy.deepcopy(source["fx_status"][_source("product")]),
                              ratio: copy.deepcopy(source["fx_status"][_source("normalized")])},
                "matmul_lowering": [{**copy.deepcopy(source["matmul_lowering"][0]),
                                     "fx_id": core, "source_fx_id": _source("product")}],
                "accepted_fusions": [], "stages": {},
                "source_projection": {
                    "version": 1, "source_capture": str(path), "sha256": hashlib.sha256(persisted).hexdigest(),
                    "fx_mapping": mapping, "fusion_nodes": ["matmul_scope"],
                    "full_lowering_count": 5, "full_fusion_count": 2, "source_original_fx_count": 16,
                    "method": "Actual full-graph schedule; only FX references and displayed node scope are projected. No region recompilation.",
                },
            }
            for stage, nodes, edges in (
                ("lowering", projected_lower, source["stages"]["lowering"]["edges"][:2]),
                ("fusion", [fused], []),
            ):
                legacy["stages"][stage] = {
                    "stage": stage, "graph_id": _GRAPH, "nodes": nodes, "scheduler_nodes": nodes,
                    "scheduler_node_count": len(nodes), "edges": copy.deepcopy(edges),
                    "source_stage_reference": {"capture": str(path), "stage": stage},
                }
            self.assertEqual(legacy["execution_annotation"]["version"], 3)
            self.assertEqual(fused["matmul_original_mm_ids"], [_source("biased"), _source("product")])
            self.assertNotIn(ratio, fused["execution_fx_ids"])
            before = copy.deepcopy(legacy)
            upgraded = replay_projection(legacy)
            self.assertEqual(upgraded["execution_annotation"]["version"], 4)
            upgraded_fused = upgraded["stages"]["fusion"]["scheduler_nodes"][0]
            self.assertEqual(set(upgraded_fused["execution_fx_ids"]), {core, ratio})
            self.assertEqual(upgraded_fused["matmul_original_mm_ids"], [core])
            self.assertEqual(upgraded_fused["native_matmul_leaf_count"], 2)
            self.assertEqual(upgraded["fx_status"][ratio]["lowering"], ["biased_epilogue"])
            for old, new in zip(projected_lower, upgraded["stages"]["lowering"]["scheduler_nodes"]):
                self.assertEqual(new["class"], old["class"])
                self.assertEqual(new["leaf_members"][0]["loop_ir"], old["leaf_members"][0]["loop_ir"])
            self.assertEqual(upgraded["stages"]["lowering"]["edges"], legacy["stages"]["lowering"]["edges"])
            self.assertEqual(replay_projection(upgraded), upgraded)
            self.assertEqual(replay_projection(legacy), upgraded)
            self.assertEqual(legacy, before)
            self.assertEqual(path.read_bytes(), persisted)
            for changed in ("class", "loop_ir", "edge", "checksum"):
                with self.subTest(changed=changed):
                    invalid = copy.deepcopy(legacy)
                    if changed == "class":
                        invalid["stages"]["fusion"]["scheduler_nodes"][0]["class"] = "SchedulerNode"
                    elif changed == "loop_ir":
                        invalid["stages"]["lowering"]["scheduler_nodes"][0]["leaf_members"][0]["loop_ir"]["blocks"][0]["nodes"][2]["target"] = "mul"
                    elif changed == "edge":
                        invalid["stages"]["lowering"]["edges"][0]["buffer"] = "different_buffer"
                    else:
                        invalid["source_projection"]["sha256"] = "0" * 64
                    with self.assertRaises(ValueError):
                        replay_projection(invalid)


if __name__ == "__main__":
    unittest.main()

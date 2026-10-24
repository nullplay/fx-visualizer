"""Structured original FX nodes, argument ports and input geometry."""

from __future__ import annotations

import dataclasses
import math

import torch
from torch.utils import _pytree as pytree


_MATMUL_NAMES = frozenset({
    "mm", "bmm", "matmul", "addmm", "addmm_", "baddbmm", "baddbmm_",
    "addbmm", "addbmm_", "mv", "mv_", "addmv", "addmv_", "dot", "vdot",
    "linear", "einsum", "tensordot", "_int_mm", "_scaled_mm", "_scaled_mm_v2", "_grouped_mm",
    "_scaled_grouped_mm", "_scaled_grouped_mm_v2", "_sparse_mm", "_sparse_addmm",
    "_sparse_sparse_matmul", "_sparse_semi_structured_mm", "_sparse_semi_structured_addmm",
    "_weight_int4pack_mm", "_weight_int4pack_mm_for_cpu", "_weight_int8pack_mm", "_dyn_quant_matmul_4bit",
})


def is_matmul_node(node):
    schema = getattr(node.target, "_schema", None)
    return (node.op == "call_function" and schema is not None
            and schema.name.startswith("aten::") and schema.name[6:] in _MATMUL_NAMES)


def literal(value):
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        value = value.node.hint
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [literal(item) for item in value]]
    if isinstance(value, dict):
        return ["dict", [[literal(key), literal(item)] for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))]]
    if value is None or isinstance(value, (bool, int, str)):
        return [type(value).__name__, value]
    return [f"{type(value).__module__}.{type(value).__qualname__}", str(value)]


def value_geometry(value):
    if isinstance(value, torch.Tensor):
        result = {
            "kind": "tensor", "shape": json_value(list(value.shape)),
            "dtype": str(value.dtype), "device": str(value.device),
            "requires_grad": value.requires_grad,
            "layout": str(value.layout), "is_conj": value.is_conj(), "is_neg": value.is_neg(),
        }
        try:
            result.update(stride=json_value(list(value.stride())), storage_offset=json_value(value.storage_offset()))
        except (RuntimeError, NotImplementedError):
            result["layout"] = str(value.layout)
        return result
    if isinstance(value, (tuple, list)):
        return {"kind": type(value).__name__, "items": [value_geometry(item) for item in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[literal(key), value_geometry(item)] for key, item in value.items()]}
    return {"kind": "scalar", "value": json_value(value), "type": type(value).__name__}


def input_geometry(inputs):
    geometries = [value_geometry(value) for value in inputs]
    storage_groups = {}
    aliases = []
    for path, value in pytree.tree_flatten_with_path(tuple(inputs))[0]:
        if not isinstance(value, torch.Tensor):
            continue
        entry = {"path": pytree.keystr(path), "geometry": value_geometry(value)}
        try:
            storage = value.untyped_storage()
            key = (str(value.device), storage._cdata)
            if key not in storage_groups:
                storage_groups[key] = {"id": len(storage_groups), "nbytes": storage.nbytes(), "device": str(value.device)}
            entry["alias_group"] = storage_groups[key]["id"]
        except (RuntimeError, NotImplementedError):
            entry["alias_group"] = None
        aliases.append(entry)
    return geometries, aliases, list(storage_groups.values())


def normalize_fx_graph(gm, example_inputs):
    """Describe actual FX nodes and directed argument ports, without tracing."""
    placeholders = sum(node.op == "placeholder" for node in gm.graph.nodes)
    if len(example_inputs) != placeholders:
        raise ValueError(f"Expected {placeholders} example inputs, got {len(example_inputs)}")
    ids = {node.name: f"n{index}" for index, node in enumerate(gm.graph.nodes)}
    inputs = iter(example_inputs)
    edges = []

    def argument(value, target, path):
        if isinstance(value, torch.fx.Node):
            edges.append({"source": ids[value.name], "target": target, "port": list(path)})
            return {"ref": ids[value.name], "fx_name": value.name}
        if isinstance(value, (tuple, list)):
            return {"kind": type(value).__name__, "items": [argument(item, target, (*path, index)) for index, item in enumerate(value)]}
        if isinstance(value, dict):
            return {"kind": "dict", "items": [[literal(key), argument(item, target, (*path, str(key)))] for key, item in value.items()]}
        return {"literal": literal(value)}

    nodes = []
    for node in gm.graph.nodes:
        source = node.meta.get("bb_original_fx", {})
        schema = getattr(node.target, "_schema", None)
        record = {
            "id": ids[node.name], "fx_name": node.name, "fx_op": node.op,
            "target": str(node.target), "op_label": str(node.target),
            "label": source.get("source_name") or node.name,
            "kind": {"placeholder": "input", "get_attr": "constant", "output": "output"}.get(node.op, "compute"),
            "is_matmul": is_matmul_node(node), "synthetic": False,
            "actual_args": argument(node.args, ids[node.name], ("args",)),
            "actual_kwargs": argument(node.kwargs, ids[node.name], ("kwargs",)),
            "schema_name": schema.name if schema is not None else None,
            "source_metadata": json_value(source),
        }
        if node.op == "placeholder":
            record["geometry"] = value_geometry(next(inputs))
        elif "val" in node.meta:
            record["geometry"] = value_geometry(node.meta["val"])
        nodes.append(record)
    by_id = {node["id"]: node for node in nodes}

    def output_ports(value, path=()):
        if "ref" in value:
            producer = by_id[value["ref"]]
            return [{"path": list(path), "producer": producer["id"], "fx_name": producer["fx_name"],
                     "geometry": producer.get("geometry")}]
        if value.get("kind") in ("tuple", "list"):
            return [port for index, item in enumerate(value["items"]) for port in output_ports(item, (*path, index))]
        if value.get("kind") == "dict":
            return [port for key, item in value["items"] for port in output_ports(item, (*path, key))]
        return [{"path": list(path), "literal": value.get("literal")}]

    for node in nodes:
        if node["fx_op"] == "output":
            node["output_ports"] = output_ports(node["actual_args"]["items"][0])
    return {"name": gm.__class__.__name__, "nodes": nodes, "edges": edges,
            "metadata_capture": json_value(gm.meta.get("bb_metadata_capture", {}))}, ids


def json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, torch.fx.Node):
        return {"name": value.name, "target": str(value.target), "graph_id": id(value.graph)}
    if isinstance(value, torch.Tensor):
        return {
            "shape": [str(x) for x in value.shape],
            "stride": [str(x) for x in value.stride()],
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_value(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: json_value(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return str(value)

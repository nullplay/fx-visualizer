"""Prove inlined lazy pointwise work using captured FX and LoopBody objects."""

from __future__ import annotations

from collections import defaultdict


_UNARY = {
    "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "ceil",
    "cos", "cosh", "erf", "erfc", "exp", "expm1", "floor", "isfinite",
    "isinf", "isnan", "log", "log1p", "log2", "log10", "logical_not",
    "bitwise_not", "neg", "reciprocal", "relu", "round", "rsqrt", "sigmoid",
    "sign", "signbit", "sin", "sinh", "sqrt", "tan", "tanh", "trunc",
}
_BINARY = {
    "add", "atan2", "bitwise_and", "bitwise_or", "bitwise_xor", "copysign",
    "eq", "fmod", "ge", "gt", "le", "logical_and", "logical_or",
    "logical_xor", "lt", "maximum", "minimum", "mul", "ne", "nextafter",
    "pow", "remainder", "sub",
}
_ALIASES = {"div": "truediv", "divide": "truediv", "true_divide": "truediv", "floor_divide": "floordiv"}
_ARITY = {**dict.fromkeys(_UNARY, 1), **dict.fromkeys(_BINARY, 2),
          "truediv": 2, "floordiv": 2, "where": 3}
_DTYPES = {
    "bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16",
    "int32", "int64", "float16", "bfloat16", "float32", "float64",
    "complex32", "complex64", "complex128",
}
_DTYPE_ALIASES = {"half": "float16", "float": "float32", "double": "float64",
                  "short": "int16", "int": "int32", "long": "int64"}


def _dtype(value):
    if not isinstance(value, str):
        return None
    name = value.removeprefix("torch.")
    name = _DTYPE_ALIASES.get(name, name)
    return name if name in _DTYPES else None


def _routes(value):
    if not isinstance(value, dict):
        return []
    if "ir_chain" in value:
        return [value]
    children = value.get("values", [])
    if isinstance(children, dict):
        children = children.values()
    return [route for child in children for route in _routes(child)]


def _ordered(values):
    return sorted(set(values), key=lambda value: (
        int(value[1:]) if value.startswith("n") and value[1:].isdigit() else 10**12, value))


class _PointwiseMatcher:
    def __init__(self, *, leaf, fx, values, parents, graph_id, source_ids):
        self.leaf, self.fx, self.values = leaf, fx, values
        self.parents, self.graph_id, self.source_ids = parents, graph_id, source_ids
        self.reads = {dep["name"] for dep in leaf.get("reads", [])
                      if not dep.get("weak") and dep.get("type") != "WeakDep"}
        self.expressions = {}

    def fx_reference(self, value):
        if not isinstance(value, dict) or value.get("graph_id") != self.graph_id:
            return None
        node = self.fx.get(value.get("name"))
        if not node or node.get("graph_id", self.graph_id) != self.graph_id:
            return None
        if "target" in value and value["target"] != node.get("target"):
            return None
        return node["name"]

    def expression(self, name, pending=frozenset()):
        if name in self.expressions:
            return self.expressions[name]
        if name in pending:
            return None
        node = self.fx.get(name, {})
        entry = self.values.get(name, {})
        if self.fx_reference(entry.get("fx")) != name:
            return None
        routes = _routes(entry.get("value"))
        if len(routes) != 1 or routes[0].get("contains_view"):
            return None
        route = routes[0]
        buffers = set(route.get("buffer_names", []))
        buffers.update(item["buffer_name"] for item in route.get("ir_chain", []) if item.get("buffer_name"))
        geometry = node.get("value", {})
        dtype = _dtype(geometry.get("dtype")) if isinstance(geometry, dict) else None
        if buffers:
            if len(buffers) != 1:
                return None
            result = {"name": name, "buffer": next(iter(buffers)), "dtype": dtype}
            self.expressions[name] = result
            return result
        chain = route.get("ir_chain", [])
        if (route.get("materialized") or route.get("operation_names")
                or route.get("terminal_ir_type") != "Pointwise" or not chain
                or self.fx_reference(chain[-1].get("origin_node")) != name
                or node.get("op") != "call_function"):
            return None
        target = node.get("target", "").split(".")
        if len(target) < 2 or target[0] not in {"aten", "prims"}:
            return None
        operation = _ALIASES.get(target[1], target[1])
        args, kwargs = node.get("args", []), dict(node.get("kwargs", {}))
        if node["target"] == "prims.convert_element_type.default":
            if len(args) != 2 or kwargs or _dtype(args[1]) is None:
                return None
            operation, dtype, args = "to_dtype", _dtype(args[1]), args[:1]
        elif node["target"] == "aten._to_copy.default":
            dtype = _dtype(kwargs.pop("dtype", None))
            if len(args) != 1 or dtype is None or any(value is not False for value in kwargs.values()) or set(kwargs) - {"non_blocking"}:
                return None
            operation, kwargs = "to_dtype", {}
        else:
            if operation in {"add", "sub"} and kwargs.pop("alpha", 1) != 1:
                return None
            if operation == "truediv" and kwargs.pop("rounding_mode", None) is not None:
                return None
            if kwargs or operation not in _ARITY or len(args) != _ARITY[operation]:
                return None
        operands, references = [], []
        for value in args:
            if isinstance(value, dict):
                parent = self.fx_reference(value)
                if parent is None:
                    return None
                operand = self.expression(parent, pending | {name})
                if operand is None:
                    return None
                operands.append(operand)
                references.append(parent)
            elif type(value) in {int, float, bool}:
                operands.append({"literal": value})
            else:
                return None
        if list(dict.fromkeys(references)) != self.parents.get(name, []):
            return None
        input_dtypes = {operand.get("dtype") for operand in operands} - {None}
        input_dtype = next(iter(input_dtypes)) if len(input_dtypes) == 1 else None
        result = {"name": name, "target": operation, "args": operands,
                  "dtype": dtype or input_dtype, "input_dtype": input_dtype}
        self.expressions[name] = result
        return result

    def proofs(self, roots):
        expected = [(root, self.expression(root)) for root in roots]
        expected = [(root, expression) for root, expression in expected
                    if expression and "target" in expression]
        if not expected:
            return []
        proofs = []
        owned = set(self.leaf.get("buffer_names", []))
        for block in self.leaf.get("loop_ir", {}).get("blocks", []):
            nodes = block.get("nodes", [])
            operations = {node["name"]: node for node in nodes}
            if len(operations) != len(nodes):
                continue
            loop = _LoopMatcher(block.get("graph_id"), operations, self.reads)
            for store in nodes:
                args = loop.args(store)
                if (store.get("target") not in {"store", "store_reduction"}
                        or args is None or len(args) < 3 or not isinstance(args[0], str)
                        or args[0] not in owned):
                    continue
                live = loop.live(args[2])
                if live is None or not loop.valid_references(args[1:]):
                    continue
                for root, expression in expected:
                    pairs, loads = set(), set()
                    for name in live:
                        match = loop.match(expression, {"name": name, "graph_id": loop.graph_id})
                        if match is not None:
                            pairs.update(match[0])
                            loads.update(match[1])
                    grouped = defaultdict(set)
                    for fx_name, loop_name in pairs:
                        grouped[fx_name].add(loop_name)
                    for fx_name in sorted(grouped):
                        ids = _ordered(self.source_ids([fx_name]))
                        if ids:
                            proofs.append({
                                "method": "exact retained Pointwise origins and FX operands matched to connected owned-store LoopBody dataflow",
                                "execution_fx_ids": ids, "actual_fx_name": fx_name,
                                "actual_root_fx_names": [root],
                                "actual_ir_origin": {"name": fx_name, "graph_id": self.graph_id},
                                "loop_graph_id": loop.graph_id,
                                "loop_operation_names": sorted(grouped[fx_name]),
                                "input_buffers": sorted({buffer for _, buffer, _ in loads}),
                                "matched_input_loads": [dict(fx_name=name, buffer=buffer, loop_operation=operation)
                                                        for name, buffer, operation in sorted(loads)],
                                "output_buffers": [args[0]], "store_operation_name": store["name"],
                            })
        return proofs


class _LoopMatcher:
    def __init__(self, graph_id, operations, reads):
        self.graph_id, self.operations, self.reads = graph_id, operations, reads

    def resolve(self, value):
        if not isinstance(value, dict) or value.get("graph_id") != self.graph_id:
            return None
        node = self.operations.get(value.get("name"))
        if not node or node.get("graph_id", self.graph_id) != self.graph_id:
            return None
        if "target" in value and value["target"] != node.get("target"):
            return None
        return node

    def args(self, node):
        args = node.get("args", [])
        if (node.get("op") != "call_method" or not args
                or not isinstance(args[0], dict) or args[0].get("name") != "ops"
                or args[0].get("graph_id") != self.graph_id):
            return None
        return args[1:]

    def valid_references(self, value, pending=frozenset()):
        if isinstance(value, dict):
            if "name" not in value:
                return "graph_id" not in value and all(self.valid_references(item, pending) for item in value.values())
            node = self.resolve(value)
            if node is None or node["name"] in pending:
                return False
            args = self.args(node) if node.get("op") == "call_method" else node.get("args", [])
            if args is None:
                return False
            return self.valid_references([args,
                                          node.get("kwargs", {})], pending | {node["name"]})
        if isinstance(value, (list, tuple)):
            return all(self.valid_references(item, pending) for item in value)
        return True

    def live(self, value, pending=frozenset()):
        node = self.resolve(value)
        if node is None or node["name"] in pending:
            return None
        args = self.args(node)
        if args is None:
            return None
        target = node["target"]
        if target == "load":
            return {node["name"]} if (len(args) == 2 and isinstance(args[0], str) and args[0] in self.reads
                                       and self.valid_references(args[1]) and not node.get("kwargs")) else None
        if target == "constant":
            return {node["name"]} if len(args) == 2 and type(args[0]) in {int, float, bool} and _dtype(args[1]) and not node.get("kwargs") else None
        if target == "to_dtype":
            if len(args) != 2 or _dtype(args[1]) is None or set(node.get("kwargs", {})) - {"src_dtype", "use_compute_types"}:
                return None
            operands = args[:1]
        elif target == "reduction":
            if len(args) != 4 or _dtype(args[0]) is None or _dtype(args[1]) is None:
                return None
            operands = args[3:]
        elif target == "dot":
            if len(args) != 2:
                return None
            operands = args
        else:
            if target not in _ARITY or len(args) != _ARITY[target] or node.get("kwargs"):
                return None
            operands = args
        live = {node["name"]}
        for operand in operands:
            if type(operand) in {int, float, bool}:
                continue
            upstream = self.live(operand, pending | {node["name"]})
            if upstream is None:
                return None
            live.update(upstream)
        return live

    def match(self, expected, value):
        if "literal" in expected:
            actual = value
            if isinstance(value, dict):
                node = self.resolve(value)
                args = self.args(node) if node else None
                if not node or node.get("target") != "constant" or args is None or len(args) != 2:
                    return None
                if _dtype(args[1]) is None or (expected.get("dtype") and _dtype(args[1]) != expected["dtype"]):
                    return None
                actual = args[0]
            if (type(actual) not in {int, float, bool} or actual != expected["literal"]
                    or isinstance(actual, bool) != isinstance(expected["literal"], bool)):
                return None
            return frozenset(), frozenset()
        node = self.resolve(value)
        args = self.args(node) if node else None
        if node is None or args is None:
            return None
        if "buffer" in expected:
            if (node.get("target") != "load" or len(args) != 2
                    or args[0] != expected["buffer"] or args[0] not in self.reads):
                return None
            return frozenset(), frozenset({(expected["name"], args[0], node["name"])})
        if node.get("target") != expected["target"]:
            return None
        if expected["target"] == "to_dtype":
            if len(args) != 2 or _dtype(args[1]) != expected["dtype"]:
                return None
            source = node.get("kwargs", {}).get("src_dtype")
            if source is not None and (not _dtype(source) or expected["input_dtype"] and _dtype(source) != expected["input_dtype"]):
                return None
            args = args[:1]
        elif node.get("kwargs"):
            return None
        if len(args) != len(expected["args"]):
            return None
        pairs, loads = {(expected["name"], node["name"])}, set()
        for operand, actual in zip(expected["args"], args):
            if "literal" in operand:
                operand = {**operand, "dtype": expected["input_dtype"] or expected["dtype"]}
            matched = self.match(operand, actual)
            if matched is None:
                return None
            pairs.update(matched[0])
            loads.update(matched[1])
        return frozenset(pairs), frozenset(loads)


def inlined_pointwise_proofs(*, leaf, roots, fx, values, parents, graph_id, source_ids):
    """Prove selected lazy roots and ancestors on captured owned-store paths.

    Callers select roots from independently proven native split-stage inputs.
    Every computation needs an exact retained, unmaterialized Pointwise origin;
    operands must match graph-scoped LoopBody references and strong named loads.
    Unsupported expressions produce no proof. This function changes no capture.
    """
    if leaf.get("scheduler_type") == "NopKernelSchedulerNode":
        return []
    return _PointwiseMatcher(leaf=leaf, fx=fx, values=values, parents=parents,
                             graph_id=graph_id, source_ids=source_ids).proofs(roots)

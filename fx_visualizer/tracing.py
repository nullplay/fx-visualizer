"""Trace ATen modules without folding literal-only operations."""

from __future__ import annotations

import operator


_RNG_EFFECT_SCHEMAS = {
    "prims::inductor_random", "prims::inductor_randint",
    "inductor_prims::inductor_reserve_rng_state",
}


def populate_metadata(gm, inputs, mode):
    """Populate pure ATen graph metadata with strict fake execution."""
    import torch
    from torch._dynamo import config
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    unsafe = []
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        schema = getattr(node.target, "_schema", None)
        supported = node.op == "call_function" and (
            node.target is operator.getitem or (
                isinstance(node.target, torch._ops.OpOverload) and schema is not None
                and schema.name.split("::")[0] in ("aten", "prims")
            )
        )
        if (not supported or node.is_impure() or schema is not None and schema.name in _RNG_EFFECT_SCHEMAS
                or any("nondeterministic" in str(tag) for tag in getattr(node.target, "tags", ()))):
            unsafe.append(node.name)
    if unsafe:
        return {"status": "partial", "reason": "Skipped fake propagation for effects or unrecognized Python/module calls", "skipped_nodes": unsafe}
    try:
        with config.patch(fake_tensor_cache_enabled=False):
            FakeTensorProp(gm, mode=mode).propagate_dont_convert_inputs(*inputs)
    except Exception as error:
        return {"status": "partial", "reason": f"Strict fake metadata propagation failed: {type(error).__name__}", "error": str(error)}
    return {"status": "complete", "method": "strict FakeTensorProp; no fallback kernels or real tensor execution"}


def trace_module(module):
    """Record ATen overloads, including operations with only literal inputs."""
    import torch
    from torch.overrides import TorchFunctionMode

    tracer = torch.fx.Tracer()

    class RecordOverloads(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if isinstance(function, torch._ops.OpOverload):
                return tracer.create_proxy("call_function", function, args, kwargs)
            return function(*args, **kwargs)

    with RecordOverloads():
        graph = tracer.trace(module)
    graph.lint()
    return torch.fx.GraphModule(module, graph)

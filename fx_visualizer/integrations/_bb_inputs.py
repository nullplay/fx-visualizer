"""Optional input adaptation using an external Better Benchmark checkout."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _decode_entry(entry):
    from input_codec import spec_from_compact

    opts = entry[2] if len(entry) > 2 else {}
    if isinstance(entry[0], str) and entry[0] in ("tuple", "list"):
        spec = {"kind": entry[0], "items": [_decode_entry(item) for item in entry[1]], "name": opts.get("n")}
    else:
        spec = spec_from_compact(entry)
    spec.update(opts.get("x", {}))
    return spec


def load_input_configs(repro_file):
    """Use existing BB loaders, adapting only recursive compact entries."""
    from repro_harness import load_shape_configs

    path = Path(repro_file).with_name("shapes.json")
    if not path.exists():
        return load_shape_configs(str(repro_file))
    data = json.loads(path.read_text())
    points = data.get("points", [])
    needs_adapter = any(
        isinstance(entry[0], str) and (entry[0] in ("tuple", "list") or len(entry) > 2)
        for point in points for entry in point.get("inputs", [])
    )
    if not needs_adapter:
        return load_shape_configs(str(repro_file))

    from input_codec import instantiate_point
    from repro_harness import normalize_compile_policy

    configs = {}
    for point in points:
        if "inputs" not in point:
            raise ValueError("Recursive input configurations require structured entries for every point")
        leaves = []

        def flatten(entry):
            if isinstance(entry[0], str) and entry[0] in ("tuple", "list"):
                for item in entry[1]:
                    flatten(item)
            else:
                leaves.append(entry)

        for entry in point["inputs"]:
            flatten(entry)
        evaluated = instantiate_point({**point, "inputs": leaves}, data.get("symbols") or {}, guards=data.get("guards"))
        values = iter([*value[:2], original[2]] if original[0] == "S" and len(original) > 2 else value
                      for original, value in zip(leaves, evaluated))

        def rebuild(entry):
            if isinstance(entry[0], str) and entry[0] in ("tuple", "list"):
                return [entry[0], [rebuild(item) for item in entry[1]], *entry[2:]]
            return next(values)

        config = {"inputs": [_decode_entry(rebuild(entry)) for entry in point["inputs"]]}
        for key in ("alias_group_nbytes", "compile_policy"):
            if key in point:
                config[key] = point[key]
        if "compile_policy" in config:
            config["compile_policy"] = normalize_compile_policy(config["compile_policy"])
        first_model = next(iter(point.get("models", {})), "").rsplit("/", 1)[-1]
        shape_hash = point.get("shape_hash", "unknown")
        configs[f"{first_model}_{shape_hash}" if first_model else shape_hash] = config
    return configs


def make_repro_inputs(config, factory, *, geometry_only=False):
    """Keep saved geometry locally while reusing real BB input generators."""
    if config is None or "inputs" not in config:
        return factory(config)

    import torch

    leaves = []

    def flatten(spec):
        if spec.get("kind") in ("tuple", "list"):
            for item in spec["items"]:
                flatten(item)
        else:
            leaves.append(spec)

    for spec in config["inputs"]:
        flatten(spec)
    storages = {}
    if geometry_only:
        generated = [None] * len(leaves)
    else:
        flat_config = {**config, "inputs": leaves} if len(leaves) != len(config["inputs"]) or any(
            spec.get("kind") in ("tuple", "list") for spec in config["inputs"]
        ) else config
        generated = factory(flat_config)
        if len(generated) != len(leaves):
            raise ValueError(f"The input factory returned {len(generated)} values for {len(leaves)} saved inputs")

    for index, spec in enumerate(leaves):
        kind = spec.get("kind", "tensor")
        if kind == "shape":
            generated[index] = tuple(spec["dims"]) if spec.get("container") == "tuple" else list(spec["dims"])
            continue
        if kind in ("scalar", "symint"):
            generated[index] = spec["value"]
            continue
        if spec.get("layout", "torch.strided") not in ("torch.strided", "strided", torch.strided):
            raise ValueError(f"Unsupported input layout {spec['layout']}")
        shape = list(spec["shape"])
        stride = spec.get("stride")
        if stride is None:
            stride = [math.prod(max(int(dim), 1) for dim in shape[position + 1:]) for position in range(len(shape))]
        offset = int(spec.get("storage_offset", 0) or 0)
        span = 0 if math.prod(shape) == 0 else 1 + sum((size - 1) * step for size, step in zip(shape, stride))
        dtype = getattr(torch, spec["dtype"].removeprefix("torch."))
        device = spec.get("device", "cuda")
        group = spec.get("alias_group")
        tensor = generated[index]
        if geometry_only:
            if group is None:
                storage = torch.empty(offset + span, dtype=dtype, device=device)
            else:
                sizes = config.get("alias_group_nbytes") or []
                if not 0 <= group < len(sizes) or sizes[group] is None or sizes[group] < 0:
                    raise ValueError(f"Alias group {group} requires a recorded allocation size")
                if group not in storages:
                    storages[group] = torch.empty(sizes[group], dtype=torch.uint8, device=device)
                storage = storages[group].view(dtype)
            tensor = storage.as_strided(shape, stride, offset)
        elif tensor.storage_offset() != offset:
            if group is not None:
                raise ValueError(f"The input factory changed alias group {group}'s saved offset")
            storage = torch.empty(offset + span, dtype=tensor.dtype, device=tensor.device)
            with torch.no_grad():
                storage[offset:].copy_(tensor.as_strided((span,), (1,), tensor.storage_offset()))
            tensor = storage.as_strided(shape, stride, offset)
        if spec.get("is_conj") and not tensor.is_conj():
            tensor = tensor.conj()
        if spec.get("is_neg") and not tensor.is_neg():
            tensor = tensor._neg_view()
        if spec.get("requires_grad"):
            tensor.requires_grad_()
        generated[index] = tensor

    values = iter(generated)

    def rebuild(spec):
        if spec.get("kind") in ("tuple", "list"):
            items = [rebuild(item) for item in spec["items"]]
            return tuple(items) if spec["kind"] == "tuple" else items
        return next(values)

    return [rebuild(spec) for spec in config["inputs"]]

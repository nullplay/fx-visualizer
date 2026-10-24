"""Visualize repros using loaders from an external Better Benchmark checkout."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..api import visualize_graph
from ..rendering import visualize_matmul_comparison
from ..tracing import trace_module
from ._bb_inputs import load_input_configs, make_repro_inputs


_BB_MODULES = ("oracle_harness", "repro_harness", "input_codec")


@contextmanager
def _benchmark_checkout(bb_root):
    root = Path(bb_root).expanduser().resolve() if bb_root is not None else None
    if root is not None:
        missing = [name + ".py" for name in _BB_MODULES if not (root / (name + ".py")).is_file()]
        if missing:
            raise ImportError(f"bb_root={str(root)!r} is not a Better Benchmark checkout; missing {', '.join(missing)}")
        for name in (*_BB_MODULES, "numerics"):
            if name not in sys.modules or sys.modules[name] is None:
                continue
            filename = getattr(sys.modules[name], "__file__", None)
            if filename is None or Path(filename).resolve() != root / (name + ".py"):
                raise ImportError(
                    f"Better Benchmark helper {name!r} is already loaded from {filename!r}, "
                    f"which differs from bb_root={str(root)!r}. Use a fresh Python process for this checkout."
                )
    before = sys.path[:]
    try:
        if root is not None:
            sys.path.insert(0, str(root))
        try:
            for name in _BB_MODULES:
                importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name not in _BB_MODULES:
                raise
            raise ImportError(
                "The optional Better Benchmark integration needs its external checkout. "
                "Clone Better Benchmark and pass bb_root='/path/to/better-benchmark', "
                "or put that checkout on PYTHONPATH. The graph API has no Better Benchmark dependency."
            ) from error
        yield root
    finally:
        sys.path[:] = before


def visualize_repro(
    *,
    repro_path: str | Path,
    output_dir: str | Path,
    bb_root: str | Path | None = None,
    configuration: int | str | Mapping[str, Any] = 0,
    config: Mapping[str, Any] | None = None,
    title: str | None = None,
    triton_matmul: bool = False,
    clustered: bool = True,
) -> dict[str, Any]:
    """Capture scheduler boundaries for a Better Benchmark repro.

    ``bb_root`` points to an external Better Benchmark checkout. A relative
    ``repro_path`` is resolved under that checkout; the path accepts a repro
    directory or its ``repro.py``. Omit ``bb_root`` when BB is already on
    ``PYTHONPATH``. The checkout is added to ``sys.path`` only while loading
    the repro, creating inputs and tracing; the original path is restored.

    ``configuration`` selects a saved input index, key or explicit input
    configuration. ``config`` supplies compiler options; conflicts with saved
    compile policies raise before input creation. Existing BB loaders retain
    input geometry and generation behavior. Other options and returned paths
    follow ``visualize_graph``. Capture stops after scheduler fusion.
    """
    gm, inputs, policy, name = _prepare_repro(
        repro_path=repro_path, bb_root=bb_root, configuration=configuration, config=config,
    )
    return visualize_graph(
        gm=gm, example_inputs=inputs, output_dir=output_dir, config=policy,
        title=title or name, triton_matmul=triton_matmul, clustered=clustered,
    )


def visualize_repro_matmul_comparison(
    *,
    repro_path: str | Path,
    output_dir: str | Path,
    bb_root: str | Path | None = None,
    configuration: int | str | Mapping[str, Any] = 0,
    config: Mapping[str, Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Compare template/native schedules for one external BB input setting.

    Repro resolution, input selection and compiler policy follow
    ``visualize_repro``. The comparison controls ``triton.native_matmul`` and
    rejects saved or explicit overrides before input creation. Both captures
    stop after fusion; ``output_dir`` contains the combined interactive viewer
    and separate template/native captures.
    """
    gm, inputs, policy, name = _prepare_repro(
        repro_path=repro_path, bb_root=bb_root, configuration=configuration, config=config,
        comparison=True,
    )
    return visualize_matmul_comparison(
        gm=gm, example_inputs=inputs, output_dir=output_dir,
        config=policy, title=title or name,
    )


def _prepare_repro(*, repro_path, bb_root, configuration, config, comparison=False):
    with _benchmark_checkout(bb_root) as root:
        from oracle_harness import load_repro_module
        from repro_harness import compile_policy_from_config

        path = Path(repro_path).expanduser()
        if root is not None and not path.is_absolute():
            path = root / path
        path = path.resolve()
        directory = path.parent if path.is_file() else path
        module = load_repro_module(directory)
        points = load_input_configs(directory / "repro.py")
        if isinstance(configuration, Mapping):
            point = dict(configuration)
        elif isinstance(configuration, str):
            if configuration not in points:
                raise ValueError(f"Unknown saved input configuration {configuration!r}")
            point = points[configuration]
        elif isinstance(configuration, int) and not isinstance(configuration, bool):
            if configuration < 0 or configuration >= max(len(points), 1):
                raise ValueError(f"Configuration index must be in 0..{max(len(points), 1) - 1}")
            point = list(points.values())[configuration] if points else None
        else:
            raise TypeError("configuration must be a saved index, key or dictionary")
        policy = dict(compile_policy_from_config(point).get("inductor", {}))
        for key, value in (config or {}).items():
            if key in policy and (type(policy[key]) is not type(value) or policy[key] != value):
                raise ValueError(f"Compiler option {key!r} conflicts with the saved configuration's compile_policy")
            policy[key] = value
        if comparison and any(key in policy for key in ("triton.native_matmul", "triton.native-matmul")):
            raise ValueError("Matmul comparison controls triton.native_matmul; remove the saved or explicit option")

        inputs = make_repro_inputs(point, module.make_inputs)
        gm = trace_module(module.Repro())
        return gm, inputs, policy, directory.name

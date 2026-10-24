"""Public entry point for an already captured FX graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .rendering import _visualize_scheduler

if TYPE_CHECKING:
    from torch.fx import GraphModule


def visualize_graph(
    *,
    gm: GraphModule,
    example_inputs: Sequence[Any],
    output_dir: str | Path,
    config: Mapping[str, Any] | None = None,
    title: str | None = None,
    triton_matmul: bool = False,
    clustered: bool = True,
) -> dict[str, Any]:
    """Capture lowering and fusion over an already captured FX graph.

    ``gm`` is a GraphModule; ``example_inputs`` follow its forward argument
    order, retaining shapes, strides, offsets, aliases and nested containers.
    ``output_dir`` is the folder for interactive HTML, SVG and JSON artifacts.
    The returned mapping contains their absolute paths, including ``html``.

    ``config`` supplies Inductor options. ``title`` labels the viewer.
    ``triton_matmul`` requests Triton GEMM templates; ``clustered`` controls
    whether the layout groups nodes by their scheduler memberships.

    Capture always stops after scheduler fusion, before final codegen,
    autotuning and compiled execution. Counts describe a heuristic schedule
    preview. Use a dedicated process: collection patches compiler hooks.
    """
    return _visualize_scheduler(
        gm=gm, example_inputs=example_inputs, output_dir=output_dir,
        config=config, title=title, triton_matmul=triton_matmul, clustered=clustered,
    )

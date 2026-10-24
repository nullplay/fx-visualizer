"""Interactive FX DAGs with observed Inductor lowering and fusion boundaries.

Importing this package and rendering saved captures does not import PyTorch.
Live graph APIs import it only when called. Optional corpus adapters live in
``fx_visualizer.integrations``.
"""

from .api import visualize_graph
from .rendering import render_capture, render_matmul_comparison, visualize_matmul_comparison, write_index

__all__ = [
    "visualize_graph", "render_capture",
    "render_matmul_comparison", "visualize_matmul_comparison", "write_index",
]

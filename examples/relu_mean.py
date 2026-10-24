"""Visualize a captured ATen graph without a Better Benchmark checkout."""

import argparse

import torch

from fx_visualizer import visualize_graph


def forward(x, output_shape):
    activated = torch.ops.aten.relu.default(x)
    mean = torch.ops.aten.mean.dim(activated, [-1, -2], True)
    pooled = torch.ops.aten.view.default(mean, output_shape)
    mask = torch.ops.aten.le.Scalar(activated, 0)
    return pooled, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="fx_demo")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    gm = torch.fx.symbolic_trace(forward)
    inputs = (
        torch.randn(32, 1000, 13, 13, dtype=torch.bfloat16, device=args.device),
        [32, 1000],
    )
    result = visualize_graph(
        gm=gm, example_inputs=inputs, output_dir=args.output_dir,
        title="ReLU and mean",
    )
    print(result["html"])


if __name__ == "__main__":
    main()

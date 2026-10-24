"""Export interactive FX graphs from saved captures or optional corpus adapters."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .rendering import render_capture, render_matmul_comparison, write_index


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fx-visualize", description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repro", type=Path, help="Any BB directory containing repro.py, or its repro.py path")
    source.add_argument("--capture", type=Path, action="append", help="Saved capture JSON; repeat to render several graphs without PyTorch")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for HTML, SVG and JSON artifacts")
    parser.add_argument("--compare-matmul", action="store_true", help="Compare Triton template and native matmul scheduler previews")
    parser.add_argument("--configuration", default="0", help="Saved input configuration index or key for --repro")
    parser.add_argument("--bb-root", type=Path, help="External Better Benchmark checkout; relative --repro paths resolve here")
    parser.add_argument("--title", help="Display title for one graph")
    parser.add_argument("--triton-matmul", action="store_true", help="Force Triton GEMM templates for one scheduler preview")
    parser.add_argument("--inductor-config", action="append", default=[], metavar="KEY=JSON")
    args = parser.parse_args(argv)
    if args.compare_matmul and args.triton_matmul:
        parser.error("--compare-matmul controls both template and native matmul lowering")
    if args.capture:
        if args.configuration != "0" or args.triton_matmul or args.inductor_config or args.bb_root:
            parser.error("Compiler and input configuration options require --repro")
        if args.compare_matmul:
            if len(args.capture) != 2:
                parser.error("--compare-matmul requires exactly two captures: template/native")
            artifacts = [render_matmul_comparison(
                template_capture=args.capture[0], native_capture=args.capture[1],
                output_dir=args.output_dir, title=args.title,
            )]
        else:
            if args.title and len(args.capture) != 1:
                parser.error("--title requires exactly one --capture")
            artifacts, names = [], set()
            for path in args.capture:
                capture = json.loads(path.read_text())
                name = Path(capture.get("name") or path.parent.name).name
                if name in ("", ".", "..") or name in names:
                    parser.error(f"Invalid or duplicate graph name {name!r}; use separate output directories")
                names.add(name)
                destination = args.output_dir / name if len(args.capture) > 1 else args.output_dir
                artifacts.append(render_capture(capture=capture, output_dir=destination, title=args.title))
    else:
        from .integrations.better_benchmark import visualize_repro, visualize_repro_matmul_comparison

        config = {}
        for option in args.inductor_config:
            key, separator, value = option.partition("=")
            if not key or not separator:
                parser.error(f"Expected KEY=JSON, got {option!r}")
            try:
                config[key.replace("-", "_")] = json.loads(value)
            except json.JSONDecodeError as error:
                parser.error(f"Invalid JSON for {key}: {error.msg}")
        try:
            configuration = int(args.configuration) if args.configuration.lstrip("-").isdigit() else args.configuration
            options = {
                "repro_path": args.repro, "output_dir": args.output_dir,
                "bb_root": args.bb_root,
                "configuration": configuration, "title": args.title, "config": config,
            }
            if args.compare_matmul:
                artifacts = [visualize_repro_matmul_comparison(**options)]
            else:
                artifacts = [visualize_repro(**options, triton_matmul=args.triton_matmul)]
        except (ImportError, TypeError, ValueError) as error:
            parser.error(str(error))
    index = write_index(artifacts=artifacts, output_dir=args.output_dir)
    print(json.dumps({"index": str(index), "graphs": artifacts}, indent=2))

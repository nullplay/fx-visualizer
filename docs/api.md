# API reference

## Direct FX capture

```python
from fx_visualizer import visualize_graph

result = visualize_graph(
    gm=gm,
    example_inputs=example_inputs,
    output_dir="fx_demo",
    config=None,
    title=None,
    triton_matmul=False,
    clustered=True,
)
```

- `gm`: captured `torch.fx.GraphModule`. ATen graphs are the most direct input
  for interpreting lowering membership.
- `example_inputs`: inputs matching `gm.forward` argument order, including
  nested tensors, scalars and shape arguments. Inputs keep their geometry.
- `output_dir`: directory created for the viewer, metadata and compiler evidence.
  Returned artifact paths are absolute.
- `config`: optional Inductor compiler overrides.
- `title`: human-readable viewer title.
- `triton_matmul`: request Triton GEMM templates for eligible matmul operations.
- `clustered`: compact layout based on scheduler memberships.

Every argument is keyword-only. Live capture always stops after actual scheduler
fusion. There is no `mode` or `schedule_only` option. Selection is deterministic
and unbenchmarked, so the displayed schedule is a preview of fusion decisions.
No final kernel/wrapper codegen, autotuning or compiled graph execution is run.
Template candidate construction may still render candidate source during lowering.

The returned dictionary includes `html`, `svg`, `widget`, `payload`, `capture`,
`lowering_count`, `after_fusion_count`, and `fusion_region_count`.
`fusion_region_count` counts all scheduler nodes remaining after fusion,
including unchanged nodes of every class. The payload's `fusion_groups` contains
one entry per final scheduler. Each group includes its class and `matmul_fx_ids`
from computation membership. Unmapped final schedulers receive separate dashed
markers without assigning them FX members.

Casts can materialize as standalone scheduler nodes or execute inline in a
larger pointwise body. Inlined casts are associated with the scheduler that
actually loads and converts their input, including native matmul epilogues.
`fx_status[fx_id]["inlined_cast_proofs"]` retains the named input, target dtype,
exact lazy IR origin and captured load/conversion operations. The node inspector
shows the owning lowering schedulers. Matmul finalization does not count as a
second dot computation.

## Matmul comparison

```python
from fx_visualizer import visualize_matmul_comparison

comparison = visualize_matmul_comparison(
    gm=gm,
    example_inputs=example_inputs,
    output_dir="fx_demo_comparison",
    config=None,
    title="Matmul scheduler comparison",
)
print(comparison["html"])
```

For a graph containing GEMMs, this captures Triton template and native dot/reduction
lowering over the same FX graph and inputs. The Template / Native buttons switch
between the recorded schedules. Each viewer keeps independent selection and camera
state. The comparison controls `triton.native_matmul`, so conflicting explicit
overrides are rejected. Eligible `mm`, `addmm`, `bmm` and `baddbmm` use the native
path; scaled/grouped GEMMs retain their existing lowering.

## Render saved captures

```python
from fx_visualizer import render_capture, render_matmul_comparison, write_index

one = render_capture(capture="source_capture.json", output_dir="results/one")
comparison = render_matmul_comparison(
    template_capture="template/source_capture.json",
    native_capture="native/source_capture.json",
    output_dir="results/comparison",
)
index = write_index(artifacts=[one, comparison], output_dir="results/gallery")
print(index)
```

These APIs require Python and Graphviz, without importing PyTorch. Older membership
annotations are refreshed from retained compiler IR when needed. The source graph,
compiler provenance and original capture file remain intact.

```bash
fx-visualize --capture source_capture.json --output-dir results/one
python -m fx_visualizer --capture capture_a.json --capture capture_b.json \
    --output-dir results/gallery
```

## Better Benchmark adapter

```python
from fx_visualizer.integrations.better_benchmark import (
    visualize_repro,
    visualize_repro_matmul_comparison,
)

one = visualize_repro(
    bb_root="/path/to/better-benchmark",
    repro_path="repros/canonical/mean_2a13f6e9f02f",
    output_dir="fx_demo_bb",
    configuration=0,
    config=None,
)

# Use a BB repro containing matmul operations for comparison.
comparison = visualize_repro_matmul_comparison(
    bb_root="/path/to/better-benchmark",
    repro_path="repros/canonical/my_matmul_pattern",
    output_dir="fx_demo_bb_comparison",
    configuration=0,
)
```

`bb_root` points to an external BB checkout. Relative repro paths are resolved
under it; absolute paths also work. If BB helpers are already importable, it may
be omitted. The adapter restores the Python import search path after loading.
When helpers from a different checkout are already loaded, use a fresh process.

`configuration` selects a saved index, key or explicit input configuration
dictionary. Input shapes, strides, offsets, aliases and precision policies are
preserved. `config` supplies separate compiler overrides; conflicting saved
policies are rejected before creating inputs.

```bash
fx-visualize --bb-root /path/to/better-benchmark \
    --repro repros/canonical/mean_2a13f6e9f02f --configuration 0 \
    --output-dir fx_demo_bb
```

## Membership and compatibility

Boundaries show observed computation and indexing membership. Views can contribute
indexing without a separate computation scheduler node. Broad convex envelopes may
enclose unrelated FX nodes; clicking a boundary highlights its exact membership.
One FX operation can map to multiple scheduler stages. Named intermediate buffer
reads stop upstream ancestry; unresolved mixed-input stages remain `partial`.

Live capture uses PyTorch private compiler interfaces and is tested against the
local PyTorch 2.16.0a0 build. Older builds may lack required interfaces. It requires
the graph's backend, such as CUDA for CUDA inputs. Hooks and compiler options are
scoped and restored; run capture in a dedicated process when other threads might
compile concurrently. Scheduler node counts do not establish runtime kernel launches.

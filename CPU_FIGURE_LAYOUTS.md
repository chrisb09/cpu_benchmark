# Revised CPU Figure Layouts

All regenerated figures live in [figures/cpu_final](figures/cpu_final/README.md).
That index contains PNG/PDF links, measurement notes, concise caption suggestions,
data tables and the SHA256 source manifest. No source data, solver or existing
MAIA/mini-app analysis script is modified by this pipeline.

## Regeneration

From `cpu_benchmark`, using the installed plotting environment:

```sh
/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 plot_cpu_comparison.py --include-mlp-ov0 --preprocessing
/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 -m unittest -v test_plot_cpu_comparison
```

The default invocation omits the MLP OV0 and preprocessing supplementary figures.
`--output PATH` selects another output directory. Source tables always include
OV0 for audit, but it is never a fifth main model. The optional figure is under
`supplement`; its readiness figure is indexed separately under `readiness`.

## Layout Choices

- `main/*_comparison`: four model-specific triptychs; recommended detailed view.
- `main/terrain_overview`, `main/maia_overview`: two model rows by three panels;
  compact solver-specific views, each subplot with its own linear scale.
- `alternatives/latency_2x2`, `alternatives/memory_2x2`: four model-specific axes
  for presentations focused on a single metric.
- `alternatives/phases_2x2`: grouped, never stacked, provider-phase columns with
  the full descriptive external phase legend preserved from the prior concept.
- `readiness/*_comparison`: four separate triptychs; AIx reference in latency and
  memory, blocking/readiness pairs within each of four DL phase groups.
- `readiness/readiness_latency_2x2`: compact latency-only readiness alternative.
- `supplement/mlp_ov0_comparison`: optional MLP OV0 triptych.
- `supplement/preprocessing`: optional extraction/reconstruction regions, not
  an inferred whole preprocessing/postprocessing breakdown.

The full optional invocation produces 17 PNG/PDF pairs; the default produces 14.
Plot titles contain only models and concise panel headings. Longer scope and
statistical explanations live in the shared index, not in plot-level captions.
The serif type, light horizontal grids, phase colors and grouped columns follow
the terrain production phase reference. Readiness uses the prior paired-bar
concept, with hatching distinguishing readiness and RSS rather than new colors.

## Scientific Constraints

Main configurations are AIx, PhyDLL blocking, SmartSim default and three tuned
SmartSim configurations. Readiness is excluded from all main layouts.
Five independent terrain run means support a terrain-only 95% t interval
(`t=2.776445`, `df=4`); five correlated MAIA calls from one run do not.
MAIA latency uses actual rank-maximum ML-step time, not a region or inferred sum.
Memory bars are maxima without CIs: terrain sampled PSS/RSS versus MAIA kernel
cgroup whole-job lifetime peaks. GiB means bytes divided by `2**30`.

MAIA phase audits validate individual cnodes on all 96 ranks of the proper
communicator. AIx inference has seven visits/rank. Each selected PhyDLL DL node
has six visits/rank: one warmup remains after the first frame was excluded.
The input sum is divided by six frames, not twelve summed node visits. Inclusive
selected nodes must not be ancestors of one another; output includes only
`dl_send_output`, never its nested `dl_send` a second time. Whole MPI/root scopes
are not library phases. MAIA SmartSim client phases are absent and displayed as
N/A, with missing values and `unavailable_not_recorded` status in the phase CSV.
Terrain uses its existing SIMPLIFIED groups on ten validated steady steps per run.
No phase values are scaled to match uninstrumented timing.

The optional extraction scope has 35 visits/rank (five history updates per call)
and reconstruction seven. Their inclusive totals are amortized over seven calls
including warmups; per-visit ratios and actual counts are retained separately.

## Verification

The script fails closed for missing/ambiguous required regions, wrong visit or
run counts, inclusive ancestor overlap, nonfinite values, and bad render metrics.
The manifest checks 63 MAIA selected cells and 70 terrain cells in each campaign
table, hashes all consumed CSVs, and records output dimensions and panel counts.
Tests cover the CI formula, single-run CI absence, missing-data rejection, six
versus seven frames, double counting, selections, scope counts, memory semantics,
full legend labels and rendered image dimensions. Numerical model parity remains
outside this plotting task; no solver runs are launched.

The full optional invocation was rendered successfully on 2026-10-13 and all
21 tests passed. Final PNGs were visually inspected for the WaterCNN main view,
WaterCNN and MLP OV4 readiness views, both solver overviews, the standalone
full-legend phases view and the preprocessing supplement. Tall-bar headroom and
subsecond tick precision were corrected before that inspection; no visible
label/legend collisions remained in those checked layouts. Linear scaling can
make very small bars hard to see when one provider dominates; numeric labels
retain those measured values rather than changing the requested axis scale.

Final exports use tight bounding boxes with padding so external labels and
legends are not clipped. The generator was rerun after this adjustment; all
21 tests passed again. Generated images and derived audit tables are ignored
by Git under `figures/cpu_final/`; no legacy outputs were overwritten.

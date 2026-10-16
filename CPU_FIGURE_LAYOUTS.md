# Revised CPU Figure Layouts

All regenerated figures live in [figures/cpu_final](figures/cpu_final/README.md).
That index contains PNG/PDF links, measurement notes, concise caption suggestions,
data tables and the SHA256 source manifest. No source data, solver or existing
MAIA/mini-app analysis script is modified by this pipeline.

## Regeneration

From `cpu_benchmark`, using the installed plotting environment:

```sh
/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 plot_cpu_comparison.py --preprocessing
/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 -m unittest -v test_plot_cpu_comparison
```

The default invocation omits the preprocessing supplementary figure.
`--output PATH` selects another output directory. MAIA profile phases now come
from the two-warmup steady scope under
`mmcp_benchmark/results/cpu_final_steady_20261014/maia/`; MLP OV0 is not
reprofiled and is no longer a figure model.

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
- `supplement/preprocessing`: optional extraction/reconstruction regions, not
  an inferred whole preprocessing/postprocessing breakdown.
- `solver/{terrain,maia}_time_memory`: four panels in one row, ordered as
  model 1 time/memory then model 2 time/memory.
- `solver/{watercnn,giant,transformer,mlp_ov4}_phases`: four separate full-width
  model figures with rotated phase labels
  below individual bars and provider/configuration labels below each group.
  No color-only legend is needed in this layout.

The optional invocation produces 33 PNG/PDF pairs; the default produces 32.
This includes twelve individual solver panels under `solver/panels/`.

### Native Thesis Size

The thesis at `~/Master-Thesis/gitlab/thesis` uses 12pt `scrbook`, A4 and
BCOR=1cm with Latin Modern. Its compiled `thesis.log` reports a text width of
426.7914 TeX pt, or 150 mm. Solver previews are exactly that width, without
tight-bounding-box resizing. Four-panel rows allocate 36 mm per panel with 2 mm
gaps. Phase figures each use the full 150 mm. Combined time/memory previews have
embedded (a)-(d) model/metric subcaptions; phase previews identify their model
below the axis. Only the standalone exports under `solver/panels/` omit captions
and panel letters. Those exports preserve the panel widths but omit the preview
caption band, avoiding empty bottom space.
The thesis source is not modified and no LaTeX integration is generated.

Native solver panels use 7pt axis labels, 6pt ticks and 5.5-6pt numerical labels.
Configuration labels spell out SmartSim and use TPQ/INTRA pairs. Phase figures
use seconds per rank to reduce long tick labels at print size; the source and
audit tables remain in milliseconds. Other figure layouts keep their own sizes.
Plot titles contain only models and concise panel headings. Longer scope and
statistical explanations live in the shared index, not in plot-level captions.
The serif type, light horizontal grids, phase colors and grouped columns follow
the terrain production phase reference. Readiness uses the prior paired-bar
concept, with hatching distinguishing readiness rather than new colors.
Figures use reduced heights, `[TS]`/`[TBL]` model labels, `PhyDLL` in main
comparisons and `default` versus `readiness` in readiness comparisons. AIx's
single phase is labeled Forward and uses the same bar width as other phases;
its measured scope remains the inclusive AIx library call, not isolated compute.

## Scientific Constraints

Main configurations are AIx, PhyDLL blocking, SmartSim default and three tuned
SmartSim configurations. Readiness is excluded from all main layouts.
Latency whiskers are 95% t intervals (`t=2.776445`, `df=4`) over five independent
run means for both solvers: terrain run means use ten steady calls, MAIA run means
use five steady calls (inference calls 3-7). MAIA latency is the actual rank-maximum
ML-step time, not a region or inferred sum.
Memory plots show one bar per configuration, without CIs: terrain sampled PSS
versus MAIA kernel cgroup whole-job lifetime peaks. RSS remains in the audit CSV
but is not plotted. This is visual consistency, not identical measurement scope:
PSS apportions shared resident pages; cgroup peaks also include charged cache.
GiB means bytes divided by `2**30`.
Memory axes say only `Peak memory (GiB)`; measurement details remain in the index
and audit CSVs. Error-bar caps use a narrow 1.3pt half-width.

### Timing And Phase Comparability

Latency whiskers are confidence intervals of the mean, not standard deviations.
The sampling unit is an independent execution: each run first averages its steady
calls, then the five run means supply the t interval (df=4). Ranks and calls within
one execution are not counted as independent replicates. With only five runs,
normality of run means is not assured; wide intervals warrant more independent
runs rather than pooling calls or claiming a default/readiness ordering.

Phase bars are not a decomposition of adjacent plain latency bars: they average
across scope ranks, while synchronized solver latency reflects slow ranks and
additional coupling work. They also come from separate profiling executions.
MAIA has one independent profile per configuration, terrain five; no phase CIs
are drawn. More independent profiles are needed to quantify MAIA phase variability.

For Transformer default, the preserved profiling run has mean-rank Forward
28.855 s, maximum rank's five-call Forward mean 59.019 s, and synchronized
ML-step mean 60.493 s. There is one forward per DL rank per frame, not two;
the forecast window of two is not two model calls. Maximum rank means are not
the same statistic as mean per-call maxima, but reveal substantial imbalance.
For MLP ov4, the profiling jobs' synchronized ML-step means are 2.269 s default
and 2.810 s readiness, whereas the five-run plain means are 3.625 s and 2.827 s.
This documents a run-selection/order difference, not a swapped phase label.

MAIA phase audits validate individual cnodes on all 96 ranks of the proper
communicator. All MAIA providers now share one two-warmup steady scope: inference
calls 3-7, so every selected node has exactly five visits/rank and the denominator
is five, never seven AIX calls or six PhyDLL frames. Runs instrumented under
`PROFILE_SCOPE_VERSION=steady-v1` select only the `cmi_inference_steady` /
`dl_inference_steady` subtrees, so warmup calls are excluded by construction.
Inclusive selected nodes must not be ancestors of one another; output includes
only `dl_send_output`, never its nested `dl_send` a second time. Whole MPI/root
scopes are not library phases. MAIA SmartSim client phases are now recorded
(`smartsim_put_tensor`, `smartsim_run_model`, `smartsim_unpack_tensor`,
`smartsim_chunk_plan`) and are no longer N/A; they remain client scope, not
RedisAI-internal compute. Terrain uses its existing SIMPLIFIED groups on ten
validated steady steps per run with its fixed one-warmup policy.
No phase values are scaled to match uninstrumented timing.

The optional extraction scope has five visits/rank for both
`flowex_extract_cubes` and `flowex_reconstruct_output` (one per measured call).
Their inclusive totals are amortized over those five steady calls; per-visit
ratios and actual counts are retained separately.

## Verification

The script fails closed for missing/ambiguous required regions, wrong visit or
run counts, inclusive ancestor overlap, nonfinite values, and bad render metrics.
The manifest checks 63 MAIA selected cells and 70 terrain cells in each campaign
table, hashes all consumed CSVs, and records output dimensions and panel counts.
Tests cover the CI formula, run-mean CI scoping, missing-data rejection,
steady five-call scopes, double counting, selections, scope counts, memory
semantics, full legend labels and rendered image dimensions. Numerical model
parity remains outside this plotting task; no solver runs are launched.

The figures use the two-warmup steady profiles (jobs 5007387-5007401).
The revised layout suite contains 26 tests. Generated images and derived
audit tables are ignored by Git under `figures/cpu_final/`; no legacy outputs
were overwritten. Linear scaling can make very small bars hard to see when one
provider dominates; numeric labels retain those measured values rather than
changing the requested axis scale.

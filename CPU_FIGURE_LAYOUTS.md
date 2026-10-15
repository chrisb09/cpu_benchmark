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

The optional invocation produces 15 PNG/PDF pairs; the default produces 14.
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
Tests cover the CI formula, single-run CI absence, missing-data rejection,
steady five-call scopes, double counting, selections, scope counts, memory
semantics, full legend labels and rendered image dimensions. Numerical model
parity remains outside this plotting task; no solver runs are launched.

The figures were regenerated on 2026-10-14 from the two-warmup steady profiles
(jobs 5007387-5007401) and all 21 tests passed. Generated images and derived
audit tables are ignored by Git under `figures/cpu_final/`; no legacy outputs
were overwritten. Linear scaling can make very small bars hard to see when one
provider dominates; numeric labels retain those measured values rather than
changing the requested axis scale.

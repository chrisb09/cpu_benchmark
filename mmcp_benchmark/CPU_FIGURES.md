# Local CPU Figures

For the current shared column-bar layouts, use
[`../figures/cpu_final/README.md`](../figures/cpu_final/README.md) and
[`../CPU_FIGURE_LAYOUTS.md`](../CPU_FIGURE_LAYOUTS.md). They combine both solvers
in one output location, separate readiness from the main comparison, and use
provider/DL phases wherever those metrics were recorded. The directories below
retain the earlier validated extraction tables and original plot exports.

The regenerated figures use explicit successful production selections, not a
recursive scan pooling failed attempts, diagnostics, measurement modes, or ov6.
Generated PNG/PDF images and their CSV/JSON evidence remain in ignored output
directories. No legacy images are overwritten.

## m-aia

Output: `results/cpu_final_20261013/maia/` relative to this directory.

- `latency.png` / `.pdf`: Transformer ov0, MLP ov0, and MLP ov4; seven
  configurations; five steady ML calls after two warmups from one run per cell.
- `memory.png` / `.pdf`: kernel whole-job memory high-water marks in GiB,
  including startup, all processes, runtime, and charged page cache.
- `profile_diagnostics.png` / `.pdf`: disjoint whole-solver-run exclusive bins.
- `region_visit_diagnostics.png` / `.pdf`: extraction/reconstruction time
  divided by actual region visits, not inferred ML-call counts.

All 63 timing/profile/memory selections are validated. The profiling figures
are diagnostics, **not final steady inference-phase decompositions**: the
profiles do not expose a reliable ML-call parent or separate steady/warmup
scopes. They include startup, non-ML work, and warmups, and use rank-average
metrics rather than solver-rank maxima. SmartSim client profiles do not measure
RedisAI internals. A true steady-phase figure would require suitable profiling
scopes and new profiling measurements; do not estimate it by scaling these
whole-run profiles to plain timing.

## Terrain

Output: `../../mini_app/campaign_figures/terrain_cpu_complete_20261013/`
relative to this directory.

- `latency_boxplots.png` / `.pdf`: plain timing for WaterCNN and Giant MLP,
  seven configurations, five independent repetitions, ten steady calls each.
- `memory_peaks.png` / `.pdf`: maximum sampled PSS/RSS peaks across five
  repetitions, with winning source records retained independently per metric.
- `application_stacked.png` / `.pdf`: steady Score-P application phases.
- `provider_stacked.png` / `.pdf`: steady provider phases with explicit
  solver versus PhyDLL DL communicator scopes.

Timing, memory, and profile datasets each have 70 validated records. Failed and
superseded task outputs are excluded, including the original memory task 0.
Phase figures use exclusive bins under validated steady scopes, not inclusive
double counting or normalization to unprofiled latency. Terrain memory retains
Score-P instrumentation and sampled, non-atomic lower bounds; it is not the same
quantity as m-aia's kernel job-cgroup peaks. Resolution warnings remain visible
in the evidence (38 of 70 memory runs).

## Regeneration

From the CPU benchmark workspace root:

```bash
PYTHONDONTWRITEBYTECODE=1 /hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 mmcp_benchmark/regenerate_cpu_final.py
```

From the sibling `mini_app` directory:

```bash
PYTHONDONTWRITEBYTECODE=1 /hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 terrain_cpu_complete_analysis.py --refresh-local
```

Both output directories contain source manifests, data tables, and validation
notes. Ten m-aia figure tests and 43 terrain analysis/validator tests passed.
Numerical parity remains unverified; these are local analysis figures, not a
claim of final thesis-ready validation.

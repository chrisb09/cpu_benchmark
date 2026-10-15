# Terrain Second-Call Check

## Scope

Read-only check of the 70 selected final terrain plain runs: two models, seven
configurations, five independent repetitions. Selection is taken from
`../mini_app/campaign_figures/terrain_cpu_complete_20261013/timing_replicates.csv`
relative to the CPU workspace. The script validates exact model/configuration/
repetition coverage, the SHA-256 of every raw solver log against that manifest,
and all 700 selected per-call durations against the manifest's recorded values.

Each run has 11 ML calls at solver steps 2, 4, ..., 22. Current analysis excludes
call 1 at step 2. This check compares call 2 at step 4 with the mean of the nine
later calls (calls 3-11, steps 6-22) separately within each repetition.

## Results

"Median difference" is the median of five within-run percentage differences,
not a ratio of pooled samples. "Above later max" counts runs where call 2 is
slower than every later call. "Mean change" is the change in the equally weighted
mean of five run means if call 2 is removed from the current ten-call range.

| Model | Configuration | Median difference | Replicate range | Above later max | Mean change |
|---|---|---:|---:|---:|---:|
| WaterCNN | AIX | +3.49% | +1.71% to +11.50% | 5/5 | -0.54% |
| WaterCNN | PhyDLL blocking | +0.62% | -1.22% to +7.45% | 1/5 | -0.16% |
| WaterCNN | PhyDLL readiness | +2.92% | +2.00% to +19.28% | 5/5 | -0.60% |
| WaterCNN | SmartSim default | +0.20% | -0.21% to +3.84% | 1/5 | -0.10% |
| WaterCNN | SmartSim TPQ1/I96 | +5.12% | +3.39% to +12.49% | 2/5 | -0.69% |
| WaterCNN | SmartSim TPQ12/I8 | +6.53% | +4.88% to +18.17% | 0/5 | -0.83% |
| WaterCNN | SmartSim TPQ96/I1 | +27.31% | +23.16% to +47.35% | 5/5 | -3.04% |
| Giant | AIX | +0.61% | +0.37% to +0.99% | 1/5 | -0.07% |
| Giant | PhyDLL blocking | -0.24% | -0.70% to +0.17% | 0/5 | +0.02% |
| Giant | PhyDLL readiness | +0.04% | -0.57% to +0.91% | 0/5 | -0.02% |
| Giant | SmartSim default | +0.79% | -1.48% to +1.41% | 2/5 | -0.03% |
| Giant | SmartSim TPQ1/I96 | +0.72% | -0.64% to +4.59% | 0/5 | -0.16% |
| Giant | SmartSim TPQ12/I8 | +2.81% | -1.34% to +5.88% | 0/5 | -0.27% |
| Giant | SmartSim TPQ96/I1 | +99.79% | +1.40% to +144.60% | 4/5 | -7.61% |

## Interpretation

- The one-warmup policy is not uniformly supported. SmartSim TPQ96/I1 has a
  clear second-call excess for WaterCNN in all five runs and for Giant in four
  of five runs. Giant's exceptional run has only a +1.40% second-call excess.
- Giant's other configurations have relatively small second-call differences;
  excluding call 2 changes their campaign means by no more than 0.27%.
- WaterCNN AIX and PhyDLL readiness have a smaller but consistent second-call
  excess: call 2 is above all later calls in all five runs.
- Excluding two warmups would remove these second-call contributions, but does
  not establish that all subsequent calls are steady. In particular, WaterCNN
  TPQ96/I1's third call has a median normalized duration of 1.223 relative to
  each run's mean of calls 3-11. For WaterCNN TPQ12/I8, it is 1.183. Early-call
  effects may therefore extend beyond call 2 in some configurations.

Recommendation: do not describe the current one-warmup range as uniformly
validated steady state. Removing call 2 is supported for the problematic cases,
but inspect the later-call trajectories before selecting a common final cutoff;
do not simply assume m-aia's two-warmup rule suffices for terrain. Keep a common,
documented rule for comparable figures rather than selecting exclusions to
minimize each configuration's measured latency.

These are descriptive comparisons across five repetitions, not a formal
stationarity test or a diagnosis of the underlying cause. The nine later calls
within a run are not independent samples. No confidence intervals or hypothesis
tests are assigned to these warmup comparisons. The existing latency CI method
remains unchanged, and no raw data, policy, figures, or jobs were modified.
This check concerns plain timings, not the individual phase contributions in
the instrumented runs.

## Reproduce

From the CPU benchmark workspace:

```bash
PYTHONDONTWRITEBYTECODE=1 /hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 mmcp_benchmark/check_terrain_warmup.py
PYTHONDONTWRITEBYTECODE=1 /hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 -m unittest discover -s mmcp_benchmark -p 'test_check_*warmup.py' -v
```

The script also prints the five individual second-call differences and the
median normalized call-2-through-call-11 trajectory for each configuration.

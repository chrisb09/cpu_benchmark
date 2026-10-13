# m-aia CPU coupling benchmark

Replicates the terrain-solver (TS) CPU coupling campaign
(`cpu_benchmark/readiness_campaign.py`) on the m-aia solver from the MMCP
artifact. Goal: measure the per-inference ML-step latency (two-barrier max),
Score-P phases on warm inferences, and maximum sampled memory for the real TBL
transformer and the small artificial TBL MLP, across the coupling providers.

Artifact root: `/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference`
(parent repo branch `debug/current-prepost`).

## How m-aia differs from the TS campaign

| Aspect | TS | m-aia |
|---|---|---|
| Grid | synthetic, freely sized | fixed LES medium `grid_les_medium.hdf5`; interior `125x130x732` = 11.895 M cells |
| ML patch | `(B,18)` flat | `[B,5,512]`, `512 = 8^3 = mlCubeD^3` so `mlCubeD` must stay 8 |
| Samples/inference | chosen per run | ~25k cubes x 3 fields ~ **75k globally**, independent of rank count |
| Inference schedule | every 2nd step | global steps `15, 51, 101, 151, ...` (spacing 50 after 51) |
| Per-step timing | `STEP_TIMING` print | **added here** as opt-in `ML_STEP_TIMING` |
| Rank grid | `12x8` forced | automatic block decomposition |
| Warm samples | 10 (steps 4..22 even) | **10** (drop step 15, keep 51..501) |

Resolution is effectively fixed: only one grid exists and the TBL models force
`mlCubeD=8`. The tunable knobs are rank count, `mlCubeOverlap`, step count, and
model. `mlCubeOverlap > 0` is the only way to increase the sample count without a
new grid and is probed in Phase 1.

## Final parameters

- **Steps: 510.** Inferences at `15, 51, 101, ..., 501`; dropping the cold first
  inference (step 15) leaves **10 warm inferences** (`51..501`), matching TS.
  A higher inference frequency (smaller `mlInterval`/`mlStepCoefficient`) is a
  possible future optimization to shorten runs; not used here so the coupling
  semantics match the artifact.
- **Ranks:** characterization `24, 48, 96`; campaign rank = best of those
  (default 96, TS parity).
- **Reps:** 5 paired repetitions.
- **Models:** `tbl_transformer`
  (`input/transformer_inference_scripted_fw2.pt`, 732 MB) and `mmcp_test_mlp_m5`
  (1.45 MB, 361,536 params, staged into `input/`).
- **Schedule / physics (artifact smoke setup):** `hostFraction="1.00"`,
  `mlInterval=5`, `mlInputLength=5`, `mlStepCoefficient=12`, `mlForecastWindow=2`,
  `mlScalingFactor=1`, `mlInputStepDistance=1`, `mlCubeD=8`, `mlCubeOverlap=0`.

## Providers

| # | Config | Notes |
|---|---|---|
| 1 | AIX collective (default) | no oversubscription |
| 2 | PhyDLL C++ blocking | `library.solver_readiness_wait = false` |
| 3 | PhyDLL C++ readiness | `library.solver_readiness_wait = true` |
| 4 | SmartSim default | literal defaults |
| 5 | SmartSim TPQ1_I96_B96 | TS variant |
| 6 | SmartSim TPQ12_I8_B96 | TS variant |
| 7 | SmartSim TPQ96_I1_B96 | TS variant |

## Nodes and accounts

- **devel** (2 nodes, 96 cores, 244 GB, 1 h): characterization and model sweeps.
  **Default account** (no `--account`).
- **c23mm/c23ms/c23ml** (96 cores, 500 GB, 30 d): main campaign.
  **Account `rwth2150`**.

## Phases

1. **Phase 1 - workload/decomposition (devel, AIX, transformer).** Ranks
   `24,48,96`, 300 steps. Record decomposition, cells/cubes/batch per rank,
   `maxNoCells`/`scratchSize` fit, `ML_STEP_TIMING`, max memory. Probe
   `mlCubeOverlap` 0 vs 2. Gate: 48/96 must decompose with non-trivial steps.
2. **Phase 2 - model characterization (devel, AIX, MLP).** Same rank sweep.
   Establish the MLP's meaningful rank range (expect overhead-bound at 96).
3. **Phase 3 - provider smoke matrix (devel).** 7 configs x 2 models, 24 ranks,
   101 steps. Validate clean CPU inference, timing, and max memory.
4. **Phase 4 - main campaign (c23mm, `rwth2150`).** 7 configs x 2 models x 5 reps
   x 510 steps at the Phase-1 rank count. `ML_STEP_TIMING` + Score-P profiling
   (tracing off, warm inferences only) + max-PSS/RSS sampler.

## Metrics (TS parity)

| TS | m-aia |
|---|---|
| `STEP_TIMING step_ms` | `ML_STEP_TIMING step ml_ms rank_min_ms rank_max_ms` |
| 10 steady samples | 10 warm inferences (>=51) |
| Score-P phases | Score-P profiling, warm inferences |
| max sampled PSS/RSS | `mini_app/terrain_memory_sampler.py` max peaks |
| solver wall | `BENCHMARK_SOLVER_WALL_SECONDS` |

## Instrumentation

`ML_STEP_TIMING` is emitted by `maia/src/FV/fvstructuredsolver.cpp::solutionStep`
when `MAIA_ML_STEP_TIMING_PRINT=1`. The timer spans the existing two-barrier ML
region (entry barrier excluded, exit barrier included) and reports
`ml_ms = max` and `rank_min_ms` across the solver communicator. See
`instrumentation/` and the README there.

## Layout

```
mmcp_benchmark/
  README.md                 this file
  instrumentation/          m-aia ML_STEP_TIMING patch + notes
  slurm/                    sbatch drivers (characterization, campaign, build)
  mmcp_campaign.py          run/analyze driver
  configs/                  provider config variants (readiness, SmartSim TPQ)
  results/                  campaign datasets (provenance, per-run outputs)
  figures/                  rendered plots
```

## Open items

- Score-P phase coverage in m-aia/AIX not yet validated (Phase 0 gate).
- SmartSim TPQ -> m-aia DB batching threshold mapping (Phase 0).
- 48/96-rank decomposition untested.
- PhyDLL oversubscription (192 procs / 96 cores) memory on c23mm.

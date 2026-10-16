# m-aia Two-Warmup Profiling

## Selection

Replacement profiling only: Transformer overlap 0 and MLP overlap 4, with
AIX, PhyDLL blocking/readiness, and four SmartSim configurations. Fourteen
production cells; MLP overlap 0 is deliberately excluded. Existing plain
timings, memory runs, and terrain measurements remain unchanged.

Runs retain 200 solver steps and seven actual inference calls. The selected
phase scope excludes calls 1 and 2 (steps 15 and 44), and includes calls 3-7
(steps 73, 102, 131, 160, 189).

## Instrumentation

- m-aia opts into CMI's two-warmup inference policy under `WITH_SCOREP`.
- The policy counts genuine inference calls, not history updates, solver
  solution-step invocations, or returned timestep deltas.
- `cmi_inference_steady` wraps the selected CMI pipeline. It is not identical
  to the broader synchronized `ML_STEP_TIMING` region.
- AIX internal user regions remain recorded outside and inside that parent.
  Extraction must select descendants of `cmi_inference_steady`; whole-profile
  AIX totals must not be divided by five and labelled steady.
- The C++ PhyDLL client accepts `CMI_SCOREP_WARMUP_CALLS=2`; its selected parent
  `dl_inference_steady` includes receive, staging, inference, and send. Shutdown
  control is outside this parent. Unconfigured shared clients retain their
  prior one-warmup behavior.
- SmartSim detailed client regions now follow the CMI gate. They measure client
  operations, not a decomposition of RedisAI backend internals.
- `PROFILE_SCOPE_VERSION=steady-v1` selects the new runtime/verifier contract.
  Historical profiles retain their legacy verifier behavior.

## Build And Smoke

Build job **4991872** completed successfully (6m57s, exit 0:0).

Fresh isolated directories:

```text
/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/maia/build_cpu_steady_20261014_scorep
/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/maia/build_cpu_steady_20261014_phydll
```

The Score-P build uses a private AIX source worktree at the original CPU baseline
revision `86968b64380307a5113db8077ffe4b9494309c71` and a private, byte-checked copy
of its prebuilt instrumented library. Live shared AIX installs and existing
solver binaries are not rebuilt or overwritten. The CMI source revision is
`0ea8cdd70c675cbaad0005efc0cb4872aea8f6d5` plus the recorded instrumentation patch;
relative to the prior campaign revision its committed difference is the AIX
submodule pointer, overridden by the private baseline worktree for this build.

Source archives, dirty patches, file fingerprints, dependency resolution, and
binary hashes are under the build's `provenance/`. `BUILD_READY` includes:

```text
PROFILE_SCOPE_VERSION=steady-v1
PROFILE_SCOPE=cmi_inference_steady
PROFILE_WARMUP_CALLS=2
BUILD_JOB=4991872
CODE_FINGERPRINT=6ff16b657dfdded57bb57bd67d2729c361da613f6b9429f51013e781a4469de1
SOURCE_FINGERPRINT=47e42f08580d4b611047dc763f0e5df74521411163fdb1f72ffaba34a148eea2
```

Smoke job **5005051** was submitted on `c23mm`. It sequentially exercises AIX,
PhyDLL blocking, and SmartSim TPQ96/I1 with MLP overlap 4: 96 solver ranks,
80 solver steps, three calls, and exactly one selected steady call.
SmartSim DB batching remains disabled (`SS_DB_BATCH=0`).

First smoke attempt **5005051 failed**, but only in the verifier: all three
solvers exited successfully (`SOLVER_EXIT_STATUS=0`; 77 s, 230 s, 58 s) and
preserved valid `profile.cubex` outputs. The verifier imported the plotting
analysis module, which needs `pandas`, absent from the module Python. This was
a tooling defect, not a profiling or scope defect.

Fix: `verify_cpu_run.py` now parses the CUBE call tree directly through
`cube_dump` with only the standard library, so verification needs Score-P but no
pandas/numpy. Re-verifying the three preserved 5005051 runs with the new parser
passes for AIX, PhyDLL, and SmartSim.

A clean smoke was resubmitted as job **5007209** to prove the integrated
engine-plus-validator path end to end. Smoke evidence directories:

```text
mmcp_benchmark/scratch/steady_profile_5005051/   (first attempt, verifier defect)
mmcp_benchmark/scratch/steady_profile_5007209/   (clean resubmission)
```

Success requires all three engines and numeric scope verifications to pass;
the batch writes `SMOKE_PASSED` only after all three providers pass.
Production submission remains gated on smoke success.

## Steady Extraction

`mmcp_benchmark/extract_steady_profiles.py` selects only descendants of
`cmi_inference_steady` (solver ranks) and `dl_inference_steady` (PhyDLL DL
ranks), derives the denominator from the recorded parent visit count, and emits
`profile_regions.csv`/`profile_rank_metrics.csv` in a new `solver_steady`/
`dl_steady` scope. It reuses the dependency-free `cube_dump` parser. Validated
against the three preserved smoke profiles:

- AIX: 20 regions, solver frames = 1 (smoke).
- PhyDLL: 32 regions over solver and DL scopes, frames = 1 each.
- SmartSim: 11 regions, frames = 1. The previously missing detailed client
  phases are now present: `smartsim_put_tensor`, `smartsim_run_model`,
  `smartsim_unpack_tensor`, `smartsim_chunk_plan`.
- DL scope includes `dl_recv`, `dl_frame_copy`, `dl_input_*`, `dl_h2d`,
  `dl_torch_forward`, `dl_d2h`, `dl_output_reorder`, `dl_send_output`, with
  nested `MPI_Irecv`/`MPI_Waitall`/`MPI_Isend`.

The figure pipeline still consumes the old whole-run scopes (AIX 7 / PhyDLL 6
frames). Rewiring it to the steady scope with a uniform five-call denominator is
the remaining step once production profiles exist.

`extract_steady_campaign.py` batches the 14 replacement jobs: it reads each
retest log's `RUN_DIR`, requires the engine `VERIFIED` marker, and writes
combined steady tables plus `steady_cells.csv`. All completed production cells
extract at five steady visits. SmartSim top-level phases (`smartsim_put_tensor`,
`smartsim_run_model`, `smartsim_unpack_tensor`, `smartsim_chunk_plan`) are
positive, and PhyDLL solver (`phydll_prepack`/`phydll_send`/`phydll_recv`/
`phydll_unpack`) and DL (`dl_recv`/`dl_frame_copy`/`dl_torch_forward`/
`dl_send_output` and staging) phases are present at five visits each.

## Verification

The new verifier maps Cube locations explicitly to MPI ranks, checks the selected
parent visits on each rank, checks parent ancestry and static-provider scopes,
and requires positive provider-phase metrics within the selected scopes.
Expected counts are five visits per rank for production and one for the smoke.
PhyDLL separately checks solver and DL rank scopes.

Lightweight checks passed:

- CMI warmup/scheduling tests, with and without Score-P instrumentation.
- Three PhyDLL compiled-mock test groups covering frame count, shutdown, legacy
  behavior, invalid configuration, and uninstrumented compilation.
- Fifteen numeric steady-profile verifier tests.
- Six existing CPU retest contract tests.
- Shell syntax, build preflight, and diff whitespace checks.

## Production Submission

After `5005051` completes successfully and its `SMOKE_PASSED` marker exists,
preview the exact selection from the CPU benchmark workspace:

```bash
env MAIA_BUILD_DIR=/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/maia/build_cpu_steady_20261014_scorep PROFILE_SCOPE_VERSION=steady-v1 RETEST_FILTER='^retest_profile_(transformer|mlp_ov4)_' RETEST_WALLTIME=05:00:00 CASE_TIMEOUT_SECONDS=14400 bash mmcp_benchmark/slurm/submit_cpu_retest.sh profile 0 --dry-run
```

It must contain fourteen cells: ten on `c23mm` and four Transformer SmartSim
cells on `c23ms`. The old explicit source selection and plots remain unchanged
until replacement profiles pass and a steady-subtree extractor is used.

Submitted 2026-10-14 with `--dependency=afterok:5007209`, so production starts
only if the clean smoke passes:

| Job | Partition | Cell |
|---|---|---|
| 5007387 | c23mm | transformer AIX |
| 5007388 | c23mm | transformer PhyDLL blocking |
| 5007390 | c23mm | transformer PhyDLL readiness |
| 5007391 | c23ms | transformer SmartSim default |
| 5007392 | c23ms | transformer SmartSim TPQ1/I96 |
| 5007393 | c23ms | transformer SmartSim TPQ12/I8 |
| 5007394 | c23ms | transformer SmartSim TPQ96/I1 |
| 5007395 | c23mm | mlp_ov4 AIX |
| 5007396 | c23mm | mlp_ov4 PhyDLL blocking |
| 5007397 | c23mm | mlp_ov4 PhyDLL readiness |
| 5007398 | c23mm | mlp_ov4 SmartSim default |
| 5007399 | c23mm | mlp_ov4 SmartSim TPQ1/I96 |
| 5007400 | c23mm | mlp_ov4 SmartSim TPQ12/I8 |
| 5007401 | c23mm | mlp_ov4 SmartSim TPQ96/I1 |

If any job fails verification or the smoke fails, no new profiles are adopted.

## Completion (2026-10-14)

All 14 production jobs completed and verified; smoke `5007209` passed first.
`extract_steady_campaign.py` produced, into
`mmcp_benchmark/results/cpu_final_steady_20261014/maia/`:

- `profile_regions.csv`, `profile_rank_metrics.csv`: 14 cells, 262 region rows,
  25152 rank rows, all at five steady visits.
- `steady_cells.csv`: per-cell job, run, and recorded frame counts.
- `source_manifest.csv`, `latency_calls.csv`, `memory.csv`: unchanged plain
  timing and memory selections copied for a self-contained figure input.

`plot_cpu_comparison.py` now reads that directory, uses `solver_steady` /
`dl_steady` scopes with `STEADY_CALLS=5`, and no longer marks SmartSim phases
unavailable. Figures regenerated (15 PNG/PDF pairs with preprocessing) and all
21 plotting tests pass. MLP ov0 is not reprofiled and was removed from the
figure set. Plain timing, memory, and terrain measurements are unchanged.

Verified steady values match the timing trend, e.g. transformer AIX steady
inference ~19235 ms/visit against a plain steady ML-step mean of ~21084 ms.

## Latency CI Campaign

m-aia latency previously had one independent run per cell. To match the terrain
95% t CI (five independent run means, df=4, t=2.776445), 4 extra plain-timing
repetitions were submitted per cell for the two presented workloads x seven
variants: **56 jobs** (rep 1 is the existing selected run). No rebuild: the
frozen `build_cpu_retest_plain` binary is reused so every rep shares rep-1
provenance.

- Submission: `mmcp_benchmark/slurm/submit_maia_ci_reps.sh` (dry-run first).
- Job map: `results/cpu_final_steady_20261014/maia_ci_jobs.csv`.
- Analysis: `mmcp_benchmark/maia_latency_ci.py` reads each run's solver.log,
  requires the seven-call schedule and rank-maximum timings, computes a run mean
  over calls 3-7, then the mean and t CI over the five run means. Output
  `maia/maia_latency_summary.csv` and `maia/maia_latency_reps.csv`.
- Figures: `plot_cpu_comparison.py` now reads the summary for MAIA latency and
  draws the same style of CI whisker as terrain; the cell rows record five
  independent runs and their sample means.

Memory maxima and phase panels remain single-run by design.

All 56 CI jobs completed. Example five-run CIs: transformer AIX 21.28 +/- 1.32 s,
SmartSim default 1459.0 +/- 6.6 s, TPQ1/I96 1039.1 +/- 11.6 s, TPQ96/I1
27.71 +/- 0.68 s. The PhyDLL modes are genuinely variable run to run: transformer
readiness run means span 21.7-106.0 s (CI +/- 39.5 s) and blocking 57.6-73.9 s,
which the whiskers now show honestly. Figures regenerated with CI whiskers for
both solvers; the plotting suite is 22 tests.

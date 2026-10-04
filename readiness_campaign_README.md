# Native PhyDLL Readiness Campaign

This isolated extension does not replace `production_4644312`, production builds,
or default controls. Results are under `readiness_results/readiness_20261003`.
The same freshly built Score-P solver and C++ DL executable serve both policies.
`readiness_false.toml` explicitly disables readiness (`PHYDLL_CPP`);
`readiness_true.toml` enables it (`PHYDLL_CPP_READY`). Updated solver and DL
participants are required even for the disabled policy.

This repository versions the campaign tools and narrative, not generated builds,
raw datasets or full-environment provenance. Back up `readiness_results/` and
the original model/input files separately; these commits are a source backup,
not a measurement archive. Reproduction requires the separately versioned
readiness protocol and diagnostic hooks in sibling `CPP-ML-Interface` and
`mini_app`, including `terrain_cpu_diagnostics.py`, `extract_scorep_tables.py`,
`terrain_memory_sampler.py` and solver sources. Native builds use this
repository's `forward_cpu_diagnostic.hpp`. Shell wrappers and TOML paths target
the recorded RWTH installation and require its environment helpers, dependencies
and models; adjust those site-specific paths for another installation.

Completed results and job IDs are summarized in `readiness_campaign_results.md`.
Resume audit `4711504` passed on `devel`, without modifying either dataset; its
report is under `readiness_results/audit_4711504`. Both the full campaign and
the explicit-yield smoke were already complete, so no duplicate runs were needed.

## Methodology

- Two models, watercnn and giant, five matched repetitions per model and mode.
- Twenty performance runs: 22 steps, initial ML step 2 excluded, ten steady ML
  steps 4, 6, ..., 22. No memory sampler or forward diagnostic in these builds.
- Twenty separate memory runs: six steps and original one-second `/proc` sampler.
  Require simultaneous coverage and exactly 96 solver plus 96 DL processes.
- One exclusive `c23mm` node per allocation, 480 GiB requested, 96 solver and 96
  DL ranks oversubscribed and paired to physical cores `world_rank % 96`.
  A strict pin wrapper records and validates every rank, failing pin errors.
- Original CPU models, inputs, flat-contiguous layout, 12x8 decomposition,
  chunk size 12, batch size 10000000, intra/inter-op threads 1/1, parallel HDF5
  independent transfers, no MPI sync mode. Original `STEP_TIMING` sync remains
  enabled, as in `terrain_cpu_final.sbatch`; no additional timing barriers.
- PhyDLL I/O logging, I/O logging barriers, telemetry and internal memory logging
  are disabled. No C PhyDLL source or library was changed by this campaign.
- Policy order alternates by repetition (baseline first on odd reps, ready first
  on even reps). Policies run serially within one allocation; array concurrency
  is capped at two tasks per mode. No expected speedup is built into validation.
- HDF5 numerical outputs are compared per pair, excluding only `/runtime_seconds`.
  Smoke uses both models, 96+96 ranks and six steps before the full campaign.
- Metadata v4 is 88 bytes, readiness boolean at offset 76. All participants
  duplicate the control communicator after initialization. Enabled DL sends a
  sequence notification before payload send; solver posts payload receive,
  tests readiness requests with 100 us sleeps, then completes the payload.

## Build And Run

Run from `cpu_benchmark`. Parent directories must exist before building.
The scripts use existing environment helpers and dependencies, not production
solver/DL binaries. Registry generation is fresh in each isolated build.
All long build/tool work belongs in scheduler allocations, not on login. The
build wrapper uses `devel` with account `default`; runtime campaigns use exclusive
`c23mm` with account `rwth2150`. Never launch giant96 on `devel`.

```bash
sbatch --export=ALL,BUILD_KIND=scorep build_readiness_campaign.sbatch
sbatch --export=ALL,BUILD_KIND=native build_readiness_campaign.sbatch
# Wait for successful builds; runtime allocations must stay on exclusive c23mm.
sbatch --export=ALL,MODE=smoke,DATASET_ID=readiness_20261003 run_readiness_campaign.sbatch
sbatch --export=ALL,MODE=native,DATASET_ID=readiness_20261003 run_readiness_campaign.sbatch
# Submit only after successful smoke validation:
sbatch --array=0-9%2 --export=ALL,MODE=performance,DATASET_ID=readiness_20261003 run_readiness_campaign.sbatch
sbatch --array=0-9%2 --export=ALL,MODE=memory,DATASET_ID=readiness_20261003 run_readiness_campaign.sbatch
```

Use a new dataset identifier for reruns; run directories cannot be overwritten.
Each allocation writes a `results.csv`; `COMPLETE` is written only after all
mode-specific validations and numerical comparisons. The analyzer rejects
missing, failed, duplicate, or incomplete measurements rather than silently
aggregating partial evidence. Slurm status must be checked independently.

## Analysis

```bash
sbatch --dependency=afterok:<runtime-job-ids> --export=ALL,CAMPAIGN_JOBS=<comma-separated-runtime-job-ids>,DATASET_ID=readiness_20261003 finish_readiness_campaign.sbatch
```

The parser reuses `mini_app/extract_scorep_tables.py` and its callpath, rank-scope,
visit-count and application-phase sum validation. Performance archives must
contain exactly 96 steady solver ranks and 96 DL ranks with ten visits each.
No DL warmup subtraction is guessed. Enabled readiness requires 960 steady visits.
All common binary/library/model/input/registry hashes must match across runs.

`analysis/steady_steps.csv`, `runs.csv`, `timing_summary.csv` and `paired.csv`
contain latency and replicate-level paired comparisons. `scorep_callpaths.csv`
contains inclusive and exclusive times. `phases.csv` and `phase_summary.csv`
contain separate application, solver-provider and DL-provider views. Each view
attributes exclusive self time to the deepest semantic ancestor. Readiness is
separate from residual receive time, and the solver-provider phase sum must
equal inclusive `app_provider_inference`. Do not add different views together,
or add inclusive readiness to inclusive `phydll_recv`: readiness is nested.
Times are average rank milliseconds per steady step, not critical-path wall time.

`memory.csv` and `memory_summary.csv` report per-run and replicate-mean sampled
simultaneous RSS/PSS peaks, not summed per-process high-water marks. Sampling
is non-atomic, peaks are lower bounds, and RSS double-counts shared libraries.
PSS apportions pages among all sharers. Lifetime allocation cgroup peaks are not
used as configuration peaks. Memory-mode timings are not performance evidence.

The separate native giant ABBA diagnostic uses the same solver sources and
dependencies with `FORWARD_CPU_DIAGNOSTIC` and without Score-P or PMPI interposition.
It records wall/thread/process CPU clocks for `solver_recv` (including readiness
polling/sleep and payload completion) and DL forward. Nine calls per rank are
required; the first three are excluded, matching the existing diagnostic.
Its analyzer validates all 192 affinities and per-rank call counts, Torch model
and input dtype/device/layout, thread settings and original giant batch shapes.
This diagnostic is supporting evidence, not pooled with production-length runs.

## Explicit Yield Smoke

The additional dataset `readiness_results/yield_smoke_20261003` uses a fresh
isolated build in `readiness_build/yield_native`. It does not replace the
authorized baseline/readiness treatments or change their completed measurements.
For each model, all six cases run serially on one exclusive `c23mm` allocation:
explicit yield, default environment, readiness, readiness, default environment,
explicit yield. Each case uses 18 steps, nine diagnostic calls per rank, and
excludes the first three calls from the forward/receive summaries.

The baseline and explicit-yield cases both use `readiness_false.toml`. Only the
explicit-yield cases set `OMPI_MCA_mpi_yield_when_idle=1`, exported with `-x` in
both MPMD application contexts. The other cases unset that variable. Unset is
not a claim that MPI disables yielding: the diagnostic reads the effective
`mpi_yield_when_idle` control variable through MPI-T on all 192 participants,
outside the timed regions. Open MPI version and the MCA parameter inventory,
every rank's environment and binding, and all model/binary/config/input hashes
are retained. No `LD_PRELOAD` or PMPI interposer is used.

```bash
sbatch --export=ALL,BUILD_KIND=native,BUILD_DESTINATION=yield_native build_readiness_campaign.sbatch
# Submit runtime only after the isolated build succeeds:
sbatch --export=ALL,MODE=yield_smoke,DATASET_ID=yield_smoke_20261003 run_readiness_campaign.sbatch
sbatch --dependency=afterok:<smoke-job-id> --export=ALL,MODE=yield_smoke,DATASET_ID=yield_smoke_20261003,CAMPAIGN_JOBS=<smoke-job-id> finish_readiness_campaign.sbatch
```

The analyzer checks nine forward/complete-receive samples per rank, six retained
calls per rank, all bindings and explicit/effective MCA settings. Watercnn has
batch counts `{82368: 16, 85536: 32, 86112: 16, 89424: 32}`; giant retains
`{288: 32, 576: 64}`. Each batch has 18 float features, contiguous CPU storage,
and intra/inter-op threads 1/1. Numerical HDF5 equality is required across all
six cases per model, excluding only runtime seconds. Output files include
`forward_summary.csv`, per-model detailed timing CSV/JSON, and
`mpi_rank_validation.csv`. These two reversed-order observations per treatment
are smoke evidence, not five independent performance repetitions.

The completed Score-P campaign additionally verifies actual runtime policy
behavior: every enabled run has 960 steady readiness visits, while disabled
runs must have no steady readiness visits. Metadata v4 protocol and boolean
offset are recorded from the jointly rebuilt participant sources; the native
smoke does not independently dump the metadata wire payload.

## Provenance

Each job records full environment, command argv, CPU topology, Slurm allocation,
resolved libraries, SHA-256 hashes, generated registry, config and source hashes,
git HEAD/status/diff, protocol, polling interval and sampler policy. Per-run
environments and all-rank affinity records remain next to logs and HDF5 outputs.
Job logs are under `logs/readiness_JOBID.log`. No graphics or thesis files are
modified, and no commits are made.

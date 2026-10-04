# Giant Forward CPU Diagnostic

This diagnostic explains the AIx versus PhyDLL forward-time difference without
changing the number of inference processes. Production executables, libraries,
models, and input data are not rebuilt or overwritten.

This repository versions the diagnostic header, tools and documentation, not
the generated builds or raw results. Archive those separately to preserve the
recorded measurements. Rebuilding also requires the opt-in diagnostic hooks in
the sibling CMI and AIx sources listed below, plus the sibling mini_app solver
and environment/dependency installation. Those sources are versioned separately;
this repository alone is not a standalone solver distribution.

## Measurement

- One exclusive c23mm node, 96 pinned cores.
- AIx: 96 solver/inference processes, one per core.
- PhyDLL: 96 solvers plus 96 DL processes; ranks r and 96+r share core r.
- Existing `giant_cpu.pt`, `prep_giant_240x192.h5`, flat float32 tensors.
- 32 ranks have batch 288 and 64 have batch 576, mean batch 480.
- 18 solver steps, nine forwards per rank, first three forwards discarded.
- Six steady forwards per rank give 576 forward samples per run. These samples
  are correlated within a node; they are not 576 independent repetitions.
- Identical `CLOCK_MONOTONIC`, `CLOCK_THREAD_CPUTIME_ID`, and
  `CLOCK_PROCESS_CPUTIME_ID` instrumentation wraps only
  `model.forward(inputs).toTensor()`. Input vector creation, tensor slicing,
  output assignment/push, concatenation, conversion, and logging are excluded.
- Solver waiting clocks wrap exactly `phydll_recv()`.
- Metadata is captured before the first forward. Timer records are buffered in
  memory and written to per-process files at normal process shutdown, with no
  timer-related prints, barriers, or collectives between measured calls.
- Source instrumentation is enabled only by `FORWARD_CPU_DIAGNOSTIC`, which is
  absent in ordinary builds. The diagnostic DL build also suppresses per-frame
  debug prints. CMI logging is set to error in private TOML configs.
- Score-P, memory sampling, telemetry, PhyDLL I/O logging/barriers, collective
  memory logging, and step-timing synchronization are disabled.

## Build And Run

Run from the cpu_benchmark directory:

```bash
bash build_forward_cpu_diagnostic.sh > logs/forward_diag_build.log 2>&1
sbatch run_forward_cpu_diagnostic.sbatch
```

The script configures a fresh `forward_diag_build` with GCC 11.3/OpenMPI 4.1.4,
Release, `WITH_SCOREP=OFF`, and `AIX_USE_PREBUILT=OFF`. AIx is built from the
existing source worktree into that isolated build. Both forward paths use the
same existing `CPP-ML-Interface/extern/libtorch` prefix. The existing PhyDLL C
library is reused with its logging switches disabled. Registry generation and
CMI/solver/DL builds write only into the isolated build directory.

Results go to `forward_diag_results/job_JOBID`. Each run records its environment,
exact MPI command, solver log, all 96/192 actual bindings, per-process metadata
and timings, and trajectory. The job root records library resolution, input
hashes, CPU topology, Slurm allocation, raw `timings.csv`, `summary.json`, and
`summary.txt`. `logs/forward_diag_JOBID.log` records start/completion times.

```bash
../CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python \
  analyze_forward_cpu_diagnostic.py forward_diag_results/job_4680203
```

The analyzer validates rank coverage, exact forward counts, pairing, positive
clocks, CPU dtype/device, contiguous layout, tensor shape/strides, and model
parameter dtype/device before producing a summary.

## Paced Wait Follow-Up

The baseline measured approximately half a CPU core in DL forward and the other
half in solver receive. The causal follow-up keeps all counts, pinning, model,
thread settings, and binaries fixed. The only treatment is a diagnostic PMPI
interposer preloaded into solver processes, never DL processes. It replaces
solver `MPI_Waitall` with `PMPI_Testall` plus a 100-microsecond `nanosleep` between
unsuccessful polls. It applies to all solver Waitall calls, including startup;
it is not a proposed production implementation or general MPI error-handling
replacement. PhyDLL's receive uses nonblocking receives followed by Waitall.

```bash
sbatch --nodelist=n23m0150 --export=ALL,WAIT_DIAGNOSTIC=1 \
  run_forward_cpu_diagnostic.sbatch
```

This uses baseline/paced/paced/baseline order on the same node. Analyze with:

```bash
../CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python \
  analyze_forward_cpu_diagnostic.py forward_diag_results/job_4680411 \
  --runs 1_phydll 2_paced 3_paced 4_phydll
```

## Baseline Results

Job 4680203 completed with exit 0 on n23m0150 in 5m18s. All validation passed.

| Run | Forward Wall (s) | Forward Thread CPU (s) | Thread CPU / Wall |
| --- | ---: | ---: | ---: |
| AIx first | 3.554917 | 3.535380 | 0.994504 |
| PhyDLL first | 6.748026 | 3.361724 | 0.498179 |
| PhyDLL second | 6.774641 | 3.361688 | 0.496216 |
| AIx last | 3.555297 | 3.535531 | 0.994440 |

PhyDLL solver receive averaged 6.748445/6.801229 wall seconds and
3.354527/3.402667 thread CPU seconds. Process CPU agrees with thread CPU to
microsecond precision in the aggregated summaries. Actual torch thread counts
are AIx intra=1/inter=96 (unforced) and PhyDLL intra=1/inter=1. In these forwards,
the inter-op-count difference did not result in appreciable extra worker CPU.

The baseline forward slowdown is 1.902x, close to the production 1.903x
(6.676/3.508). PhyDLL does not spend twice as much CPU time executing forward;
it receives approximately half the core while its paired solver actively waits.
The approximately 5% difference in forward CPU time is a residual not explained
by these timers alone. Frequency/cache/memory effects are not directly measured.

HDF5 comparison between AIx and PhyDLL trajectories found only the expected
`/runtime_seconds` difference; numerical trajectory datasets matched exactly.
Final `h5diff --exclude-path /runtime_seconds -v` checks passed for all three
comparisons against the first trajectory in each job, including both paced
runs. The complete outputs are saved as each job's `hdf5_validation.txt`.

These are short diagnostic runs on one node, not a new production performance
estimate or multi-node statistical study. No production wait policy is changed.

## Causal Follow-Up Results

Job 4680411 completed with exit 0 on the same n23m0150 in 5m05s. All rank,
metadata, binding, clock, and sample-count validation passed.

| Run | Forward Wall (s) | Forward Thread CPU (s) | Solver Receive CPU (s) |
| --- | ---: | ---: | ---: |
| Original wait first | 6.745903 | 3.360830 | 3.353860 |
| Paced wait first | 3.665615 | 3.565085 | 0.047490 |
| Paced wait second | 3.667668 | 3.567023 | 0.047402 |
| Original wait last | 6.724236 | 3.349865 | 3.343316 |

Original wait averaged 6.735070 seconds forward wall and 3.348588 seconds
solver receive CPU. Paced wait averaged 3.666642 seconds forward wall and
0.047446 seconds solver receive CPU: a 45.56% reduction in forward wall time,
with no change to the 96 inference processes or 96 solvers. DL thread CPU/wall
rose from 49.82% to 97.26%; solver receive CPU/wall fell from 49.72% to 1.29%.
Restoring the original wait restored the slowdown, excluding simple run-order
drift as an explanation for the large treatment effect.

Paced PhyDLL is about 3.14% slower than baseline AIx (3.555107 seconds), so
this wait-only intervention removes approximately 96.5% of the original
PhyDLL-versus-AIx wall-time gap using the follow-up controls. The small
remaining difference is not isolated by this experiment; sleep/wakeup and
MPI progress overhead, scheduling, cache/memory effects, and differing torch
thread configuration remain possible contributors. Do not interpret the
residual as a measured single cause or the paced interposer as production tuning.

## Changed Files

New files in cpu_benchmark are `forward_cpu_diagnostic.hpp`,
`build_forward_cpu_diagnostic.sh`, `run_forward_cpu_diagnostic.sbatch`,
`forward_diag_pin.sh`, `forward_diag_aix.toml`, `forward_diag_phydll.toml`,
`forward_diag_paced_wait.c`, `analyze_forward_cpu_diagnostic.py`, and this README.
Generated files are under `forward_diag_build`, `forward_diag_results`, and
the three `logs/forward_diag*` logs. Source additions are limited to opt-in
instrumentation in:

- `CPP-ML-Interface/dl_clients/dl_client.cpp`
- `CPP-ML-Interface/include/provider/ml_coupling_provider_phydll.hpp`
- `CPP-ML-Interface/extern/AIxeleratorService/src/inferenceStrategy/torchInference/torchInference.cpp`

Existing unrelated modifications in mini_app, CMI, and the AIx submodule were
preserved. No mini_app source/configuration edits, production rebuilds, thesis
repository edits, or commits were performed.

Final validation also passed `bash -n` for all three shell files and
`git diff --check` in CMI and its AIx submodule. Both jobs' analyzer was rerun
with checks for identical resolved CPU libtorch and the exact 32/64 batch
distribution. The current launch script includes numerical HDF5 comparisons
automatically; for the two completed jobs these were run afterward. The
baseline launch script and pin wrapper were extended only after job 4680203
completed to support the wait-only follow-up, so its original source hashes
describe the earlier launch-script revision. Per-run `command.sh` and `run.env`
retain the actual commands and settings used.

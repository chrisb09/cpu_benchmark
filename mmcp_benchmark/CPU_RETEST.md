# m-aia CPU Retest

Use `slurm/submit_cpu_retest.sh`, which reuses `cpu_engine.sh` through one
generalized `cpu_retest.sbatch` wrapper. The older grouped campaign wrappers
are not the retest entry point. The dedicated artifact build script is
`slurm/build_cpu_retest.sbatch`; the main agent supplies the successful Slurm
build dependency and submits the matrix separately.

## Matrices

Each model case (Transformer overlap 0, MLP overlap 0, MLP overlap 4) has seven
cells: AIX, PhyDLL blocking, PhyDLL readiness, and SmartSim `SS_DEFAULT`,
`SS_TPQ1_I96_B96`, `SS_TPQ12_I8_B96`, `SS_TPQ96_I1_B96`.

| Matrix | Build | Steps | Jobs | c23mm | c23ms |
| --- | --- | ---: | ---: | ---: | ---: |
| profile | Score-P | 200 | 21 | 10 | 11 |
| memory | plain | 80 | 21 | 10 | 11 |
| timing | plain, all providers | 200 | 21 | 10 | 11 |
| Total | | | 63 | 30 | 33 |

Every job uses account `rwth2150`, one exclusive node, 96 solver ranks,
one thread per rank, and full-node memory. PhyDLL adds 96 DL-client ranks in
the same 192-rank MPI launch, with `--overcommit` and round-robin core pinning.
Transformer AIX/PhyDLL and all overlap-4 cells use `c23mm`; all others
use `c23ms`. SmartSim DB batching is always zero; `_B96` is an existing variant
name, not a DB batch-size setting.

Overlap 4 replaces overlap 6 to match the GPU campaign. Existing overlap-6
outputs are diagnostic data, not final comparison cells. No c23ml retry is used.
Set `RETEST_FILTER` to a case-name regular expression to submit only missing
cells; always inspect the corresponding `--dry-run` first.

The real logs establish calls at steps 15, 44, 73, 102, 131, 160, and 189.
Performance runs discard the first two calls and measure the final five.
The 80-step memory run includes three actual calls and reaches the first
steady-state call at step 73. Verification rejects missing/extra calls.
`parse_ml_step_timing.py` uses `WARMUP_CALLS=2` in retest metadata; historical
metadata without that field retains its previous one-call warmup treatment.

## Submission

The build script uses the checked-out artifact source (including dirty solver
edits) and creates only fresh `maia/build_cpu_retest_plain`,
`maia/build_cpu_retest_scorep`, and `maia/build_cpu_retest_phydll` directories.
It never rebuilds the legacy MAIA directories or shared PhyDLL library. Run
`bash mmcp_benchmark/slurm/build_cpu_retest.sbatch --preflight` without an
allocation to check inputs without writing. On a successful build, each MAIA
directory has a `BUILD_READY` marker; this does not certify a runtime run.
Do not reuse a partial build directory after failure. The build job requests 32
CPUs, 180 GB, and 4 hours on c23ms (`rwth2150`). The main agent handles build
and matrix submission; no jobs are submitted by this script.

Run from the workspace root, after the main agent supplies actual values:

```bash
export PLAIN_BUILD=/absolute/path/to/plain/build
export SCOREP_BUILD=/absolute/path/to/scorep/build
export PLAIN_BUILD_JOB=1234567
export SCOREP_BUILD_JOB=1234568

MAIA_BUILD_DIR="$SCOREP_BUILD" bash mmcp_benchmark/slurm/submit_cpu_retest.sh profile "$SCOREP_BUILD_JOB"
MAIA_BUILD_DIR="$PLAIN_BUILD" bash mmcp_benchmark/slurm/submit_cpu_retest.sh memory "$PLAIN_BUILD_JOB"
MAIA_BUILD_DIR="$PLAIN_BUILD" bash mmcp_benchmark/slurm/submit_cpu_retest.sh timing "$PLAIN_BUILD_JOB"
```

IDs above are placeholders, not submitted jobs. Each job gets an
`afterok:<build-job-id>` dependency. Add `--dry-run` as the third argument to
inspect the cells/resources without submitting or creating directories.
The wrapper reads the complete engine at allocation startup, then runs that
in-memory version. Edits while queued still apply; edits after startup cannot
change a running shell's input. Previously submitted wrappers are Slurm snapshots
and do not retroactively acquire this protection.
Use `RETEST_EXCLUDE_NODES` for an explicit scheduler node exclusion on retries.
`MAIA_BUILD_DIR` must be absolute but need not exist at submission time.
Use build job ID `0` to reuse an already completed build whose job has expired
from Slurm's live dependency table. This requires `BUILD_READY` and an executable
`bin/maia`, and submits without a build dependency; runtime checks still apply.
Use `RETEST_WALLTIME` to override the five-hour allocation and
`CASE_TIMEOUT_SECONDS` to override the four-hour per-cell solver timeout.
Older SmartSim transformer logs include calls lasting over 900 seconds;
the former 1800-second timeout was not sufficient for seven calls.
SmartSim now defaults to a one-hour per-request/output wait via
`SS_TIMEOUT_SECONDS=3600`. Command limits use seconds; socket replies and model
limits use milliseconds (3600000). The engine records all units in run.info.
This replaces the 600-second output wait that failed on the first Transformer
inference. The overall case timeout and Slurm walltime remain independent limits.

## Measurement And Verification

Every engine invocation has a unique preserved directory under
`scratch/cpu_bench` in the workspace (override with `SCRATCH_ROOT`). Logs are
under `mmcp_benchmark/logs`. `run.info` records the mode, model, overlap, MPI
rank counts, expected calls, build executable, exit status and experiment path.

Score-P profiling is enabled, tracing disabled. Both the solver and PhyDLL
DL-client must contain `SCOREP_InitMeasurement`; plain runs reject such
executables. AIX library lookup uses the selected build's CMake cache, avoiding
the old runner's unconditional plain-library preference. Profiles are written
to a unique `scorep` directory without overwriting. Post-run verification opens
the real CUBE4 tar/`anchor.xml` format and checks contiguous MPI ranks (96 or
192), `m-AIA-Solver::run`, and PhyDLL's `dl_torch_forward`, `dl_recv`, `dl_send`
regions. This checks recorded topology/region definitions, not each region's
per-rank nonzero metric values. Inspect the CUBE metrics for that final check.
Raw profiles include startup and both warmup calls; the timing summary alone
selects the five steady-state calls. SmartSim profiles cover the MPI solver/CMI,
not the uninstrumented Python controller or Redis/Torch server internals.

Timing and profile runs never start a memory sampler. Memory runs use the job
cgroup's kernel maximum (v1 `memory.max_usage_in_bytes`, v2 `memory.peak`) and
a two-second current-usage sampler fallback. One fresh job per cell avoids
cumulative maxima from previously executed cells. The reported peak is a
job-level high-water mark including startup, ranks, controller/server, runtime,
and charged page cache; it is not inference-only RSS or a steady-state-only
peak. Unreadable job cgroups fail the run (`MEMORY_VALID=false`); neither task
cgroups nor node-root counters are accepted as job memory.

```bash
bash -n mmcp_benchmark/slurm/cpu_engine.sh mmcp_benchmark/slurm/cpu_retest.sbatch mmcp_benchmark/slurm/submit_cpu_retest.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest mmcp_benchmark.test_cpu_retest -v
python3 mmcp_benchmark/verify_cpu_run.py /absolute/path/to/a/run
python3 mmcp_benchmark/parse_ml_step_timing.py scratch/cpu_bench --csv /path/to/timings.csv --summary /path/to/summary.csv
```

Local tests mock submission and all provider launches. Actual inference,
instrumented MPMD finalization, CUBE metrics, and kernel job-memory visibility
still require compute-node runs after the artifact builds complete. Do not
interpret an unverified/missing/failed cell as a valid result.

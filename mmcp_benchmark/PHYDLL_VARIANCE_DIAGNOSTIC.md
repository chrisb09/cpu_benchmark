# PhyDLL Variance Diagnostic

This campaign is deliberately not an input to `plot_cpu_comparison.py` or any
publication selection. No new plain timing or memory measurements are included.

## Selection

Four additional independent profiling executions per cell:

- Transformer overlap 0, PhyDLL default and readiness.
- MLP overlap 4, PhyDLL default and readiness.

Sixteen exclusive single-node jobs on c23mm, account rwth2150. Each requests 96
cores and launches 96 solver plus 96 DL ranks, retaining the existing overcommit
and CPU-binding policy. Seven calls over 200 solver steps; two warmups excluded,
five measured calls. Default/readiness submission order alternates by repetition;
Slurm may still reorder starts. Nodes are not forced to match across executions.

The isolated Score-P build is `build_cpu_steady_20261014_scorep`, build job
4991872, scope version steady-v1. Binary/runtime hashes are checked at job startup.
The engine is snapshotted at submission; shared engine/solver sources are not
edited. Models/configuration remain the original CPU benchmark selection.

## Submission

```sh
bash mmcp_benchmark/slurm/submit_phydll_variance.sh --dry-run
bash mmcp_benchmark/slurm/submit_phydll_variance.sh
```

Default destination: `diagnostics/phydll_variance_20261016/`. Submission refuses
an existing campaign directory to prevent accidentally duplicating the jobs.
The manifest `jobs.csv` records job, workload, variant, rep and absolute log path.

Submitted jobs: 5051391-5051398 and 5051400-5051407. Baseline rep 1 is the preserved
profile for each cell: 5007388, 5007390, 5007396 and 5007397. It is not a new run,
and its temporal separation from the extra profiles must be acknowledged.

## Evidence And Analysis

`metadata/JOB/` holds node topology (`lscpu`), Slurm allocation, launch settings
and startup rank affinity masks (`bind_HOST.txt`). Affinity masks are allowed CPU
sets, not samples of where a rank actually ran. They do not by themselves prove
contention or a causal placement effect. Sampling/NUMA pinning options are disabled;
the existing wrapper's binding logger is enabled. A rank mask showing a large
set is meaningful because the launch uses `--cpu-bind=none`.

`runs/` preserves solver logs, provider configs and Cube profiles. Workload shapes
can be inspected where the existing logs expose them; no per-call tensor-shape
instrumentation or timeline tracing is added to the frozen binaries.

```sh
source /hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/setup_env_claix23.sh
/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 \
  mmcp_benchmark/phydll_variance.py \
  --campaign diagnostics/phydll_variance_20261016 --allow-missing --plots
```

The initial extraction successfully read all four real baseline profiles. Pending
extras are skipped only with `--allow-missing`. Final strict analysis requires all
20 profiles (four historical plus sixteen new), each verified and with correct
rank sets and warmup scope. A dependent analysis job runs strict extraction after
all sixteen profiling jobs terminate; failed inputs cause that analysis to fail
instead of silently dropping runs.
Dependent analysis job: **5051425** (afterany of all sixteen jobs, one core,
8 GiB, 30-minute limit on c23ms).

Outputs are exclusively under `analysis/`: steady per-call synchronized timings,
per-run latency summaries, per-rank five-call phase means, rank-distribution
statistics, across-run standard deviations/CVs and five-run t intervals, rank
ECDF figures and a latency-versus-forward diagnostic. See its generated README
for field definitions and limitations. Inclusive parent/child timings overlap;
they are not additive. Rank maxima of run means are not per-call critical paths.

Profiling overhead may affect timings, and repeating profiles does not increase
the sample size of the independent plain-timing campaign. Neither run variation
nor correlation with a phase alone establishes causation. No diagnostic output
is automatically promoted into the publication data.

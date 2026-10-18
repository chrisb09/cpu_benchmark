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

## Completed Results

All sixteen new profiles and dependent analysis job 5051425 completed with exit
0:0. Strict analysis accepted all twenty profiles, including the four historical
baselines, with no missing repetitions. Publication inputs and figures were not
changed. Raw summaries/diagnostic images are under
`diagnostics/phydll_variance_20261016/analysis/`.

Across five independent profiling run means (seconds, sample SD, not CI):

| Workload | Mode | Mean | SD | CV | Range of run means |
|---|---|---:|---:|---:|---:|
| Transformer ov0 | Default | 67.562 | 15.765 | 23.3% | 58.498-95.628 |
| Transformer ov0 | Readiness | 175.226 | 135.024 | 77.1% | 65.414-410.408 |
| MLP ov4 | Default | 4.202 | 1.901 | 45.2% | 2.269-6.568 |
| MLP ov4 | Readiness | 2.991 | 1.433 | 47.9% | 1.867-5.481 |

These are profiling measurements, not replacements for the five-run plain-timing
statistics. Original baseline execution times are separated from the new campaign
in time; run/subset and node effects are not experimentally separated.

### Transformer: DL Forward Tail

The slowest readiness profile, job 5051395 on n23m0151, has synchronized ML mean
410.408 s. DL forward rank means have median 20.669 s, average 31.136 s and
maximum 409.948 s. The slowest ranks are 125 (409.948 s), 154 (265.010 s), and
99 (171.312 s). Their sum accounts for 28.3% of total rank-forward time.
In the historical readiness profile the rank median is 20.770 s and maximum
57.305 s. Thus the slow run does not represent a uniform multiplicative slowdown
of the forward rank distribution: it has an extreme tail.

The five synchronized calls in job 5051395 span 330.243-480.863 s: all are slow,
not just one exceptional call. Its solver receive maximum rank mean is 410.146 s,
consistent with the solver waiting for slow remote forward work. Forward and
receive are overlapping elapsed scopes, not independent costs to add. Aggregate
Cube data cannot prove that one particular rank governs every call.

Across the five readiness profiles the rank-average forward CV is 11.7%, much
smaller than the synchronized ML-step CV of 77.1%. The slow tail is what the
rank-averaged phase panels conceal. Even excluding the extreme job descriptively,
the other three new readiness runs average 133.435 s versus the one historical
baseline's 65.414 s; the difference is not entirely that single extreme job.

### MLP: Solver-Side Tail

The largest MLP latencies do not coincide with the largest DL forward times.
For example, default job 5051407 has ML mean 6.568 s but DL forward rank mean
0.085 s and maximum 0.174 s. Its solver CMI parent maximum rank mean is 6.124 s,
with prepack maximum 2.236 s and send maximum 2.516 s. In baseline job 5007396,
ML mean is 2.269 s, DL forward maximum 0.410 s, prepack maximum 0.443 s and send
maximum 0.759 s. The slower solver step therefore cannot be explained simply by
more expensive model execution. Packing/send and other solver-side tails matter.
The maxima may occur on different ranks; do not add them as a critical path.

### Affinity And Workload Checks

All sixteen new jobs recorded all 192 distinct ranks, each with allowed CPU mask
`0-95`. All nodes report Xeon Platinum 8468, 96 physical cores, two sockets and
eight NUMA domains. Jobs were node-exclusive but oversubscribed within the job:
96 solver plus 96 DL processes, without rank-specific pinning. Affinity masks
do not show actual CPU residence, scheduling delay or memory placement.

The solver logs contain all 480 steady rank/frame shape records per run
(96 DL ranks times five calls). Comparing tuples of total input size, batch,
field/rank and used/rank at each identical rank/frame gives exact equality
across all ten profiles of each workload, including baselines and both modes.
Transformer batches are 768 on 24 ranks and 1008 on 72 ranks; MLP batches are
4725 on 24 ranks and 5400 on 72 ranks. This verifies the logged workload sizes,
not tensor content or physical memory placement. Work is unequal between some
ranks but its logged distribution does not change between runs.

### Interpretation And Next Experiment

The evidence localizes the variation: Transformer shows remote forward
stragglers, while MLP shows substantial solver-side overhead/stragglers. It does
not establish whether CPU scheduling, MPI progress, NUMA/cache effects, profiling
perturbation or another factor causes them. Unpinned 2:1 process oversubscription
makes scheduling/contention a concrete hypothesis, not a demonstrated cause.

A controlled follow-up should compare inherited affinity with an explicit fixed
solver/DL mapping, keeping binary, node, workload and thread counts fixed and
alternating modes/treatments. Record CPU residence/scheduling and NUMA evidence
if attributing the effect to those mechanisms. Merely adding more unpinned runs
would characterize the unstable distribution better but would not isolate its
cause. No follow-up jobs have been submitted and no diagnostic measurements
have been promoted to the publication pipeline.

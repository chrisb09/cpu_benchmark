# Default Threading Diagnosis

Workload: `mmcp_transformer`, 1,000 logical inputs, 96 solver ranks, no server-side batching. Each row is an isolated `devel` job with `--mem=238G`.

## Observed Warm-Time Medians

| Configuration | TPQ | Intra | DB `set_cpus` | Solver mapping | Runs | Warm median (s) |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Implicit RedisAI, DB omitted, MPI unbound | omitted | omitted | omitted | unbound | 5 | 92.5150 |
| Implicit RedisAI, DB omitted, MPI core-bound | omitted | omitted | omitted | core-bound | 5 | 91.0279 |
| Implicit RedisAI, DB 96, MPI unbound | omitted | omitted | 96 | unbound | 5 | 95.3016 |
| Implicit RedisAI, DB 96, MPI core-bound | omitted | omitted | 96 | core-bound | 5 | 95.5359 |
| Explicit TPQ1/I1, DB omitted, MPI unbound | 1 | 1 | omitted | unbound | 5 | 90.4367 |
| Explicit TPQ1/I1, DB 96, MPI core-bound | 1 | 1 | 96 | core-bound | 5 | 94.7403 |
| Explicit TPQ1/I1, DB 1, MPI unbound | 1 | 1 | 1 | unbound | 5 | 87.7255 |
| Explicit TPQ1/I1, DB 1, MPI core-bound | 1 | 1 | 1 | core-bound | 5 | 87.1718 |
| Explicit TPQ96/I1, DB 96, MPI core-bound | 96 | 1 | 96 | core-bound | 5 | 9.1347 |
| Explicit TPQ1/I48, DB 96, MPI core-bound | 1 | 48 | 96 | core-bound | 6 | 363.7760 |

## Findings

- The implicit RedisAI configuration is performance-equivalent to explicit `THREADS_PER_QUEUE=1`, `INTRA_OP_PARALLELISM=1`, and `INTER_OP_PARALLELISM=1` for this workload. This is evidence from timing, not a direct read-out of LibTorch's internal thread-pool values.
- Setting DB CPU allocation and solver rank mapping changes the implicit configuration by about 3-4 seconds, far less than changing queue or intra-op topology.
- `TPQ=96, I=1` is about 10x faster than `TPQ=1, I=1`, showing that the dominant bottleneck for unbatched rank-local requests is the single RedisAI queue worker.
- `TPQ=1, I=48` is about 4x slower than `TPQ=1, I=1`, confirming that multi-threaded intra-op execution is harmful for these small unbatched requests even below 96 threads.
- `db.set_cpus(1)` did not slow `TPQ=1, I=1` in this measurement. A single RedisAI worker and a single PyTorch intra-op thread can run on one CPU. The diagnostic controller was launched with `--silent`, which suppressed the intended `/proc/<pid>/status` telemetry, so this sweep does not independently prove the effective Redis `Cpus_allowed_list` for DB 1 versus DB 96.

## Data Quality

- All 50 intended runs completed successfully.
- `DIAG_8_TPQ1_I48_DBCPU_96____MPI_BOUND` has one extra successful run (six rather than five), caused by overlapping submission chains during sweep recovery. `diagnosis_stats.csv` records this explicitly.
- `diagnosis_stats.csv` is generated from `diagnosis_sweep.csv` by `generate_diagnosis_stats.py`.

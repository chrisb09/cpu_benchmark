# Transformer SmartSim Socket Diagnosis

## Unit Correction

The previous timeout increase introduced a unit error: setting
`SR_SOCKET_TIMEOUT=3600` gives a 3.6-second socket wait, not one hour. Earlier
successful Transformer memory runs used the plain build with a 600000-ms socket
limit. The four later plain timing attempts failed on their first inference
after repeated Redis reply errors and the four-hour whole-case deadline.

The actual SmartRedis implementation is authoritative:
`extern/SmartRedis/include/redisserver.h` documents command seconds and socket
milliseconds; `extern/SmartRedis/src/cpp/redis.cpp` constructs socket limits
with `std::chrono::milliseconds`. Provider debug messages are not authoritative
for units. Command retry budgets are not strict whole-case wall-clock deadlines.

Corrected engine values for `SS_TIMEOUT_SECONDS=3600`:

| Limit | TOML / environment value | Unit |
| --- | ---: | --- |
| command_timeout / SR_CMD_TIMEOUT | 3600 | seconds |
| socket_timeout / SR_SOCKET_TIMEOUT | 3600000 | milliseconds |
| model_timeout / SR_MODEL_TIMEOUT | 3600000 | milliseconds |

The engine records command seconds, socket milliseconds, and model milliseconds
in run.info. Mocked launches assert the effective environment as well as the
generated TOML. Six contract tests pass, including diagnostic call counts/bounds.
There was no CPU retest engine running at correction time.

## Submitted Diagnostics (2026-10-13)

Script: `slurm/debug_transformer_smartsim.sbatch`, with `tpq96` or `default`.
Use `--plan` as the second argument for a no-allocation dry run.

| Job | Configuration | Workload | Whole-case bounds |
| --- | --- | --- | --- |
| 4962935 | TPQ96 / intra1 | 80 steps / 3 calls, then 200 steps / 7 calls | 600 s, then 900 s |
| 4962936 | Default CPU settings | 20 steps / 1 call | 2700 s |

Both request devel, default account, one exclusive 96-core node, 230G memory,
and one-hour allocation. The usable devel node is shared by the partition;
these are diagnostic data, not final production measurements. Each probe uses
the actual plain build, timing mode, and no memory sampler or Score-P.
The script snapshots the engine text at allocation startup and logs its SHA256.
Each engine invocation creates fresh run directories and its own Redis instance.
TPQ96's full probe only runs after the short probe succeeds. Both verify the
exact inference schedule; default's single call does not measure steady latency.

Logs: `logs/debug_transformer_smartsim_JOBID.{out,err}`.
Outputs: `scratch/cpu_bench/debug_socketfix_*` under the workspace root.
Check `PROBE_SUCCESS`, run.info, solver timing records, smartredis.log, and
controller.log. A scheduler success is not a proof of numerical parity.

These corrected-setting probes compare against preserved failing runs; they
are not a same-node A/B experiment reproducing the defective 3.6-second value.
Success supports the socket diagnosis; residual issues need further isolation.

## Production Gates And Outstanding Work

Existing retries now have explicit afterok dependencies:

| Production retry | Measurement | Diagnostic dependency |
| --- | --- | --- |
| 4957655 | Transformer TPQ1 profile | 4962935 |
| 4957652 | Transformer default memory | 4962936 |
| 4957653 | Transformer default profile | 4962936 |

These pending jobs load the corrected engine at runtime, without rebuilding.
Default's earlier 600-second missing-output failure predates the socket
regression, so default is independently gated on its own diagnostic. Failure
of a gate is not a successful production measurement and must be investigated.

Plain timing attempts 4952700, 4952702, 4952707, and 4952709 all failed with
exit 124. Their outputs remain preserved and excluded from final coverage.
Both diagnostic jobs completed with exit 0:0. TPQ96 verified three calls in
175 solver seconds and seven calls in 230 solver seconds. Its five steady calls
averaged 27.083 seconds. Default verified one call in 1450 solver seconds, with
1431.501 seconds in the first ML coupling call; this is not a steady-state sample.

After those gates passed, four fresh production timing replacements were
submitted on c23ms, account rwth2150, one exclusive 96-rank node each:

| Configuration | Failed attempt | Replacement |
| --- | --- | --- |
| SS_DEFAULT | 4952700 | 4966829 |
| SS_TPQ1_I96_B96 | 4952702 | 4966830 |
| SS_TPQ12_I8_B96 | 4952707 | 4966831 |
| SS_TPQ96_I1_B96 | 4952709 | 4966832 |

All use the existing plain build, 200 steps, seven calls (two warmups and five
measured), no profiler or memory sampler, a five-hour case limit within a
six-hour allocation, and SS_TIMEOUT_SECONDS=3600 with corrected socket/model
milliseconds. The completed build is reused through build-job argument 0.
New wrappers freeze the engine text at allocation startup. Output directories
are unique per job, and failed attempts remain preserved and excluded.
Existing profiling/memory retries 4957652/4957653/4957655 were not duplicated.
Submission is not runtime success; require exit 0 and verified call schedules.
The two MLP final matrices and terrain campaign are unaffected and complete at
the measurement-job level. Final parity and Score-P metric extraction remain.

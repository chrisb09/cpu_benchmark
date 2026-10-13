# CPU Measurement Completion, 2026-10-12

Final workloads: Transformer overlap 0, MLP overlap 0, MLP overlap 4.
Each has AIX, PhyDLL blocking/readiness, and four SmartSim variants.
Timing, profiling, and memory are distinct runs, not interchangeable estimates.

## Builds And Superseded Runs

Reuse successful build job 4940254: `maia/build_cpu_retest_plain` and
`maia/build_cpu_retest_scorep` in the artifact repository. Both have BUILD_READY.
Slurm accounting records COMPLETED/0:0, but the live job record has expired;
the completion submissions therefore use the checked completed-build path
(`build job 0`) without a scheduler dependency. Runtime binary checks apply.

Cancelled pending obsolete overlap-6 AIX profile/memory jobs 4940355/4940356.
Completed overlap-6 SmartSim and failed overlap-6 PhyDLL outputs are preserved
as diagnostic data, excluded from the final overlap-4 comparison. No c23ml
retry was submitted. All overlap-4 cells initially use full-node c23mm memory.

## New Submissions

| Configuration | ov4 Profile | ov4 Memory | ov4 Plain Timing |
| --- | --- | --- | --- |
| AIX | 4952697 | 4952698 | 4952723 |
| PhyDLL blocking | 4952699 | 4952701 | 4952726 |
| PhyDLL readiness | 4952704 | 4952703 | 4952727 |
| SS_DEFAULT | 4952706 | 4952705 | 4952728 |
| SS_TPQ1_I96_B96 | 4952710 | 4952708 | 4952729 |
| SS_TPQ12_I8_B96 | 4952713 | 4952714 | 4952730 |
| SS_TPQ96_I1_B96 | 4952717 | 4952715 | 4952731 |

| Plain Timing | Transformer | MLP ov0 |
| --- | --- | --- |
| AIX | 4952696 | 4952712 |
| SS_DEFAULT | 4952700 | 4952716 |
| SS_TPQ1_I96_B96 | 4952702 | 4952718 |
| SS_TPQ12_I8_B96 | 4952707 | 4952719 |
| SS_TPQ96_I1_B96 | 4952709 | 4952721 |

These 31 jobs close the replacement workload and plain-timing gaps without
duplicating existing patched PhyDLL timings: Transformer blocking/readiness
4940320/4940323 completed; MLP ov0 4940326/4940329 remain queued.
Existing queued Transformer SmartSim and MLP ov0 profile/memory jobs remain
unchanged. Old plain timings do not substitute for these patched timing jobs.

## Terrain

The original patched PhyDLL performance/memory arrays 4940251/4940252 succeeded.
Additional terrain submissions and smoke gates are recorded in
`../../mini_app/terrain_cpu_missing_20261012_README.md` and its campaign jobs CSV.
They cover 30 missing memory measurements and 70 true plain timing
measurements, with five repetitions for each applicable model/configuration.
The plain terrain build is non-Score-P; memory retains the previous instrumented
treatment. Terrain sampled RSS/PSS and m-aia kernel job-cgroup peaks measure
different quantities and must be labelled accordingly.

## Validation Still Required

Submission is not completion. Check terminal status, verified call schedules,
profiles and valid memory before accepting each cell. Numerical parity against
pre-patch outputs remains unverified. Extract and inspect Score-P metric values
before presenting phase tables; topology/region-definition checks alone do not
validate metric values. SmartSim profiles cover solver/CMI, not RedisAI internals.
Phase totals include warmups unless the extraction explicitly excludes them;
plain timing uses two warmups and five measured inference calls.

Tests: five local retest contract tests pass, including overlap-4 cell counts,
partition routing, filtered submission, mocked launches, and profile rejection.

## Transformer SmartSim Timeout Retries

Jobs 4940330, 4940331, and 4940334 failed on missing output tensors during
the first inference after roughly 600 seconds. No OOM was recorded; the wait
limit is the likely cause, not a confirmed RedisAI diagnosis. Engine defaults
now use 3600 seconds for command/socket limits and 3600000 milliseconds for
model/output polling, with both recorded in metadata. Pending jobs invoke the
engine at runtime and inherit the new default without duplicate submissions.

| Replacement | Failed Job | Measurement / Configuration |
| --- | --- | --- |
| 4957653 | 4940330 | Transformer SS_DEFAULT profile |
| 4957652 | 4940331 | Transformer SS_DEFAULT memory |
| 4957655 | 4940334 | Transformer SS_TPQ1_I96_B96 profile |

Retries request six-hour walltime and a five-hour whole-case timeout. The
memory retry excludes n23m0348 due to the separate unreadable-counter issue.
Original failed results remain preserved and excluded; retries are not valid
measurements until runtime verification succeeds. Tests check timeout units
in generated SmartSim configuration and recorded metadata.

## Clean Measurement Retries (2026-10-13)

New wrappers read the entire engine at allocation startup and execute the
in-memory text. Queued engine edits still apply, while running jobs no longer
stream a mutable file. Old queued wrappers remain Slurm snapshots; avoid editing
the shared engine while any of those old wrappers are running.

| Replacement | Failed Job | Measurement / Configuration |
| --- | --- | --- |
| 4958563 | 4940345 | MLP ov0 PhyDLL readiness profile |
| 4958564 | 4940350 | MLP ov0 SS_TPQ1_I96_B96 memory |
| 4958565 | 4952705 | MLP ov4 SS_DEFAULT memory |
| 4958566 | 4952708 | MLP ov4 SS_TPQ1_I96_B96 memory |

The three memory retries explicitly exclude n23m0348 and n23m0011. The same
exclusions were applied to pending memory jobs 4957652 and 4940354. This avoids
the observed unreadable-counter nodes; it does not resolve their underlying
cgroup layout/permissions. All failed original outputs remain preserved.
Five contract tests passed, now exercising the startup-loading wrapper and
explicit scheduler node exclusions. The engine was not edited during this fix.

# Readiness Campaign Results

The retained campaign and the additional explicit-yield smoke completed on
exclusive `c23mm` allocations. No duplicate runtime or build jobs were submitted
when resuming. This report describes two separate datasets, not pooled results.

## Authorized Performance Campaign

Dataset: `readiness_results/readiness_20261003`.
Twenty performance runs and twenty separate memory runs cover watercnn and giant,
five paired repetitions of `PHYDLL_CPP` versus `PHYDLL_CPP_READY` per model/mode.
Both policies use the same isolated, updated Score-P binaries and protocol.
Performance runs use 22 steps and ten steady ML steps; memory runs use six steps
with the one-second sampler. All runs use 96 solver plus 96 DL ranks, paired
core pinning, original CPU models/inputs, and Torch intra/inter-op threads 1/1.

| Model | Baseline Steady Step (ms) | Readiness Steady Step (ms) | Reduction |
| --- | ---: | ---: | ---: |
| watercnn | 315.694 | 228.219 | 27.7% |
| giant | 8021.656 | 4321.369 | 46.1% |

These are means of five replicate means from solver `STEP_TIMING`, not DL
forward times. Replicate standard deviations are 4.843/3.194 ms for watercnn
baseline/readiness and 89.233/36.682 ms for giant. Every paired performance
repetition favors readiness; no statistical significance claim is inferred.

Score-P DL forward averages change from 213.169 to 161.007 ms for watercnn and
6675.961 to 3631.602 ms for giant. In the disjoint solver-provider view, enabled
readiness owns 178.269/3632.121 ms, leaving 1.908/1.441 ms residual receive time
for watercnn/giant. Readiness is nested under receive: do not add its inclusive
duration to inclusive receive. Different application/provider views must not be
summed together. These phase times are average-rank time, not critical-path time.

Mean sampled simultaneous total PSS peaks are approximately 82.3/84.4 GiB for
watercnn and 259.4/258.8 GiB for giant (baseline/readiness). The sampler validates
96 solver plus 96 DL processes and simultaneous coverage. Peaks are non-atomic
sampled lower bounds, not lifetime cgroup peaks; memory-run latency is not
performance evidence. Exact bytes and separate solver/DL RSS/PSS are in
`analysis/memory_summary.csv` and `analysis/memory.csv`.

## Explicit Yield Smoke

Dataset: `readiness_results/yield_smoke_20261003/yield_smoke/job_4706084_0`.
Open MPI 4.1.4; native diagnostic binaries without Score-P or PMPI interposition.
Each model uses the serial reversed order yield1, default, readiness, readiness,
default, yield1 on the same exclusive node. Each case has nine diagnostic calls
per rank, with the first three excluded: 576 retained samples per role/case.
Default means the MCA environment variable is unset, not that yielding is off.
Every explicit-yield rank receives `OMPI_MCA_mpi_yield_when_idle=1`.
The independent audit confirms MPI-T reports effective yield `1` in all 2304
rank/case records, including unset-default and readiness cases. Thus this smoke
compares an explicit setting against an already-active default, not yield on
versus yield off.

| Model | Treatment | DL Forward Wall (s) | Total Solver Receive Wall (s) | Solver Receive Thread CPU (s) |
| --- | --- | ---: | ---: | ---: |
| watercnn | Default | 0.222646 | 0.247997 | 0.123625 |
| watercnn | Yield1 | 0.220796 | 0.246394 | 0.122867 |
| watercnn | Readiness | 0.159758 | 0.177610 | 0.003259 |
| giant | Default | 6.747373 | 6.747806 | 3.354884 |
| giant | Yield1 | 6.745673 | 6.746066 | 3.353807 |
| giant | Readiness | 3.673442 | 3.674573 | 0.049071 |

Values average the two reversed-order observations per treatment. Complete
receive timing includes readiness polling/sleep and payload completion. Process
CPU results closely match thread CPU and are retained in `forward_summary.csv`.
Explicit yield does not resolve the contention seen in this smoke: baseline and
yield1 consume about half a core while waiting, whereas readiness releases most
of that CPU time. These are short same-node observations, not five independent
repetitions. The authorized experiment was not replaced by a yield treatment.
The full campaign completed before the additional yield smoke; its measurements
were retained rather than rerun to change that chronology.

## Jobs And Validation

- Full-rank two-model Score-P smoke: `4696303`, completed, exit `0:0`.
- Native giant ABBA diagnostic: `4696631`, completed, exit `0:0`.
- Performance array: `4696726_0-9`, all completed, exit `0:0`.
- Memory array: `4696727_0-9`, all completed, exit `0:0`.
- Campaign analysis: `4696962`, completed, exit `0:0`.
- Two-model yield smoke: `4706084`, completed, exit `0:0`.
- Yield analysis: `4706360`, completed, exit `0:0`.
- Independent resume audit: `4711504`, completed on `devel`, exit `0:0`, 4m44s.

Individual array task job IDs differ from the parent array IDs and are recorded
in each job directory's provenance. Successful original analyzers require exact
run coverage, solver/DL steady rank coverage, ten steady calls per performance
rank, 960 enabled readiness visits and no disabled readiness visits. The runners
validate every rank's affinity and numerical HDF5 equality, excluding only
`/runtime_seconds`. The yield analyzer also checks 2304 rank/case MCA records,
rank environments, nine raw calls per rank, original batch shapes, CPU float
contiguous model input, and threads 1/1.

The resume audit is read-only and scheduled on `devel`; it does not launch an MPI
benchmark or rebuild artifacts. Its independent current artifact hashes,
reparsed Score-P coverage, repeated HDF5 comparisons, exact task accounting and
yield rank validation are written to `readiness_results/audit_JOBID`. A
`COMPLETE` marker there is required before claiming the independent audit passed.
The completed report is `readiness_results/audit_4711504`: it verifies 32 retained
binary/registry/library/model/input artifacts, all 60 runtime cases across the
five modes, all repeated numerical comparisons, and 2304 yield rank/case records.
Its `slurm_status.txt` independently confirms all 23 runtime allocations/task IDs
completed successfully on `c23mm`; `validation.json` records exact coverage and
`yield_summary.csv` records treatment-level wall/thread/process CPU means.

## Files And Commands

`readiness_campaign_README.md` documents the runner, protocol and methodology.
`analysis/timing_summary.csv`, `paired.csv`, `steady_steps.csv`, `runs.csv`,
`scorep_callpaths.csv`, `phases.csv`, `phase_summary.csv`, `memory.csv` and
`memory_summary.csv` contain the full campaign tables. The yield job contains
`forward_summary.csv`, per-model timing CSV/JSON, `mpi_rank_validation.csv`,
Open MPI version/MCA inventory, commands, all-rank environments, HDF5 outputs,
libraries and SHA-256 provenance. Logs are under `logs/`.

```bash
# Independent read-only verification of the retained completed datasets:
sbatch --parsable audit_readiness_campaign.sbatch
# Future isolated builds only, not needed to duplicate this campaign:
sbatch --export=ALL,BUILD_KIND=scorep,BUILD_DESTINATION=scorep_next build_readiness_campaign.sbatch
sbatch --export=ALL,BUILD_KIND=native,BUILD_DESTINATION=yield_native_next build_readiness_campaign.sbatch
```

Use a fresh build destination/dataset for future changes. Never run giant96 on
`devel`; its memory capacity is insufficient. Runtime scripts remain exclusive
`c23mm` with 480 GiB requested. `devel` uses account `default`, while runtime
campaigns use `rwth2150`. Only short inspections/submissions run on login.
No production artifacts, prior datasets, C PhyDLL implementation, figures or
thesis files were modified during this resume, and no commits were created.

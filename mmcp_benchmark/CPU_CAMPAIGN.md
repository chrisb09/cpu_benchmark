# m-aia CPU Benchmark Campaign

Full provider × model × overlap matrix for the m-aia solver CPU coupling benchmark, replicating the terrain-solver CPU campaign methodology.

## Campaign matrix (96 ranks, 200 steps, compact schedule)

| Job | Partition | Model | Provider | Config | Overlap | Est. wall |
|---|---|---|---|---|---|---|
| 1 | c23mm | Transformer | AIX + PhyDLL | AIX, PhyDLL-blocking, PhyDLL-readiness | 0 | ~25 min |
| 2 | c23ms | Transformer | SmartSim | SS_DEFAULT, TPQ1_I96_B96, TPQ12_I8_B96, TPQ96_I1_B96 | 0 | ~25 min |
| 3 | c23ms | MLP | AIX | — | 0 | ~3 min |
| 4 | c23ms | MLP | PhyDLL | blocking, readiness | 0 | ~8 min |
| 5 | c23ms | MLP | SmartSim | 4 variants | 0 | ~15 min |
| 6 | c23ms | MLP | AIX | — | 6 | ~4 min |
| 7 | c23mm | MLP | PhyDLL | blocking, readiness | 6 | ~10 min |
| 8 | c23mm | MLP | SmartSim | 4 variants | 6 | ~20 min |

**Total:** 8 sbatch jobs, 25 individual runs, ~110 min if serial (much less if concurrent).

> **Update (first run):** overlap-6 uses ~160 GB for AIX alone (gather buffers), so
> ov6 PhyDLL and ov6 SmartSim exceeded the 244 GB c23ms nodes and were moved to
> **c23mm** (480 GB). ov6 AIX still fits c23ms and stayed there.

**PhyDLL oversubscription:** all PhyDLL jobs allocate `--ntasks=96 --exclusive` (one full
node) and launch 96 solver + 96 DL = 192 ranks via `srun --overcommit` inside the
allocation, matching `slurm/run_phydll_cpp_300_devel.sh`. c23ms/c23mm have
`OverSubscribe=NO`, so the extra ranks must come from `--overcommit`, not from a
larger `--ntasks`. (devel has `OverSubscribe=FORCE:2` but currently only one node.)


## Quick start

```bash
cd /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/mmcp_benchmark
bash slurm/submit_cpu_campaign.sh
```

Monitor:
```bash
squeue -u $USER -o "%.10i %.12j %.8T %.10M %.6D %R"
```

Results land in `scratch/cpu_bench/<case_name>_<jobid>_<hash>/`, each with:
- `run.info` — metadata (provider, model, ranks, steps, overlap, batch, memory peak, wall time)
- `solver.log` — solver output with `ML_STEP_TIMING` lines
- `properties.toml` — patched solver config
- `config_{aix,phydll,smartsim}.toml` — patched provider config

## Engine

`slurm/cpu_engine.sh` — single-node CPU benchmark engine supporting:
- **Providers**: AIX (in-process), PhyDLL (2× ranks, blocking/readiness), SmartSim (orchestrator + 4 TPQ/intra/bind variants)
- **Models**: transformer (`input/transformer_inference_scripted_fw2.pt`), MLP (`input/mmcp_test_mlp_m5_cpu.pt`)
- **Overlap**: 0 (default) or 6 (64× cubes, stride 2)
- **Compact schedule**: `solutionInterval=10000`, `RUN_STEPS=200` → 7 inferences (2 warmup, 5 measured)
- **Instrumentation**: `MAIA_ML_STEP_TIMING_PRINT=1` → `ML_STEP_TIMING step=... ml_ms=...` lines
- **Memory sampling**: cgroup `memory.current` sampled every 2 s → `MEMORY_PEAK_BYTES` in `run.info`

Env knobs:
```bash
PROVIDER={AIX,PHYDLL,SMARTSIM}
SOLVER_RANKS=96
MODEL=<path>
ML_CUBE_OVERLAP={0,6}
RUN_STEPS=200
SOLUTION_INTERVAL=10000
ML_STEP_TIMING_PRINT=1
MAIA_AIX_BATCHSIZE={50000,1000000}
PHYDLL_MODE={blocking,readiness}
SS_CONFIG={SS_DEFAULT,SS_TPQ1_I96_B96,SS_TPQ12_I8_B96,SS_TPQ96_I1_B96}
SS_DB_BATCH=0
```

## Analysis

Parse all runs:
```bash
cd /hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference
python3 scripts/analyze_cmi_bench.py scratch/cpu_bench \
    --out /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/mmcp_benchmark/results/cpu_campaign_summary.csv
```

The analyzer extracts per run:
- Model label, overlap, solver ranks, provider/config
- Per-inference ML-step makespan (warmup vs measured, median/mean)
- Memory peak (cgroup), solver RAM total
- Wall time, exit status

## Conventions (TS parity)

- **Compact schedule**: `solutionInterval=10000` (output disabled) → inference spacing 29 steps vs 50 (default)
- **Warmup**: first 2 inferences (steps 15, 44) are warmup; steps 73, 102, 131, 160, 189 are measured
- **ML_STEP_TIMING semantics**: entry barrier excluded, exit barrier included, solver communicator only → `ml_ms` = max per-rank region time (slowest rank, which the exit barrier waits for)
- **SmartSim batch**: `batch_size=0` (DB batching disabled), matching TS campaign baseline
- **AIX batch**: transformer 50k, MLP 1M (from GPU campaign)
- **Memory**: cgroup peak (whole job, includes page cache) + solver RAM total (sum of per-rank RSS)

## Implementation notes / gotchas

- **PhyDLL oversubscription**: c23ms/c23mm have `OverSubscribe=NO`, so `--ntasks` must stay at 96
  (the node's core count); the extra DL ranks come from `srun --overcommit` inside the allocation.
  The engine launches `np_solver + np_dl` ranks (default `np_dl = np_solver`) via a `--multi-prog`
  conf, matching `run_phydll_cpp_300_devel.sh` and the CPU devel smokes (24 solver + 24 DL).
- **PhyDLL field count**: the m-aia `uniform_chunks` provider requires `PHYDLL_DL_COUNT=2` and
  `PHYDLL_DL_FIELD_COUNT=2` (the terrain value of 1 aborts with "expects dl_count=2").
- **SmartSim timeouts**: the CMI devel smokes require 600 s Redis timeouts; the artifact
  `config_smartsim.toml` ships 60 s, which makes the (slower) transformer inference time out and
  the output keys come back empty. The engine rewrites `command_timeout`/`socket_timeout`/
  `model_timeout` to `600000` ms.
- **cgroup memory (compute nodes are cgroup v1)**: the job peak is
  `/sys/fs/cgroup/memory/slurm/uid_<uid>/job_<jobid>/memory.max_usage_in_bytes`; the engine reads
  that kernel-tracked peak and also samples `memory.usage_in_bytes` every 2 s.
- **Slurm spool**: sbatch scripts derive `project_dir` from `SLURM_SUBMIT_DIR`, not `$0`
  (Slurm copies the script to `/var/spool/slurm/...`, so `$0` is not the submit path).
- **Memory request**: c23ms jobs use `--mem=230G` so they can land on both the 244 GB and
  500 GB node types (with `--mem=0` Slurm demands the full 500 GB and scheduling stalls);
  c23mm jobs use `--mem=0` (all nodes 480 GB).
- **Error handling**: campaign sbatch scripts use `set -uo pipefail` (no `-e`) so one failing
  variant does not abort the remaining variants in the same job.


## Memory estimates (96 ranks)

| Config | Model | Overlap | Est. total | Partition |
|---|---|---|---|---|
| AIX | Transformer | 0 | ~280 GB | c23mm (480 GB) |
| PhyDLL | Transformer | 0 | ~290 GB | c23mm |
| SmartSim | Transformer | 0 | ~110 GB | c23ms (238 GB) |
| AIX | MLP | 0 | ~60 GB | c23ms |
| PhyDLL | MLP | 0 | ~120 GB | c23ms |
| SmartSim | MLP | 0 | ~60 GB | c23ms |
| AIX/PhyDLL/SS | MLP | 6 | ~110–140 GB | c23ms |

AIX and PhyDLL load the model per rank (in-process or per DL rank); SmartSim shares one DB model.

## Files

- `slurm/cpu_engine.sh` — benchmark engine (AIX, PhyDLL, SmartSim)
- `slurm/cpu_campaign_c23mm_transformer.sbatch` — job 1 (c23mm transformer AIX+PhyDLL)
- `slurm/cpu_campaign_c23ms_transformer_ss.sbatch` — job 2 (c23ms transformer SmartSim 4 variants)
- `slurm/cpu_campaign_c23ms_mlp_ov{0,6}_{aix,phydll,ss}.sbatch` — jobs 3–8 (c23ms MLP matrix)
- `slurm/submit_cpu_campaign.sh` — master submission script (all 8 jobs)
- `../instrumentation/README.md` — `ML_STEP_TIMING` instrumentation details
- `parse_ml_step_timing.py` — extract ML_STEP_TIMING lines (drops step 15 warmup for legacy compat)
- `plot_characterization.py` — render latency vs ranks (validated with Phase 1/2 data)

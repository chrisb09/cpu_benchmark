# m-aia `ML_STEP_TIMING` instrumentation

Adds a per-inference ML-step timing line to the m-aia solver, matching the
terrain-solver `STEP_TIMING` line used by the TS CPU campaign.

## Change

File: `maia/src/FV/fvstructuredsolver.cpp`, function
`FvStructuredSolver<nDim>::solutionStep()`.

The ML region is already bracketed by two solver-communicator barriers
(`MAIA_ML_STEP_TIMING_SYNC`, default on). The patch records a steady-clock
timestamp after the entry barrier and after the exit barrier, and on inference
steps (`delta > 0`) reduces min/max across the solver communicator:

```
ML_STEP_TIMING step=<globalTimeStep> ml_ms=<max> rank_min_ms=<min> rank_max_ms=<max>
```

Semantics (TS parity): **entry barrier excluded, exit barrier included**, solver
communicator only (DL ranks do not participate). `ml_ms` equals the maximum
per-rank region time because the exit barrier waits for the slowest rank, which
is what the thesis reports as the max ML-step time.

## Enabling

```
export MAIA_ML_STEP_TIMING_PRINT=1     # default off, no overhead when unset
export MAIA_ML_STEP_TIMING_SYNC=1      # default on
```

Only rank 0 prints. The line lands in the solver log (`solver.log`), which the
smoke/characterization runners capture. With `srun --label`, lines are
rank-prefixed.

## Applying / reverting

The change is applied in place in the artifact checkout. To reproduce or revert:

```
cd /hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference
git diff -- maia/src/FV/fvstructuredsolver.cpp
git checkout -- maia/src/FV/fvstructuredsolver.cpp   # reverts ALL local solver changes
```

Note: the artifact already carried uncommitted solver changes (AIX `app_comm`
handle fix, `MAIA_AIX_BATCHSIZE`, the timing barriers) before this patch.

## Rebuild

Incremental rebuild of the existing build directory (Unix Makefiles):

```
sbatch mmcp_benchmark/slurm/rebuild_maia_devel.sbatch
```

or interactively inside an allocation:

```
source /hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/setup_env_claix23.sh
cmake --build /hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/maia/build_gnu_production -j 24
```

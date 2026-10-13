#!/usr/bin/env bash
# Resources and cell settings are supplied by submit_cpu_retest.sh.
set -euo pipefail
export SOLVER_RANKS=96 NP_DL=96
export SOLUTION_INTERVAL=10000 ML_STEP_TIMING_PRINT=1 SS_DB_BATCH=0
export MAIA_AIX_BATCHSIZE=50000
case "${MEASUREMENT:?}" in
    profile|timing) export RUN_STEPS=200 EXPECTED_INFERENCE_CALLS=7 ;;
    memory) export RUN_STEPS=80 EXPECTED_INFERENCE_CALLS=3 ;;
    *) exit 2 ;;
esac
export CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-14400}"
# Load once at job startup: queued jobs see edits, running jobs keep this version.
engine="${PROJECT_DIR:?}/mmcp_benchmark/slurm/cpu_engine.sh"
exec bash -c "$(<"${engine}")" "${engine}"

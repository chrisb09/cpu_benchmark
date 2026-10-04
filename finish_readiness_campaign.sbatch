#!/bin/bash
#SBATCH --job-name=readiness_analysis
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/logs/readiness_analysis_%j.log
set -eo pipefail
export USE_SCOREP=1
source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh
ROOT=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark
DATASET_ID=${DATASET_ID:-readiness_20261003}
[[ $DATASET_ID =~ ^[A-Za-z0-9_-]+$ ]] || exit 1
: "${CAMPAIGN_JOBS:?Specify exact smoke/native/performance/memory jobs}"
sacct -j "$CAMPAIGN_JOBS" --parsable2 --format=JobID,State,ExitCode,Elapsed,NodeList,MaxRSS \
    > "$ROOT/readiness_results/$DATASET_ID/slurm_status.txt"
PYTHON=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python
if [[ ${MODE:-campaign} == yield_smoke ]]; then
    "$PYTHON" "$ROOT/readiness_campaign.py" yield_smoke --dataset "$DATASET_ID"
    exit 0
fi
"$PYTHON" "$ROOT/readiness_campaign.py" analyze --dataset "$DATASET_ID"
"$PYTHON" "$ROOT/readiness_campaign.py" native --dataset "$DATASET_ID"

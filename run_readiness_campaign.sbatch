#!/bin/bash
#SBATCH --job-name=phydll_readiness
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/logs/readiness_%j.log
set -eo pipefail
for name in $(compgen -e); do
    case $name in SMARTSIM_*|SCOREP_*|MLCOUPLING_*|PHYDLL_*|OMP_*|MKL_*|OPENBLAS_*|KMP_*|GOMP_*|OMPI_*|PMIX_*) unset "$name" ;; esac
done
unset USE_SCOREP LD_PRELOAD PYTHONHOME PYTHONPATH TERRAIN_MEMORY_TOKEN SSDB
if [[ ${MODE:-smoke} != native && ${MODE:-smoke} != yield_smoke ]]; then export USE_SCOREP=1; fi
source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh
export LD_LIBRARY_PATH="$EBROOTCUDA/stubs/lib64:$EBROOTCUDA/lib64/stubs:${LD_LIBRARY_PATH:-}"
export SCOREP_ENABLE_PROFILING=true SCOREP_ENABLE_TRACING=false SCOREP_METRIC_PAPI=
export SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY=false
export MLCOUPLING_MEM_LOG_VERBOSE=0 MLCOUPLING_MEM_LOG_COLLECTIVE=0 MLCOUPLING_STEP_TIMING_SYNC=1
export PHYDLL_IO_LOG=0 PHYDLL_IO_LOG_BARRIERS=0 PHYDLL_TELEMETRY=0 SMARTSIM_FLAG_TELEMETRY=0
export MLCOUPLING_INTRA_OP_THREADS=1 MLCOUPLING_INTER_OP_THREADS=1 PHYDLL_DL_COUNT=1 PHYDLL_DL_FIELD_COUNT=1
export HDF5_USE_FILE_LOCKING=FALSE
/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python \
    /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/readiness_campaign.py run \
    --mode "${MODE:-smoke}" --dataset "${DATASET_ID:-readiness_20261003}" --task "${SLURM_ARRAY_TASK_ID:-0}"

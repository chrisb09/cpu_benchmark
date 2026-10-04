#!/bin/bash
#SBATCH --job-name=readiness_audit
#SBATCH --account=default
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/logs/readiness_audit_%j.log
set -eo pipefail
export USE_SCOREP=1
source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh
/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python audit_readiness_campaign.py

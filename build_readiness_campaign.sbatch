#!/bin/bash
#SBATCH --job-name=readiness_build
#SBATCH --account=default
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --chdir=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/logs/readiness_build_%j.log
set -eo pipefail
bash build_readiness_campaign.sh "${BUILD_KIND:-scorep}" "${BUILD_DESTINATION:-${BUILD_KIND:-scorep}}"

#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:40:00
#SBATCH --job-name=mmcp-maia-rebuild
#SBATCH --output=logs/mmcp-maia-rebuild_%j.out
#SBATCH --error=logs/mmcp-maia-rebuild_%j.err

# Incremental rebuild of the artifact m-aia build directory after source edits.
# devel uses the user's default account (no --account directive).
set -euo pipefail

project_dir="${PROJECT_DIR:-/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference}"
maia_build="${MAIA_BUILD_DIR:-${project_dir}/maia/build_gnu_production}"

source "${project_dir}/setup_env_claix23.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LIBCLANG_PATH="${HOME}/.local/lib/python3.11/site-packages/clang/native"
cuda_lib=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="${LIBCLANG_PATH}:${cuda_lib}/stubs:${cuda_lib}:${LD_LIBRARY_PATH:-}"

printf 'Rebuilding m-aia in %s\n' "${maia_build}"
cmake --build "${maia_build}" -j "${SLURM_CPUS_PER_TASK:-24}"
printf 'MAIA_REBUILD_PASS\n'

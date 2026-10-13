#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=5G
#SBATCH --time=01:00:00
#SBATCH --job-name=mmcp-scorep-rebuild
#SBATCH --output=logs/mmcp-scorep-rebuild_%j.out
#SBATCH --error=logs/mmcp-scorep-rebuild_%j.err

# Incrementally rebuild the existing Score-P MAIA build with the current source
# (our ML_STEP_TIMING instrumentation) and the fresh Score-P AIX prebuilt.
# devel uses the user's default account (no --account directive).
set -euo pipefail

project_dir="${PROJECT_DIR:-/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference}"
build="${MAIA_SCOREP_BUILD_DIR:-${project_dir}/maia/build_gnu_production_cmi_scorep}"
cmi="${project_dir}/CPP-ML-Interface"
aix_prebuilt="${AIXELERATOR_PREBUILT_INSTALL_PREFIX:-${cmi}/extern/AIxeleratorService/INSTALL-SCOREP}"

[[ -f "${build}/CMakeCache.txt" ]] || { printf 'No Score-P build dir: %s\n' "${build}" >&2; exit 1; }

source "${project_dir}/setup_env_claix23.sh"
cuda_lib=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib
export LIBCLANG_PATH="${HOME}/.local/lib/python3.11/site-packages/clang/native"
export LD_LIBRARY_PATH="${LIBCLANG_PATH}:${cuda_lib}/stubs:${cuda_lib}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export SCOREP_ENABLE_CUDA=0
export SCOREP_MPP_SYSTEM=mpi
export SCOREP_THREADING_SYSTEM=none
export SCOREP_ENABLE_COMPILER=OFF
export SCOREP_ENABLE_USER=ON

cd "${build}"
echo "=== Reconfigure (AIX prebuilt: ${aix_prebuilt}) ==="
cmake . \
    -DAIX_USE_PREBUILT=ON \
    -DAIX_SKIP_VENV_CREATION=ON \
    -DAIXELERATOR_PREBUILT_INSTALL_PREFIX="${aix_prebuilt}" \
    -DLIBTORCH_DIR="${cmi}/extern/libtorch" \
    -DTORCH_VERSION=2.4.0 \
    -DWITH_SCOREP=ON \
    -DCMAKE_EXE_LINKER_FLAGS:STRING="-L${cuda_lib}/stubs -Wl,-rpath-link,${cuda_lib}/stubs" \
    -DCMAKE_SHARED_LINKER_FLAGS:STRING="-L${cuda_lib}/stubs -Wl,-rpath-link,${cuda_lib}/stubs"

echo "=== Build ==="
cmake --build . -j "${SLURM_CPUS_PER_TASK:-48}"
ls -lh bin/maia
printf 'SCOREP_REBUILD_PASS\n'

#!/bin/bash
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "${BENCH_DIR}/../.." && pwd)"
CPP_DIR="${BASE_DIR}/CPP-ML-Interface"
BUILD_DIR="${PROVIDER_BENCH_BUILD_DIR:-${BENCH_DIR}/build-provider-bench}"
DL_BUILD_DIR="${PHYDLL_DL_BUILD_DIR:-${BUILD_DIR}/dl-client}"
SMARTSIM_PYTHON="${SMARTSIM_PYTHON:-${CPP_DIR}/extern/python/smartsim_cpu/bin/python}"
TORCH_DIR="${TORCH_DIR:-${CPP_DIR}/extern/libtorch/share/cmake/Torch}"
PHYDLL_BUILD_DIR="${PHYDLL_BUILD_DIR:-${CPP_DIR}/extern/phydll/build}"
CUDA_STUB="${CUDA_STUB:-/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64/libcuda.so}"

if [[ "${SCOREP_MODE:-off}" == "on" ]]; then
    echo "provider_bench/build.sh currently builds the non-Score-P configuration only." >&2
    exit 2
fi

if [[ ! -x "${SMARTSIM_PYTHON}" ]]; then
    echo "SmartSim Python runtime not found: ${SMARTSIM_PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${TORCH_DIR}/TorchConfig.cmake" ]]; then
    echo "Torch CMake package not found: ${TORCH_DIR}" >&2
    exit 1
fi
if [[ ! -d "${PHYDLL_BUILD_DIR}/lib" ]]; then
    echo "PhyDLL build not found: ${PHYDLL_BUILD_DIR}" >&2
    exit 1
fi

if [[ -f "${CPP_DIR}/install.sh" ]]; then
    pushd "${CPP_DIR}" >/dev/null
    source ./set_env_claix23_cuda12.4.sh
    popd >/dev/null
fi

cmake -S "${BENCH_DIR}" -B "${BUILD_DIR}" \
    -DSMARTSIM_PYTHON="${SMARTSIM_PYTHON}" \
    -DTorch_DIR="${TORCH_DIR}" \
    -DPROVIDER_BENCH_WITH_AIX="${PROVIDER_BENCH_WITH_AIX:-ON}" \
    -DPROVIDER_BENCH_WITH_SCOREP=OFF \
    -DAIXELERATOR_PREBUILT_INSTALL_PREFIX="${AIXELERATOR_INSTALL_PREFIX:-${CPP_DIR}/extern/AIxeleratorService/INSTALL}" \
    -DAIXELERATOR_PREBUILT_LIB_DIR="${AIXELERATOR_LIB_DIR:-${CPP_DIR}/extern/AIxeleratorService/INSTALL/lib}" \
    -DAIXELERATOR_CMAKE_ARGS="-DWITH_TORCH=ON -DWITH_SCOREP=OFF -DBUILD_TESTS=OFF"
cmake --build "${BUILD_DIR}" -j"${BUILD_JOBS:-2}"

cmake -S "${CPP_DIR}/dl_clients" -B "${DL_BUILD_DIR}" \
    -DLIBTORCH_DIR="${CPP_DIR}/extern/libtorch" \
    -DPHYDLL_BUILD_DIR="${PHYDLL_BUILD_DIR}" \
    -DCUDA_CUDA_LIB="${CUDA_STUB}" \
    -DCUDA_cuda_driver_LIBRARY="${CUDA_STUB}" \
    -DWITH_SCOREP=OFF
cmake --build "${DL_BUILD_DIR}" -j"${BUILD_JOBS:-2}"

echo "Built non-Score-P provider_bench artifacts:"
echo "  solver: ${BUILD_DIR}/benchmark_solver"
echo "  DL client: ${DL_BUILD_DIR}/phydll_dl_client"

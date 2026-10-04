#!/bin/bash
set -eo pipefail
ROOT=$(pwd)
BASE=${ROOT%/cpu_benchmark}
kind=${1:-scorep}
destination=${2:-$kind}
[[ $destination =~ ^[A-Za-z0-9_-]+$ ]] || exit 1
[[ $kind == scorep || $kind == native ]] || exit 1
unset USE_SCOREP LD_PRELOAD
if [[ $kind == scorep ]]; then export USE_SCOREP=1; fi
source "$BASE/set_env_claix23_cuda12.4.sh"
ROOT=$(pwd)
BASE=${ROOT%/cpu_benchmark}
CPP=$BASE/CPP-ML-Interface
export LD_LIBRARY_PATH="$EBROOTCUDA/stubs/lib64:$EBROOTCUDA/lib64/stubs:${LD_LIBRARY_PATH:-}"
compiler=mpicxx; ccompiler=mpicc; flags=; profile=OFF
if [[ $kind == scorep ]]; then compiler=scorep-mpicxx; ccompiler=scorep-mpicc; profile=ON
else flags="-DFORWARD_CPU_DIAGNOSTIC -I$ROOT"; fi
cmake -S "$BASE/mini_app/solver_cpp" -B "$ROOT/readiness_build/$destination" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$compiler" -DCMAKE_C_COMPILER="$ccompiler" \
    -DCMAKE_CXX_FLAGS="$flags" -DUSE_CPP_ML_INTERFACE=ON -DWITH_PHYDLL=ON \
    -DWITH_AIX=ON -DWITH_SMARTSIM=ON -DWITH_SCOREP="$profile" -DWITH_FORTRAN=OFF \
    -DAIX_USE_PREBUILT=OFF -DWITH_TORCH=ON -DBUILD_TESTS=OFF -DAIX_SKIP_VENV_CREATION=ON \
    -DUSE_PYTHON_TORCH_CMAKE_PREFIX=OFF -DLIBTORCH_DIR="$CPP/extern/libtorch" \
    -DSMARTSIM_PYTHON="$CPP/extern/python/smartsim_cuda-12/bin/python" \
    -DAIX_VENV_DIR="$CPP/extern/python/smartsim_cuda-12" \
    -DFETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS="$BASE/mini_app/solver_cpp/build_scorep_cmi_fresh/_deps/tomlplusplus-src"
cmake --build "$ROOT/readiness_build/$destination" --target terrain_solver phydll_dl_client -j 4

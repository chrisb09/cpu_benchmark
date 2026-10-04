#!/bin/bash
set -eo pipefail
ROOT=$(pwd)
BASE=${ROOT%/cpu_benchmark}
unset USE_SCOREP LD_PRELOAD
source "$BASE/set_env_claix23_cuda12.4.sh"
CPP=$BASE/CPP-ML-Interface
export LD_LIBRARY_PATH="$EBROOTCUDA/stubs/lib64:$EBROOTCUDA/lib64/stubs:${LD_LIBRARY_PATH:-}"
cmake -S "$BASE/mini_app/solver_cpp" -B "$ROOT/forward_diag_build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_FLAGS="-DFORWARD_CPU_DIAGNOSTIC -I$ROOT" \
    -DUSE_CPP_ML_INTERFACE=ON -DWITH_PHYDLL=ON -DWITH_AIX=ON -DWITH_SMARTSIM=ON \
    -DWITH_SCOREP=OFF -DWITH_FORTRAN=OFF -DAIX_USE_PREBUILT=OFF -DWITH_TORCH=ON \
    -DBUILD_TESTS=OFF -DAIX_SKIP_VENV_CREATION=ON -DUSE_PYTHON_TORCH_CMAKE_PREFIX=OFF \
    -DLIBTORCH_DIR="$CPP/extern/libtorch" \
    -DSMARTSIM_PYTHON="$CPP/extern/python/smartsim_cuda-12/bin/python" \
    -DAIX_VENV_DIR="$CPP/extern/python/smartsim_cuda-12" \
    -DFETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS="$BASE/mini_app/solver_cpp/build_scorep_cmi_fresh/_deps/tomlplusplus-src"
cmake --build "$ROOT/forward_diag_build" --target terrain_solver phydll_dl_client -j 4
mpicc -O2 -fPIC -shared "$ROOT/forward_diag_paced_wait.c" -o "$ROOT/forward_diag_build/libpaced_wait.so"

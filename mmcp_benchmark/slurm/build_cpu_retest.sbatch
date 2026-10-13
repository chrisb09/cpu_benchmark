#!/usr/bin/env bash
#SBATCH --job-name=build-maia-cpu-retest
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/mmcp_benchmark/logs/build_maia_cpu_retest_%j.out

set -Eeuo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ $# == 0 || ( $# == 1 && $1 == --preflight ) ]] || fail 'Usage: build_cpu_retest.sbatch [--preflight]'

artifact=/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference
cmi=$artifact/CPP-ML-Interface
maia=$artifact/maia
phy=$maia/build_cpu_retest_phydll
plain=$maia/build_cpu_retest_plain
profile=$maia/build_cpu_retest_scorep
torch=$cmi/extern/libtorch
cuda_lib=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib
stubs=$cuda_lib/stubs
scorep_root=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/Score-P/8.4-gompi-2023b-CUDA-12.4.0

[[ -d $maia && -d $cmi/extern/phydll && -f $artifact/setup_env_claix23.sh ]] || fail 'Artifact source or environment is missing'
[[ $(git -C "$cmi" rev-parse HEAD) == 9fc6085a1113dda76bdc0c107bee7c73f59e7e1e ]] || fail 'Unexpected CMI revision'
[[ $(git -C "$cmi/extern/phydll" rev-parse HEAD) == 05e0e292e829d806b4b6477eda65c87474f31831 ]] || fail 'Unexpected PhyDLL revision'
for dir in "$phy" "$plain" "$profile"; do
    [[ ! -e $dir && ! -L $dir ]] || fail "Dedicated build path already exists (never overwrite/resume): $dir"
done
for file in "$torch/lib/libtorch.so" "$stubs/libcuda.so" \
    "$maia/build_gnu_production/_deps/highfive_fetched-src/CMakeLists.txt" \
    "$maia/build_gnu_production/_deps/tomlplusplus-src/CMakeLists.txt"; do
    [[ -r $file ]] || fail "Missing read-only dependency: $file"
done
# CMI auto-bootstraps AIx into the shared install when its stamp mismatches.
# Reject that case before configuring anything; queued jobs still use these installs.
aix_revision=$(git -C "$cmi/extern/AIxeleratorService" rev-parse HEAD)
for install in INSTALL INSTALL-SCOREP; do
    prefix=$cmi/extern/AIxeleratorService/$install
    [[ -r $prefix/.git_commit && -r $prefix/lib/libAIxeleratorService.so ]] || fail "Missing AIx prebuilt: $prefix"
    [[ $(<"$prefix/.git_commit") == "$aix_revision" ]] || fail "AIx stamp mismatch (would mutate shared install): $prefix"
done
if [[ ${1:-} == --preflight ]]; then
    printf 'Preflight passed; private PhyDLL=%s plain=%s Score-P=%s (no builds created)\n' "$phy" "$plain" "$profile"
    exit 0
fi
[[ ${SLURM_JOB_PARTITION:-} == c23ms && ${SLURM_CPUS_PER_TASK:-0} -ge 8 ]] || fail 'Requires a c23ms CPU build allocation'

source "$artifact/setup_env_claix23.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LIBCLANG_PATH="$HOME/.local/lib/python3.11/site-packages/clang/native"
export LD_LIBRARY_PATH="$LIBCLANG_PATH:$stubs:$cuda_lib:${LD_LIBRARY_PATH:-}"
export SCOREP_ENABLE_CUDA=0 SCOREP_MPP_SYSTEM=mpi SCOREP_THREADING_SYSTEM=none SCOREP_ROOT_DIR=$scorep_root
export SCOREP_ENABLE_COMPILER=OFF SCOREP_ENABLE_USER=ON
for tool in cmake mpicc mpicxx scorep; do command -v "$tool" >/dev/null || fail "Missing tool: $tool"; done

# Exclusive mkdir: no configure.py/reset, cleanup, or build in any old directory.
mkdir "$phy" "$plain" "$profile"
mkdir -p "$phy/lib" "$phy/include" "$plain/lib" "$profile/lib"
make -C "$cmi/extern/phydll" compile "BUILD=$phy" "CC=$(command -v mpicc)" \
    'CFLAGS=-O2 -g -Wall -Wextra -std=c99'
[[ -s $phy/lib/libphydll.so && -s $phy/include/phydll.h ]] || fail 'Private PhyDLL build incomplete'

for mode in plain scorep; do
    if [[ $mode == plain ]]; then
        build=$plain
        install=$cmi/extern/AIxeleratorService/INSTALL
        compiler=$(command -v mpicxx)
        instrument=OFF
    else
        build=$profile
        install=$cmi/extern/AIxeleratorService/INSTALL-SCOREP
        compiler=$(command -v scorep)
        instrument=ON
    fi
    # cpu_engine.sh prepends build/lib ahead of the legacy shared PhyDLL path.
    cp "$phy/lib/libphydll.so" "$build/lib/libphydll.so"
    export LD_LIBRARY_PATH="$build/lib:$phy/lib:$LD_LIBRARY_PATH"
    args=(
        -DCMAKE_BUILD_TYPE=production -DMAIA_COMPILER=GNU
        -DWITH_HDF5=ON -DWITH_AIX=ON -DWITH_SMARTSIM=ON -DWITH_PHYDLL=ON
        -DWITH_FORTRAN=ON -DBUILD_TESTING=OFF -DPHYDLL_DL_WITH_TORCH=ON
        "-DCMAKE_CXX_COMPILER=$compiler" "-DWITH_SCOREP=$instrument"
        -DAIX_USE_PREBUILT=ON -DAIX_SKIP_VENV_CREATION=ON
        "-DAIXELERATOR_PREBUILT_INSTALL_PREFIX=$install"
        "-DAIXELERATOR_PREBUILT_LIB_DIR=$install/lib"
        "-DAIXELERATOR_PREBUILT_LIB:FILEPATH=$install/lib/libAIxeleratorService.so"
        "-DPHYDLL_BUILD_DIR=$phy" "-DLIBTORCH_DIR=$torch" "-DTorch_DIR=$torch/share/cmake/Torch" -DTORCH_VERSION=2.4.0
        "-DFETCHCONTENT_SOURCE_DIR_HIGHFIVE_FETCHED=$maia/build_gnu_production/_deps/highfive_fetched-src"
        "-DFETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS=$maia/build_gnu_production/_deps/tomlplusplus-src"
        -DFETCHCONTENT_FULLY_DISCONNECTED=ON
    )
    if [[ $mode == scorep ]]; then
        args+=(
            '-DCMAKE_CXX_COMPILER_ARG1= --nocompiler --nocuda --thread=none --mpp=mpi --user mpicxx'
            "-DSCOREP_ROOT_DIR=$scorep_root"
            "-DSCOREP_CONFIG_EXECUTABLE=$scorep_root/bin/scorep-config"
            "-DSCOREP_INFO_EXECUTABLE=$scorep_root/bin/scorep-info"
            "-DCMAKE_EXE_LINKER_FLAGS=-L$stubs -Wl,-rpath-link,$stubs"
            "-DCMAKE_SHARED_LINKER_FLAGS=-L$stubs -Wl,-rpath-link,$stubs"
        )
    fi
    cmake -S "$maia" -B "$build" "${args[@]}"
    cmake --build "$build" --target maia --parallel "${SLURM_CPUS_PER_TASK}"
    cmake --build "$build/CPP-ML-Interface/dl_clients" --target phydll_dl_client --parallel "${SLURM_CPUS_PER_TASK}"
    solver=$build/bin/maia
    client=$build/bin/phydll_dl_client
    [[ -x $solver && -x $client && -s $build/lib/libcpp_ml_interface_library.so ]] || fail "Missing build output: $build"
    cmp "$phy/lib/libphydll.so" "$build/lib/libphydll.so"
    for binary in "$solver" "$client"; do
        dependencies=$(ldd -r "$binary") || fail "Unresolved dependencies: $binary"
        [[ $dependencies != *'not found'* && $dependencies != *'undefined symbol'* ]] || fail "Unresolved dependencies: $binary"
        [[ $dependencies == *"$build/lib/libphydll.so"* ]] || fail "Binary resolves old/shared PhyDLL: $binary"
        if [[ $mode == scorep ]]; then
            nm -D "$binary" | rg -q SCOREP_InitMeasurement || fail "Missing Score-P: $binary"
        else
            ! nm -D "$binary" | rg -q SCOREP_InitMeasurement || fail "Plain build contains Score-P: $binary"
        fi
    done
    [[ $(<"$install/.git_commit") == "$aix_revision" ]] || fail 'AIx prebuilt changed during build'
    printf 'MAIA_CPU_RETEST_BUILD_READY mode=%s cmi=%s phydll=%s job=%s\n' \
        "$mode" "$(git -C "$cmi" rev-parse HEAD)" \
        "$(git -C "$cmi/extern/phydll" rev-parse HEAD)" "${SLURM_JOB_ID:-unknown}" > "$build/BUILD_READY"
    printf 'Ready (build/link verified, runtime not tested): %s\n' "$build"
done

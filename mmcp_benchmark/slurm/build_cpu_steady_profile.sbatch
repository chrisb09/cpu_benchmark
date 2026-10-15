#!/usr/bin/env bash
#SBATCH --job-name=build-cpu-steady-profile
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/mmcp_benchmark/logs/build_cpu_steady_profile_%j.out

set -Eeuo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ $# == 0 || ( $# == 1 && $1 == --preflight ) ]] || fail 'Usage: build_cpu_steady_profile.sbatch [--preflight]'
artifact=/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference
cmi=$artifact/CPP-ML-Interface
maia=$artifact/maia
build=${STEADY_BUILD_DIR:-$maia/build_cpu_steady_20261014_scorep}
phy=${STEADY_PHYDLL_DIR:-$maia/build_cpu_steady_20261014_phydll}
aix=$cmi/extern/AIxeleratorService
aix_revision=86968b64380307a5113db8077ffe4b9494309c71
shared=$aix/INSTALL-SCOREP
private=$build/aix-install
worktree=$build/aix-source
torch=$cmi/extern/libtorch
cuda_lib=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib
stubs=$cuda_lib/stubs
scorep_root=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/Score-P/8.4-gompi-2023b-CUDA-12.4.0
for dir in "$build" "$phy"; do
    [[ $dir == "$maia"/build_cpu_steady_* && ! -e $dir && ! -L $dir ]] || fail "Not a fresh absolute build path: $dir"
done
[[ $build != "$phy" ]] || fail 'Build paths must differ'
[[ $(git -C "$cmi" rev-parse HEAD) == 0ea8cdd70c675cbaad0005efc0cb4872aea8f6d5 ]] || fail 'Unexpected CMI revision'
[[ $(git -C "$cmi/extern/phydll" rev-parse HEAD) == 05e0e292e829d806b4b6477eda65c87474f31831 ]] || fail 'Unexpected PhyDLL revision'
git -C "$aix" cat-file -e "$aix_revision^{commit}"
[[ $(<"$shared/.git_commit") == "$aix_revision" ]] || fail 'Shared AIX stamp does not match CPU baseline'
for file in "$shared/lib/libAIxeleratorService.so" "$artifact/setup_env_claix23.sh" \
    "$torch/lib/libtorch.so" "$stubs/libcuda.so" \
    "$maia/build_gnu_production/_deps/highfive_fetched-src/CMakeLists.txt" \
    "$maia/build_gnu_production/_deps/tomlplusplus-src/CMakeLists.txt"; do
    [[ -r $file ]] || fail "Missing read-only dependency: $file"
done
grep -q cmi_inference_steady "$cmi/include/scorep_profiling_state.hpp" || fail 'CMI steady scope missing'
grep -q CMI_SCOREP_WARMUP_CALLS "$cmi/dl_clients/dl_client.cpp" || fail 'DL warmup contract missing'
grep -q 'set_scorep_warmup_inference_count(2)' "$maia/src/FV/fvstructuredsolver.cpp" || fail 'MAIA warmup contract missing'
if [[ ${1:-} == --preflight ]]; then
    printf 'Preflight passed (no builds created):\nSCOREP=%s\nPHYDLL=%s\nAIX=%s\n' "$build" "$phy" "$aix_revision"
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
mkdir "$build" "$phy"
mkdir "$build/lib" "$build/provenance"
# Compile live sources only if their full file fingerprints remain unchanged.
fingerprints() {
    for repo in "$maia" "$cmi" "$cmi/extern/phydll"; do
        printf 'REPOSITORY=%s\n' "$repo"
        (cd "$repo"; git ls-files -z --cached --others --exclude-standard -- . ':!:extern' ':!:build*' ':!:BUILD*' | LC_ALL=C sort -zu | xargs -0 -r sha256sum)
    done
}
fingerprints > "$build/provenance/sources.before.sha256"
for name in maia cmi phydll aix; do
    case $name in maia) repo=$maia;; cmi) repo=$cmi;; phydll) repo=$cmi/extern/phydll;; aix) repo=$aix;; esac
    git -C "$repo" rev-parse HEAD > "$build/provenance/$name.revision"
    git -C "$repo" status --porcelain=v1 --untracked-files=all > "$build/provenance/$name.status"
    git -C "$repo" diff HEAD --binary --full-index > "$build/provenance/$name.patch"
    if [[ $name != aix ]]; then
        (cd "$repo"; git ls-files -z --cached --others --exclude-standard -- . ':!:extern' ':!:build*' ':!:BUILD*' |
            LC_ALL=C sort -zu | tar --null -T - -cf "$build/provenance/$name.sources.tar")
    fi
done
sha256sum "$shared/.git_commit" "$shared/lib/libAIxeleratorService.so" > "$build/provenance/shared-aix.before.sha256"
git -C "$aix" worktree add --detach "$worktree" "$aix_revision"
cp -aL "$shared" "$private"
[[ $(git -C "$worktree" rev-parse HEAD) == "$(<"$private/.git_commit")" ]] || fail 'Private AIX stamp mismatch'
cmp "$shared/lib/libAIxeleratorService.so" "$private/lib/libAIxeleratorService.so"
sha256sum "$private/lib/libAIxeleratorService.so" > "$build/provenance/private-aix.sha256"
make -C "$cmi/extern/phydll" compile "BUILD=$phy" "CC=$(command -v mpicc)" 'CFLAGS=-O2 -g -Wall -Wextra -std=c99'
[[ -s $phy/lib/libphydll.so && -s $phy/include/phydll.h ]] || fail 'Private PhyDLL incomplete'
cp "$phy/lib/libphydll.so" "$build/lib/libphydll.so"
export LD_LIBRARY_PATH="$private/lib:$build/lib:$phy/lib:$LD_LIBRARY_PATH"
cmake -S "$maia" -B "$build" \
    -DCMAKE_BUILD_TYPE=production -DMAIA_COMPILER=GNU \
    -DWITH_HDF5=ON -DWITH_AIX=ON -DWITH_SMARTSIM=ON -DWITH_PHYDLL=ON \
    -DWITH_FORTRAN=ON -DBUILD_TESTING=OFF -DPHYDLL_DL_WITH_TORCH=ON -DWITH_SCOREP=ON \
    "-DCMAKE_CXX_COMPILER=$(command -v scorep)" \
    '-DCMAKE_CXX_COMPILER_ARG1= --nocompiler --nocuda --thread=none --mpp=mpi --user mpicxx' \
    -DAIX_USE_PREBUILT=ON -DFORCE_AIX_REBUILD=OFF -DAIX_SKIP_VENV_CREATION=ON \
    "-DAIXELERATOR_DIR=$worktree" "-DAIXELERATOR_PREBUILT_INSTALL_PREFIX=$private" \
    "-DAIXELERATOR_PREBUILT_LIB_DIR=$private/lib" \
    "-DAIXELERATOR_PREBUILT_LIB:FILEPATH=$private/lib/libAIxeleratorService.so" \
    "-DPHYDLL_BUILD_DIR=$phy" "-DLIBTORCH_DIR=$torch" "-DTorch_DIR=$torch/share/cmake/Torch" -DTORCH_VERSION=2.4.0 \
    "-DFETCHCONTENT_SOURCE_DIR_HIGHFIVE_FETCHED=$maia/build_gnu_production/_deps/highfive_fetched-src" \
    "-DFETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS=$maia/build_gnu_production/_deps/tomlplusplus-src" \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON "-DSCOREP_ROOT_DIR=$scorep_root" \
    "-DSCOREP_CONFIG_EXECUTABLE=$scorep_root/bin/scorep-config" "-DSCOREP_INFO_EXECUTABLE=$scorep_root/bin/scorep-info" \
    "-DCMAKE_EXE_LINKER_FLAGS=-L$stubs -Wl,-rpath-link,$stubs" \
    "-DCMAKE_SHARED_LINKER_FLAGS=-L$stubs -Wl,-rpath-link,$stubs"
[[ ! -d $build/aixelerator-prebuilt-build ]] || fail 'Unexpected AIX bootstrap'
cmake --build "$build" --target maia --parallel "$SLURM_CPUS_PER_TASK"
cmake --build "$build/CPP-ML-Interface/dl_clients" --target phydll_dl_client --parallel "$SLURM_CPUS_PER_TASK"
for binary in "$build/bin/maia" "$build/bin/phydll_dl_client"; do
    [[ -x $binary ]] || fail "Missing binary: $binary"
    dependencies=$(ldd -r "$binary") || fail "ldd failed: $binary"
    printf '%s\n' "$dependencies" > "$build/provenance/$(basename "$binary").ldd"
    [[ $dependencies != *'not found'* && $dependencies != *'undefined symbol'* ]] || fail "Unresolved dependencies: $binary"
    [[ $dependencies == *"$build/lib/libphydll.so"* ]] || fail "Shared PhyDLL resolved: $binary"
    symbols=$(nm -D "$binary")
    [[ $symbols == *SCOREP_InitMeasurement* ]] || fail "Missing Score-P: $binary"
done
grep -Fq "$private/lib/libAIxeleratorService.so" "$build/provenance/maia.ldd" || fail 'Shared AIX resolved'
fingerprints > "$build/provenance/sources.after.sha256"
cmp "$build/provenance/sources.before.sha256" "$build/provenance/sources.after.sha256" || fail 'Compile sources changed during build'
sha256sum -c "$build/provenance/shared-aix.before.sha256"
sha256sum -c "$build/provenance/private-aix.sha256"
cmp "$phy/lib/libphydll.so" "$build/lib/libphydll.so"
sha256sum "$build"/provenance/*.patch "$build"/provenance/sources.before.sha256 \
    "$build/bin/maia" "$build/bin/phydll_dl_client" "$build/lib/"*.so > "$build/provenance/code.sha256"
{
    printf 'MAIA_CPU_STEADY_BUILD_READY\nPROFILE_SCOPE_VERSION=steady-v1\nPROFILE_SCOPE=cmi_inference_steady\nPROFILE_WARMUP_CALLS=2\n'
    printf 'AIX_REVISION=%s\nCMI_REVISION=%s\nBUILD_JOB=%s\n' "$aix_revision" "$(git -C "$cmi" rev-parse HEAD)" "$SLURM_JOB_ID"
    printf 'CODE_FINGERPRINT=%s\n' "$(sha256sum "$build/provenance/code.sha256" | cut -d ' ' -f1)"
    printf 'SOURCE_FINGERPRINT=%s\n' "$(sha256sum "$build/provenance/sources.before.sha256" | cut -d ' ' -f1)"
} > "$build/BUILD_READY"
printf 'Ready (build/link verified, runtime not tested): %s\n' "$build"

#!/bin/bash
#SBATCH --job-name=giant_forward_diag
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/logs/forward_diag_%j.log
set -eo pipefail
ROOT=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark
BASE=${ROOT%/cpu_benchmark}
while read -r name; do
    case $name in SMARTSIM_*|SCOREP_*|MLCOUPLING_*|PHYDLL_*|OMP_*|MKL_*|OPENBLAS_*|KMP_*|GOMP_*|OMPI_*|PMIX_*) unset "$name" ;; esac
done < <(compgen -e)
unset USE_SCOREP LD_PRELOAD PYTHONHOME PYTHONPATH TERRAIN_MEMORY_TOKEN SSDB
source "$BASE/set_env_claix23_cuda12.4.sh"
CPP=$BASE/CPP-ML-Interface
export LD_LIBRARY_PATH="$EBROOTCUDA/stubs/lib64:$EBROOTCUDA/lib64/stubs:${LD_LIBRARY_PATH:-}"
export MLCOUPLING_MEM_LOG_VERBOSE=0 MLCOUPLING_MEM_LOG_COLLECTIVE=0 MLCOUPLING_STEP_TIMING_SYNC=0
export PHYDLL_IO_LOG=0 PHYDLL_IO_LOG_BARRIERS=0 PHYDLL_TELEMETRY=0 SMARTSIM_FLAG_TELEMETRY=0
export HDF5_USE_FILE_LOCKING=FALSE
SOLVER=$ROOT/forward_diag_build/terrain_solver
DL=$ROOT/forward_diag_build/cpp-ml-interface-build/dl_clients/phydll_dl_client
OUT=$ROOT/forward_diag_results/job_$SLURM_JOB_ID
mkdir -p "$OUT"
env | sort > "$OUT/allocation.env"
lscpu > "$OUT/lscpu.txt"
scontrol show job "$SLURM_JOB_ID" > "$OUT/slurm_job.txt"
ldd "$SOLVER" > "$OUT/solver_libraries.txt"
ldd "$DL" > "$OUT/dl_libraries.txt"
if rg -q 'libscorep|not found' "$OUT/solver_libraries.txt" "$OUT/dl_libraries.txt"; then exit 1; fi
sha256sum "$SOLVER" "$DL" "$ROOT/forward_diag_build/aixelerator-build/src/libAIxeleratorService.so" \
    "$CPP/extern/libtorch/lib/libtorch_cpu.so" "$CPP/extern/phydll/build/lib/libphydll.so" \
    "$BASE/mini_app/train_models/model_a/giant_cpu.pt" "$BASE/mini_app/input/prep_giant_240x192.h5" \
    "$ROOT/forward_cpu_diagnostic.hpp" "$ROOT/run_forward_cpu_diagnostic.sbatch" > "$OUT/inputs.sha256"
common=(--device CPU --gpus-per-node 0 --ml-nodes 1 --model-backend TORCH
    --input-hdf5 "$BASE/mini_app/input/prep_giant_240x192.h5" --steps 18 --save-every 18 --save-mode periodic
    --triangular-scale 1 --chunk-size 12 --io-mode parallel_hdf5 --mpi-sync-mode none
    --hdf5-xfer-mode independent --rank-grid-x 12 --rank-grid-z 8 --ml-interface cpp --ml-batch-size 10000000
    --model-path "$BASE/mini_app/train_models/model_a/giant_cpu.pt" --model-io-layout flat_contiguous)
exports=(-x LD_LIBRARY_PATH -x FORWARD_DIAG_DIR -x MLCOUPLING_MEM_LOG_VERBOSE -x MLCOUPLING_MEM_LOG_COLLECTIVE
    -x MLCOUPLING_STEP_TIMING_SYNC -x PHYDLL_IO_LOG -x PHYDLL_IO_LOG_BARRIERS -x PHYDLL_TELEMETRY -x SMARTSIM_FLAG_TELEMETRY
    -x FORWARD_DIAG_PACED_LIBRARY)
if [[ ${WAIT_DIAGNOSTIC:-0} == 1 ]]; then
    runs=(1_phydll 2_paced 3_paced 4_phydll)
    sha256sum "$ROOT/forward_diag_build/libpaced_wait.so" "$ROOT/forward_diag_paced_wait.c" >> "$OUT/inputs.sha256"
else
    runs=(1_aix 2_phydll 3_phydll 4_aix)
fi
for run in "${runs[@]}"; do
    dir=$OUT/$run
    mkdir -p "$dir/timings"
    export FORWARD_DIAG_DIR=$dir/timings
    unset MLCOUPLING_INTRA_OP_THREADS MLCOUPLING_INTER_OP_THREADS PHYDLL_DL_COUNT PHYDLL_DL_FIELD_COUNT
    extra=()
    export FORWARD_DIAG_PACED_LIBRARY=
    if [[ $run == *_paced ]]; then
        export FORWARD_DIAG_PACED_LIBRARY=$ROOT/forward_diag_build/libpaced_wait.so
    fi
    cfg=$ROOT/forward_diag_aix.toml
    if [[ $run != *_aix ]]; then
        cfg=$ROOT/forward_diag_phydll.toml
        export MLCOUPLING_INTRA_OP_THREADS=1 MLCOUPLING_INTER_OP_THREADS=1 PHYDLL_DL_COUNT=1 PHYDLL_DL_FIELD_COUNT=1
        extra=(-x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT)
    fi
    command=(timeout --kill-after=30 1200 mpirun --bind-to none --oversubscribe "${exports[@]}" "${extra[@]}"
        -n 96 bash "$ROOT/forward_diag_pin.sh" "$SOLVER" "${common[@]}" --cpp-ml-config "$cfg" --output-hdf5 "$dir/traj.h5")
    if [[ $run != *_aix ]]; then
        command+=(: "${exports[@]}" "${extra[@]}" -n 96 bash "$ROOT/forward_diag_pin.sh" "$DL")
    fi
    env | sort > "$dir/run.env"
    printf '%q ' "${command[@]}" > "$dir/command.sh"
    printf 'START %s %s\n' "$run" "$(date -Is)"
    (cd "$dir"; exec "${command[@]}" < /dev/null) > "$dir/solver.log" 2>&1
    printf 'COMPLETE %s %s\n' "$run" "$(date -Is)"
done
for run in "${runs[@]:1}"; do
    h5diff --exclude-path /runtime_seconds -v "$OUT/${runs[0]}/traj.h5" "$OUT/$run/traj.h5" >> "$OUT/hdf5_validation.txt"
done
"$CPP/extern/python/smartsim_cuda-12/bin/python" "$ROOT/analyze_forward_cpu_diagnostic.py" "$OUT" --runs "${runs[@]}" > "$OUT/summary.txt"
printf 'RESULTS %s\n' "$OUT"

#!/bin/bash
#SBATCH --job-name=PHYDLL_MPI_STARTUP
#SBATCH --partition=devel
#SBATCH --nodes=2
#SBATCH --ntasks=192
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench/diagnostics/logs/%x_%j.log

set -euo pipefail

BASE_DIR="/hpcwork/ro092286/smartsim"
BENCH_DIR="${BASE_DIR}/cpu_benchmark/provider_bench"
DIAG_DIR="${BENCH_DIR}/diagnostics"
BUILD_DIR="${DIAG_DIR}/build"
LOG_DIR="${DIAG_DIR}/logs/${SLURM_JOB_ID}"
MODE="${MPI_STRESS_MODE:-split}"
REPEATS="${MPI_STRESS_REPEATS:-30}"
SOLVER_RANKS="${MPI_SOLVER_RANKS:-96}"
TOTAL_RANKS=$((SOLVER_RANKS * 2))

mkdir -p "${BUILD_DIR}" "${LOG_DIR}"

cd "${BASE_DIR}/CPP-ML-Interface"
source ./set_env_claix23_cuda12.4.sh

mpicxx -O2 -std=c++17 \
    "${DIAG_DIR}/mpi_startup_probe.cpp" \
    -o "${BUILD_DIR}/mpi_startup_probe"

export MPI_PROBE_BIN="${BUILD_DIR}/mpi_startup_probe"
export MPI_SOLVER_RANKS

case "${MODE}" in
    split)
        if [[ "${SLURM_JOB_NUM_NODES}" -ne 2 || "${SLURM_NTASKS}" -ne "${TOTAL_RANKS}" ]]; then
            printf 'split mode requires two nodes and %s allocated tasks\n' "${TOTAL_RANKS}" >&2
            exit 2
        fi
        SRUN_ARGS=(
            --mpi=pmix_v3
            --nodes=2
            --ntasks="${TOTAL_RANKS}"
            --ntasks-per-node="${SOLVER_RANKS}"
            --distribution=block:block
            --cpu-bind=none
        )
        ;;
    oversub)
        if [[ "${SLURM_JOB_NUM_NODES}" -ne 1 ]]; then
            printf 'oversub mode requires one allocated node\n' >&2
            exit 2
        fi
        SRUN_ARGS=(
            --mpi=pmix_v3
            --nodes=1
            --ntasks="${TOTAL_RANKS}"
            --ntasks-per-node="${TOTAL_RANKS}"
            --overcommit
            --distribution=block:block
            --cpu-bind=none
        )
        ;;
    *)
        printf 'unknown MPI_STRESS_MODE=%s; use split or oversub\n' "${MODE}" >&2
        exit 2
        ;;
esac

printf 'mode=%s repeats=%s solver_ranks=%s total_ranks=%s nodes=%s\n' \
    "${MODE}" "${REPEATS}" "${SOLVER_RANKS}" "${TOTAL_RANKS}" "${SLURM_JOB_NODELIST}"

for attempt in $(seq 1 "${REPEATS}"); do
    output_file="${LOG_DIR}/run${attempt}.log"
    printf '=== attempt %s/%s mode=%s ===\n' "${attempt}" "${REPEATS}" "${MODE}"
    set +e
    timeout 120s srun "${SRUN_ARGS[@]}" \
        --export=ALL \
        "${DIAG_DIR}/mpi_rank_dispatch.sh" \
        >"${output_file}" 2>&1
    rc=$?
    set -e
    printf 'attempt=%s mode=%s rc=%s log=%s\n' "${attempt}" "${MODE}" "${rc}" "${output_file}"
done

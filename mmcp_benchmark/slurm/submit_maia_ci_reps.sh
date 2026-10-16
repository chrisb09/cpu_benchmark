#!/usr/bin/env bash
# Submit independent m-aia plain-timing repetitions for a run-mean latency CI.
# Reuses the frozen plain build so every rep matches the original rep-1 binary.
# One cell per job; each job is one exclusive 96-rank node.
set -euo pipefail
dry_run="${1:-}"
[[ -z "${dry_run}" || "${dry_run}" == --dry-run ]] || { printf 'Usage: submit_maia_ci_reps.sh [--dry-run]\n' >&2; exit 2; }

export PROJECT_DIR="$(realpath "$(dirname "$0")/../..")"
export ARTIFACT_DIR="${ARTIFACT_DIR:-/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference}"
export MAIA_BUILD_DIR="${MAIA_BUILD_DIR:-$ARTIFACT_DIR/maia/build_cpu_retest_plain}"
export MEASUREMENT=timing
export SOLVER_RANKS=96 NP_DL=96 SS_DB_BATCH=0
export MAIA_AIX_BATCHSIZE=50000
export RUN_STEPS=200 EXPECTED_INFERENCE_CALLS=7 SOLUTION_INTERVAL=10000 ML_STEP_TIMING_PRINT=1
export CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-14400}"
# Extra independent reps beyond the existing rep 1.
REPS="${REPS:-2 3 4 5}"

[[ -f "${MAIA_BUILD_DIR}/BUILD_READY" && -x "${MAIA_BUILD_DIR}/bin/maia" ]] || {
    printf 'Plain timing build missing: %s\n' "${MAIA_BUILD_DIR}" >&2; exit 2; }
exclude=()
[[ -z "${RETEST_EXCLUDE_NODES:-}" ]] || exclude=(--exclude="${RETEST_EXCLUDE_NODES}")

submit_cell() {
    local workload=$1 variant=$2 rep=$3
    export ML_CUBE_OVERLAP=0
    if [[ "${workload}" == transformer ]]; then
        export MODEL="${ARTIFACT_DIR}/input/transformer_inference_scripted_fw2.pt"
    else
        export MODEL="${ARTIFACT_DIR}/input/mmcp_test_mlp_m5_cpu.pt"
        [[ "${workload}" != mlp_ov4 ]] || export ML_CUBE_OVERLAP=4
    fi
    unset PHYDLL_MODE SS_CONFIG
    case "${variant}" in
        AIX) export PROVIDER=AIX ;;
        blocking|readiness) export PROVIDER=PHYDLL PHYDLL_MODE="${variant}" ;;
        SS_*) export PROVIDER=SMARTSIM SS_CONFIG="${variant}" ;;
        *) exit 2 ;;
    esac
    export CASE_NAME="ci_${workload}_${variant}_96p_r${rep}"
    local partition=c23ms
    if [[ "${workload}" == transformer && "${PROVIDER}" != SMARTSIM ]] || [[ "${workload}" == mlp_ov4 ]]; then
        partition=c23mm
    fi
    local cmd=(sbatch --parsable "${exclude[@]}" --account=rwth2150
        --partition="${partition}" --nodes=1 --ntasks=96 --cpus-per-task=1
        --exclusive --mem=0 --time="${RETEST_WALLTIME:-05:00:00}" --export=ALL
        --job-name="${CASE_NAME}" --output="${PROJECT_DIR}/mmcp_benchmark/logs/${CASE_NAME}_%j.out"
        --error="${PROJECT_DIR}/mmcp_benchmark/logs/${CASE_NAME}_%j.err"
        "${PROJECT_DIR}/mmcp_benchmark/slurm/cpu_retest.sbatch")
    if [[ -n "${dry_run}" ]]; then
        printf 'CELL %-42s provider=%-8s overlap=%s partition=%s\n' "${CASE_NAME}" "${PROVIDER}" "${ML_CUBE_OVERLAP}" "${partition}"
    else
        printf '%s %s %s\n' "$("${cmd[@]}")" "${partition}" "${CASE_NAME}"
    fi
}

count=0
mkdir -p "${PROJECT_DIR}/mmcp_benchmark/logs"
for workload in transformer mlp_ov4; do
    for variant in AIX blocking readiness SS_DEFAULT SS_TPQ1_I96_B96 SS_TPQ12_I8_B96 SS_TPQ96_I1_B96; do
        for rep in ${REPS}; do
            submit_cell "${workload}" "${variant}" "${rep}"
            count=$((count + 1))
        done
    done
done
printf 'MAIA_CI JOBS=%s BUILD=%s\n' "${count}" "${MAIA_BUILD_DIR}"

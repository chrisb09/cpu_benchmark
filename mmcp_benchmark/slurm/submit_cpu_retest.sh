#!/usr/bin/env bash
# One cell per job: kernel cgroup peaks cannot be attributed across sequential cells.
set -euo pipefail
measurement="${1:?Usage: bash submit_cpu_retest.sh profile|memory|timing BUILD_JOB_ID [--dry-run]}"
build_job="${2:?Supply the corresponding build job ID}"
dry_run="${3:-}"
[[ "${build_job}" =~ ^[0-9]+$ ]] || { printf 'Invalid build job ID\n' >&2; exit 2; }
[[ -z "${dry_run}" || "${dry_run}" == --dry-run ]] || exit 2
case "${measurement}" in profile|memory|timing) ;; *) exit 2 ;; esac
export MAIA_BUILD_DIR="${MAIA_BUILD_DIR:?Set MAIA_BUILD_DIR for this matrix}"
[[ "${MAIA_BUILD_DIR}" == /* ]] || { printf 'MAIA_BUILD_DIR must be absolute\n' >&2; exit 2; }
dependency=(--dependency="afterok:${build_job}")
if [[ "${build_job}" == 0 ]]; then
    [[ -f "${MAIA_BUILD_DIR}/BUILD_READY" && -x "${MAIA_BUILD_DIR}/bin/maia" ]] || {
        printf 'Completed build requires BUILD_READY and executable maia\n' >&2; exit 2;
    }
    dependency=()
fi
exclude=()
[[ -z "${RETEST_EXCLUDE_NODES:-}" ]] || exclude=(--exclude="${RETEST_EXCLUDE_NODES}")
script_dir="$(realpath "$(dirname "$0")")"
export PROJECT_DIR="$(realpath "${script_dir}/../..")"
export MEASUREMENT="${measurement}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference}"
cd "${script_dir}/.."
if [[ -z "${dry_run}" ]]; then mkdir -p logs; fi
count=0
for model in transformer mlp_ov0 mlp_ov4; do
    export ML_CUBE_OVERLAP=0
    export MODEL="${ARTIFACT_DIR}/input/transformer_inference_scripted_fw2.pt"
    if [[ "${model}" != transformer ]]; then
        export MODEL="${ARTIFACT_DIR}/input/mmcp_test_mlp_m5_cpu.pt"
        [[ "${model}" != mlp_ov4 ]] || export ML_CUBE_OVERLAP=4
    fi
    for variant in AIX blocking readiness SS_DEFAULT SS_TPQ1_I96_B96 SS_TPQ12_I8_B96 SS_TPQ96_I1_B96; do
        unset PHYDLL_MODE SS_CONFIG
        case "${variant}" in
            AIX) export PROVIDER=AIX ;;
            blocking|readiness) export PROVIDER=PHYDLL PHYDLL_MODE="${variant}" ;;
            SS_*) export PROVIDER=SMARTSIM SS_CONFIG="${variant}" ;;
        esac
        export CASE_NAME="retest_${measurement}_${model}_${variant}_96p"
        # Optional grep -E filter on the case name, for resubmitting a subset.
        if [[ -n "${RETEST_FILTER:-}" ]] && ! grep -Eq "${RETEST_FILTER}" <<<"${CASE_NAME}"; then
            continue
        fi
        partition=c23ms
        if [[ "${model}" == transformer && "${PROVIDER}" != SMARTSIM ]] ||
           [[ "${model}" == mlp_ov4 ]]; then
            partition=c23mm
        fi
        cmd=(sbatch --parsable "${dependency[@]}" "${exclude[@]}" --account=rwth2150
            --partition="${partition}" --nodes=1 --ntasks=96 --cpus-per-task=1
            --exclusive --mem=0 --time="${RETEST_WALLTIME:-05:00:00}" --export=ALL
            --job-name="${CASE_NAME}" --output="logs/${CASE_NAME}_%j.out"
            --error="logs/${CASE_NAME}_%j.err" "${script_dir}/cpu_retest.sbatch")
        count=$((count + 1))
        if [[ -n "${dry_run}" ]]; then
            printf 'CELL %s provider=%s overlap=%s partition=%s build=%s\n' "${CASE_NAME}" "${PROVIDER}" "${ML_CUBE_OVERLAP}" "${partition}" "${MAIA_BUILD_DIR}"
            printf '%q ' "${cmd[@]}"; printf '\n'
        else
            job="$("${cmd[@]}")"
            printf '%s %s %s\n' "${job}" "${partition}" "${CASE_NAME}"
        fi
    done
done
printf 'MATRIX=%s JOBS=%s BUILD_JOB=%s\n' "${measurement}" "${count}" "${build_job}"

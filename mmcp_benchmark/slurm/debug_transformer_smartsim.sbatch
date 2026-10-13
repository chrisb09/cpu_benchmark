#!/usr/bin/env bash
#SBATCH --job-name=debug-transformer-smartsim
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=230G
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/mmcp_benchmark/logs/debug_transformer_smartsim_%j.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/mmcp_benchmark/logs/debug_transformer_smartsim_%j.err
set -euo pipefail
variant="${1:?Usage: debug_transformer_smartsim.sbatch tpq96|default [--plan]}"
plan="${2:-}"
[[ -z "${plan}" || "${plan}" == --plan ]] || exit 2
case "${variant}" in
    tpq96) config=SS_TPQ96_I1_B96; probes=("80 3 600" "200 7 900") ;;
    default) config=SS_DEFAULT; probes=("20 1 2700") ;;
    *) exit 2 ;;
esac
if [[ "${plan}" == --plan ]]; then
    for probe in "${probes[@]}"; do
        read -r steps calls limit <<<"${probe}"
        printf 'config=%s steps=%s calls=%s case_s=%s command_s=3600 socket_ms=3600000 model_ms=3600000\n' "${config}" "${steps}" "${calls}" "${limit}"
    done
    exit 0
fi
: "${SLURM_JOB_ID:?Submit diagnostics with sbatch}"
export PROJECT_DIR="${PROJECT_DIR:-/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference}"
export MAIA_BUILD_DIR="${ARTIFACT_DIR}/maia/build_cpu_retest_plain"
[[ -f "${MAIA_BUILD_DIR}/BUILD_READY" ]] || exit 1
export MODEL="${ARTIFACT_DIR}/input/transformer_inference_scripted_fw2.pt"
export PROVIDER=SMARTSIM MEASUREMENT=timing SOLVER_RANKS=96 NP_DL=96
export ML_CUBE_OVERLAP=0 SOLUTION_INTERVAL=10000 ML_STEP_TIMING_PRINT=1
export MAIA_AIX_BATCHSIZE=50000 SS_DB_BATCH=0 SS_TIMEOUT_SECONDS=3600
export READY_TIMEOUT_SECONDS=180 SS_CONFIG="${config}"
# Freeze once for the whole diagnostic; each invocation creates a fresh Redis DB.
engine="$(<"${PROJECT_DIR}/mmcp_benchmark/slurm/cpu_engine.sh")"
printf 'DIAGNOSTIC_ONLY job=%s variant=%s engine_sha256=' "${SLURM_JOB_ID}" "${variant}"
printf '%s' "${engine}" | sha256sum
for probe in "${probes[@]}"; do
    read -r RUN_STEPS EXPECTED_INFERENCE_CALLS CASE_TIMEOUT_SECONDS <<<"${probe}"
    export RUN_STEPS EXPECTED_INFERENCE_CALLS CASE_TIMEOUT_SECONDS
    export CASE_NAME="debug_socketfix_${variant}_${RUN_STEPS}steps"
    printf 'PROBE_START case=%s expected_calls=%s limit_s=%s\n' "${CASE_NAME}" "${EXPECTED_INFERENCE_CALLS}" "${CASE_TIMEOUT_SECONDS}"
    bash -c "${engine}" cpu_engine_snapshot
    printf 'PROBE_SUCCESS case=%s\n' "${CASE_NAME}"
done

#!/usr/bin/env bash
# Submit the m-aia Phase-1 characterization (rebuild + run) on devel.
# devel uses the user's default account (no --account directive).
#
#   bash slurm/submit_characterization.sh --dry-run
#   bash slurm/submit_characterization.sh --submit
#   RANKS_LIST="24 96" STEPS=300 bash slurm/submit_characterization.sh --submit
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench_root="$(dirname "${here}")"
project_dir="${PROJECT_DIR:-/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference}"

ranks_list="${RANKS_LIST:-24 48 96}"
steps="${STEPS:-300}"
provider="${PROVIDER:-AIX}"
model_path="${MODEL_PATH:-${project_dir}/input/transformer_inference_scripted_fw2.pt}"

action=""
for arg in "$@"; do
    case "${arg}" in
        --dry-run|--submit) action="${arg}" ;;
        *) printf 'Unknown argument: %s\n' "${arg}" >&2; exit 2 ;;
    esac
done
[[ -n "${action}" ]] || { printf 'Usage: %s --dry-run|--submit\n' "$0" >&2; exit 2; }

mkdir -p "${bench_root}/logs"

build_job="$(sbatch --parsable --export=ALL,PROJECT_DIR="${project_dir}" \
    "${here}/rebuild_maia_devel.sbatch")"
printf 'rebuild job: %s\n' "${build_job}"

run_args="ALL,PROJECT_DIR=${project_dir},RANKS_LIST=${ranks_list},STEPS=${steps},PROVIDER=${provider},MODEL_PATH=${model_path}"
run_job="$(sbatch --parsable --dependency="afterok:${build_job}" --export="${run_args}" \
    --chdir="${bench_root}" "${here}/characterization_devel.sbatch")"
printf 'characterization job: %s (after %s)\n' "${run_job}" "${build_job}"

if [[ "${action}" == "--dry-run" ]]; then
    printf 'dry-run: cancelling submitted jobs %s %s\n' "${build_job}" "${run_job}"
    scancel "${build_job}" "${run_job}" || true
fi

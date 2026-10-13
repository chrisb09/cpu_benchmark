#!/usr/bin/env bash
# Submit the first job of a chained devel sweep (see chain_hook.sh).
#
#   bash slurm/chain_devel.sh <plan_file>
#
# Plan file lines: <sbatch-script>|<comma-separated exports>
# Example:
#   slurm/characterization_devel.sbatch|RANKS_LIST=24,STEPS=300,MODEL_PATH=/path/mlp.pt
#   slurm/characterization_devel.sbatch|RANKS_LIST=48,STEPS=300,MODEL_PATH=/path/mlp.pt
set -euo pipefail

plan="$(realpath "${1:?usage: chain_devel.sh <plan_file>}")"
bench_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
first="$(grep -vE '^[[:space:]]*(#|$)' "${plan}" | sed -n '1p')"
[[ -n "${first}" ]] || { printf 'Empty plan: %s\n' "${plan}" >&2; exit 1; }

script="${first%%|*}"
exports="${first#*|}"
[[ "${exports}" == "${first}" ]] && exports=""
mkdir -p "${bench_root}/logs"
sbatch --chdir="${bench_root}" \
    --export="ALL,CHAIN_PLAN=${plan},CHAIN_INDEX=0,CHAIN_CHDIR=${bench_root}${exports:+,${exports}}" \
    "${script}"

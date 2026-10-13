#!/usr/bin/env bash
# Source at the start of a devel batch script to self-submit the next job in a
# chain. Works around the devel 2-job-per-user submit limit: only the first job
# is submitted up front; each running job queues its successor with
# --dependency=afterany:<self>.
#
# Contract:
#   CHAIN_PLAN   absolute path to a plan file, one task per line:
#                <sbatch-script>|<comma-separated exports>
#                (blank lines and lines starting with '#' are ignored)
#   CHAIN_INDEX  0-based index of the current task (default 0)
#   CHAIN_CHDIR  working dir for the next submission (default SLURM_SUBMIT_DIR)
chain_maybe_submit_next() {
    [[ -n "${CHAIN_PLAN:-}" && -f "${CHAIN_PLAN}" ]] || return 0
    local next=$(( ${CHAIN_INDEX:-0} + 1 ))
    local line
    line="$(grep -vE '^[[:space:]]*(#|$)' "${CHAIN_PLAN}" | sed -n "$((next + 1))p")"
    [[ -n "${line}" ]] || return 0
    local script="${line%%|*}"
    local exports="${line#*|}"
    [[ "${exports}" == "${line}" ]] && exports=""
    printf '[chain] submitting next task %s: %s\n' "${next}" "${script}"
    sbatch --dependency="afterany:${SLURM_JOB_ID}" \
        --chdir="${CHAIN_CHDIR:-${SLURM_SUBMIT_DIR:-$PWD}}" \
        --export="ALL,CHAIN_PLAN=${CHAIN_PLAN},CHAIN_INDEX=${next},CHAIN_CHDIR=${CHAIN_CHDIR:-${SLURM_SUBMIT_DIR:-$PWD}}${exports:+,${exports}}" \
        "${script}"
}

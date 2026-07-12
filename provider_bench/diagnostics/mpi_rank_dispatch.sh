#!/bin/bash
set -euo pipefail

PROBE_BIN="${MPI_PROBE_BIN:?MPI_PROBE_BIN must point to mpi_startup_probe}"
PROC_ID="${SLURM_PROCID:?SLURM_PROCID is required}"
SOLVER_RANKS="${MPI_SOLVER_RANKS:-96}"

if (( PROC_ID < SOLVER_RANKS )); then
    ROLE=solver
else
    ROLE=dl
fi

printf '[dispatch] slurm_procid=%s role=%s host=%s\n' "${PROC_ID}" "${ROLE}" "$(hostname)"
exec "${PROBE_BIN}" "${ROLE}"

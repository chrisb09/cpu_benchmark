#!/bin/bash
set -e
rank=${OMPI_COMM_WORLD_RANK:?}
core=$((rank % 96))
taskset -p -c "$core" $$ >/dev/null
printf 'rank=%s paired_core=%s ' "$rank" "$core" > "$FORWARD_DIAG_DIR/binding_${rank}.txt"
taskset -p -c $$ >> "$FORWARD_DIAG_DIR/binding_${rank}.txt"
if [[ ${FORWARD_DIAG_MPI_PROVENANCE:-0} == 1 ]]; then
    env | sort > "$FORWARD_DIAG_DIR/environment_${rank}.txt"
fi
if (( rank < 96 )) && [[ -n ${FORWARD_DIAG_PACED_LIBRARY:-} ]]; then
    export LD_PRELOAD=$FORWARD_DIAG_PACED_LIBRARY
fi
exec "$@"

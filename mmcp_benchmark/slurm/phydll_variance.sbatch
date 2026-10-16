#!/usr/bin/env bash
# Resources supplied by submit_phydll_variance.sh. Slurm snapshots this wrapper.
set -euo pipefail
cd "${PROJECT_DIR:?}"
campaign="${CAMPAIGN_DIR:?}"
sha256sum --check "$campaign/provenance/runtime.sha256"
metadata="$campaign/metadata/$SLURM_JOB_ID"
mkdir -p "$metadata"
export SCRATCH_ROOT="$campaign/runs"
export CMI_BIND_LOG=1 CMI_SAMPLER_DIR="$metadata"
unset CMI_SAMPLER_SCRIPT CMI_NUMA_PIN_SCRIPT HET_SURROGATE_HIDE_GPU
export CMI_NUMA_PIN=0
lscpu > "$metadata/lscpu.txt"
scontrol show job "$SLURM_JOB_ID" > "$metadata/slurm_job.txt"
printf 'JOB=%s\nNODE=%s\nMODE=%s\nMODEL=%s\nNUMA_PIN=0\nSLURM_CPU_BIND=none\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$PHYDLL_MODE" "$MODEL" > "$metadata/launch.txt"
bash "$campaign/provenance/cpu_engine.sh"

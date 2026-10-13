#!/usr/bin/env bash
#SBATCH --job-name=mmcp-cpu-smoke
#SBATCH --partition=c23ms
#SBATCH --account=rwth2150
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem=230G
#SBATCH --output=logs/mmcp-cpu-smoke_%j.out
#SBATCH --error=logs/mmcp-cpu-smoke_%j.err

# Validation smoke: exercises the engine for one provider/config at 96 ranks.
# Set PROVIDER / MODEL / PHYDLL_MODE / SS_CONFIG via --export at submit time.

set -uo pipefail

project_dir="$(realpath "$(dirname "${SLURM_SUBMIT_DIR:-$(pwd)}")")"
cd "${project_dir}/mmcp_benchmark"
mkdir -p logs scratch/cpu_bench

export PROJECT_DIR="${project_dir}"
export ARTIFACT_DIR="/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference"
export SOLVER_RANKS="${SOLVER_RANKS:-96}"
export RUN_STEPS="${RUN_STEPS:-60}"
export SOLUTION_INTERVAL=10000
export ML_STEP_TIMING_PRINT=1
export ML_CUBE_OVERLAP="${ML_CUBE_OVERLAP:-0}"
export MODEL="${MODEL:-${ARTIFACT_DIR}/input/mmcp_test_mlp_m5_cpu.pt}"
export PROVIDER="${PROVIDER:-PHYDLL}"
export CASE_NAME="${CASE_NAME:-smoke_${PROVIDER}_96p}"

engine="${project_dir}/mmcp_benchmark/slurm/cpu_engine.sh"

echo "=== Smoke: PROVIDER=${PROVIDER} MODEL=$(basename "${MODEL}") RUN_STEPS=${RUN_STEPS} ==="
echo "Job ID: ${SLURM_JOB_ID}  Node: ${SLURM_NODELIST}"
date

bash "${engine}"
status=$?

echo "=== Smoke complete (status: ${status}) ==="
date
exit "${status}"

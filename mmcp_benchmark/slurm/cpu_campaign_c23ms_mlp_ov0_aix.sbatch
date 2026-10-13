#!/usr/bin/env bash
#SBATCH --job-name=mmcp-cpu-c23ms-mlp-ov0-aix
#SBATCH --partition=c23ms
#SBATCH --account=rwth2150
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem=230G
#SBATCH --output=logs/mmcp-cpu-c23ms-mlp-ov0-aix_%j.out
#SBATCH --error=logs/mmcp-cpu-c23ms-mlp-ov0-aix_%j.err

# Job 3: c23ms MLP overlap-0 AIX

set -uo pipefail

project_dir="$(realpath "$(dirname "${SLURM_SUBMIT_DIR:-$(pwd)}")")"
cd "${project_dir}/mmcp_benchmark"
mkdir -p logs scratch/cpu_bench

export PROJECT_DIR="${project_dir}"
export ARTIFACT_DIR="/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference"
export SOLVER_RANKS=96
export RUN_STEPS=200
export SOLUTION_INTERVAL=10000
export ML_STEP_TIMING_PRINT=1
export ML_CUBE_OVERLAP=0
export MODEL="${ARTIFACT_DIR}/input/mmcp_test_mlp_m5_cpu.pt"
export PROVIDER=AIX
export MAIA_AIX_BATCHSIZE=1000000
export CASE_NAME="c23ms_mlp_ov0_aix_96p"

engine="${project_dir}/mmcp_benchmark/slurm/cpu_engine.sh"

echo "=== Starting c23ms MLP overlap-0 AIX ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
date

bash "${engine}"
status=$?

echo "=== c23ms MLP overlap-0 AIX complete (status: ${status}) ==="
date
exit "${status}"

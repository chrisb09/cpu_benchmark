#!/usr/bin/env bash
#SBATCH --job-name=mmcp-cpu-c23mm-transformer
#SBATCH --partition=c23mm
#SBATCH --account=rwth2150
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=0
#SBATCH --output=logs/mmcp-cpu-c23mm-transformer_%j.out
#SBATCH --error=logs/mmcp-cpu-c23mm-transformer_%j.err

# Job 1: c23mm transformer AIX + PhyDLL (combined sequential)
# Runs AIX, PhyDLL-blocking, PhyDLL-readiness at 96 ranks.
# PhyDLL oversubscribes to 192 ranks via srun --overcommit.

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
export MODEL="${ARTIFACT_DIR}/input/transformer_inference_scripted_fw2.pt"

engine="${project_dir}/mmcp_benchmark/slurm/cpu_engine.sh"

echo "=== Starting c23mm transformer campaign: AIX + PhyDLL ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
date

# Run AIX
echo ">>> Running AIX transformer..."
export PROVIDER=AIX
export MAIA_AIX_BATCHSIZE=50000
export CASE_NAME="c23mm_transformer_aix_96p"
bash "${engine}"
aix_status=$?
echo "AIX completed with status ${aix_status}"

# Run PhyDLL blocking
echo ">>> Running PhyDLL transformer (blocking)..."
export PROVIDER=PHYDLL
export PHYDLL_MODE=blocking
export CASE_NAME="c23mm_transformer_phydll_blocking_96p"
bash "${engine}"
phydll_block_status=$?
echo "PhyDLL-blocking completed with status ${phydll_block_status}"

# Run PhyDLL readiness
echo ">>> Running PhyDLL transformer (readiness)..."
export PHYDLL_MODE=readiness
export CASE_NAME="c23mm_transformer_phydll_readiness_96p"
bash "${engine}"
phydll_ready_status=$?
echo "PhyDLL-readiness completed with status ${phydll_ready_status}"

echo "=== c23mm transformer campaign complete ==="
echo "AIX: ${aix_status}, PhyDLL-blocking: ${phydll_block_status}, PhyDLL-readiness: ${phydll_ready_status}"
date

if [[ "${aix_status}" != 0 ]]; then exit "${aix_status}"; fi
if [[ "${phydll_block_status}" != 0 ]]; then exit "${phydll_block_status}"; fi
exit "${phydll_ready_status}"

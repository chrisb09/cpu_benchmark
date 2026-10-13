#!/usr/bin/env bash
#SBATCH --job-name=mmcp-cpu-c23ms-mlp-ov0-phydll
#SBATCH --partition=c23ms
#SBATCH --account=rwth2150
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=230G
#SBATCH --output=logs/mmcp-cpu-c23ms-mlp-ov0-phydll_%j.out
#SBATCH --error=logs/mmcp-cpu-c23ms-mlp-ov0-phydll_%j.err

# Job 4: c23ms MLP overlap-0 PhyDLL (blocking + readiness)
# PhyDLL oversubscribes: 96 solver + 96 DL = 192 ranks via srun --overcommit

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
export PROVIDER=PHYDLL

engine="${project_dir}/mmcp_benchmark/slurm/cpu_engine.sh"

echo "=== Starting c23ms MLP overlap-0 PhyDLL ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
date

# Run PhyDLL blocking
echo ">>> Running PhyDLL MLP (blocking)..."
export PHYDLL_MODE=blocking
export CASE_NAME="c23ms_mlp_ov0_phydll_blocking_96p"
bash "${engine}"
blocking_status=$?
echo "PhyDLL-blocking completed with status ${blocking_status}"

# Run PhyDLL readiness
echo ">>> Running PhyDLL MLP (readiness)..."
export PHYDLL_MODE=readiness
export CASE_NAME="c23ms_mlp_ov0_phydll_readiness_96p"
bash "${engine}"
readiness_status=$?
echo "PhyDLL-readiness completed with status ${readiness_status}"

echo "=== c23ms MLP overlap-0 PhyDLL complete ==="
echo "Blocking: ${blocking_status}, Readiness: ${readiness_status}"
date

if [[ "${blocking_status}" != 0 ]]; then exit "${blocking_status}"; fi
exit "${readiness_status}"

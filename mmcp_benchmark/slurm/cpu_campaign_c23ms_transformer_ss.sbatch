#!/usr/bin/env bash
#SBATCH --job-name=mmcp-cpu-c23ms-transformer-ss
#SBATCH --partition=c23ms
#SBATCH --account=rwth2150
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --mem=230G
#SBATCH --output=logs/mmcp-cpu-c23ms-transformer-ss_%j.out
#SBATCH --error=logs/mmcp-cpu-c23ms-transformer-ss_%j.err

# Job 2: c23ms transformer SmartSim (4 variants)

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
export PROVIDER=SMARTSIM
export SS_DB_BATCH=0

engine="${project_dir}/mmcp_benchmark/slurm/cpu_engine.sh"

echo "=== Starting c23ms transformer SmartSim campaign (4 variants) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
date

variants=("SS_TPQ96_I1_B96" "SS_TPQ12_I8_B96" "SS_TPQ1_I96_B96" "SS_DEFAULT")
statuses=()

for ss_cfg in "${variants[@]}"; do
    echo ">>> Running SmartSim transformer: ${ss_cfg}..."
    export SS_CONFIG="${ss_cfg}"
    export CASE_NAME="c23ms_transformer_ss_${ss_cfg}_96p"
    bash "${engine}"
    status=$?
    statuses+=("${status}")
    echo "${ss_cfg} completed with status ${status}"
done

echo "=== c23ms transformer SmartSim campaign complete ==="
for i in "${!variants[@]}"; do
    echo "${variants[i]}: ${statuses[i]}"
done
date

# Exit with first non-zero status
for s in "${statuses[@]}"; do
    if [[ "${s}" != 0 ]]; then exit "${s}"; fi
done
exit 0

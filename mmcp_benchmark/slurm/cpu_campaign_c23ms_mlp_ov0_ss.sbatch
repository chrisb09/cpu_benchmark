#!/usr/bin/env bash
#SBATCH --job-name=mmcp-cpu-c23ms-mlp-ov0-ss
#SBATCH --partition=c23ms
#SBATCH --account=rwth2150
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem=230G
#SBATCH --output=logs/mmcp-cpu-c23ms-mlp-ov0-ss_%j.out
#SBATCH --error=logs/mmcp-cpu-c23ms-mlp-ov0-ss_%j.err

# Job 5: c23ms MLP overlap-0 SmartSim (4 variants)

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
export PROVIDER=SMARTSIM
export SS_DB_BATCH=0

engine="${project_dir}/mmcp_benchmark/slurm/cpu_engine.sh"

echo "=== Starting c23ms MLP overlap-0 SmartSim (4 variants) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
date

variants=("SS_DEFAULT" "SS_TPQ1_I96_B96" "SS_TPQ12_I8_B96" "SS_TPQ96_I1_B96")
statuses=()

for ss_cfg in "${variants[@]}"; do
    echo ">>> Running SmartSim MLP ov0: ${ss_cfg}..."
    export SS_CONFIG="${ss_cfg}"
    export CASE_NAME="c23ms_mlp_ov0_ss_${ss_cfg}_96p"
    bash "${engine}"
    status=$?
    statuses+=("${status}")
    echo "${ss_cfg} completed with status ${status}"
done

echo "=== c23ms MLP overlap-0 SmartSim complete ==="
for i in "${!variants[@]}"; do
    echo "${variants[i]}: ${statuses[i]}"
done
date

for s in "${statuses[@]}"; do
    if [[ "${s}" != 0 ]]; then exit "${s}"; fi
done
exit 0

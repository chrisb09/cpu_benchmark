#!/usr/bin/env bash
# Master submission script for the m-aia CPU benchmark campaign.
# 8 jobs covering provider x model x overlap (all 96 ranks, 200 steps):
#   1 c23mm : transformer AIX + PhyDLL
#   5 c23ms : transformer SmartSim, MLP ov0/ov6 x {AIX,PhyDLL,SmartSim}
# PhyDLL oversubscribes to 192 ranks via srun --overcommit inside a 96-core allocation.

set -euo pipefail

script_dir="$(realpath "$(dirname "$0")")"
cd "${script_dir}/.."
mkdir -p logs

echo "=== Submitting m-aia CPU benchmark campaign ==="
echo "8 jobs: 1 c23mm (transformer AIX+PhyDLL), 7 c23ms (rest)"
echo ""

jobs=()

echo "[1/8] c23mm transformer AIX + PhyDLL..."
jobs+=("$(sbatch --parsable slurm/cpu_campaign_c23mm_transformer.sbatch)")
echo "  Job ID: ${jobs[-1]}"

echo "[2/8] c23ms transformer SmartSim (4 variants)..."
jobs+=("$(sbatch --parsable slurm/cpu_campaign_c23ms_transformer_ss.sbatch)")
echo "  Job ID: ${jobs[-1]}"

echo "[3/8] c23ms MLP ov0 AIX..."
jobs+=("$(sbatch --parsable slurm/cpu_campaign_c23ms_mlp_ov0_aix.sbatch)")
echo "  Job ID: ${jobs[-1]}"

echo "[4/8] c23ms MLP ov0 PhyDLL..."
jobs+=("$(sbatch --parsable slurm/cpu_campaign_c23ms_mlp_ov0_phydll.sbatch)")
echo "  Job ID: ${jobs[-1]}"

echo "[5/8] c23ms MLP ov0 SmartSim..."
jobs+=("$(sbatch --parsable slurm/cpu_campaign_c23ms_mlp_ov0_ss.sbatch)")
echo "  Job ID: ${jobs[-1]}"

echo "[6/8] c23ms MLP ov6 AIX..."
jobs+=("$(sbatch --parsable slurm/cpu_campaign_c23ms_mlp_ov6_aix.sbatch)")
echo "  Job ID: ${jobs[-1]}"

echo "[7/8] c23ms MLP ov6 PhyDLL..."
jobs+=("$(sbatch --parsable slurm/cpu_campaign_c23mm_mlp_ov6_phydll.sbatch)")
echo "  Job ID: ${jobs[-1]}"

echo "[8/8] c23ms MLP ov6 SmartSim..."
jobs+=("$(sbatch --parsable slurm/cpu_campaign_c23mm_mlp_ov6_ss.sbatch)")
echo "  Job ID: ${jobs[-1]}"

echo ""
echo "=== All 8 jobs submitted ==="
echo "Job IDs: ${jobs[*]}"
echo ""
echo "Monitor with: squeue -j $(IFS=,; echo "${jobs[*]}")"
echo "Cancel all:   scancel $(IFS=' '; echo "${jobs[*]}")"
echo ""
echo "Results will be in: scratch/cpu_bench/"

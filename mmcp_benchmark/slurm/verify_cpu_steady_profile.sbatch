#!/usr/bin/env bash
#SBATCH --job-name=verify-cpu-steady-profile
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark/mmcp_benchmark/logs/verify_cpu_steady_profile_%j.out

set -Eeuo pipefail
export PROJECT_DIR=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/cpu_benchmark
export ARTIFACT_DIR=/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference
export MAIA_BUILD_DIR=${STEADY_BUILD_DIR:-$ARTIFACT_DIR/maia/build_cpu_steady_20261014_scorep}
export MEASUREMENT=profile PROFILE_SCOPE_VERSION=steady-v1 CMI_SCOREP_WARMUP_CALLS=2
export SOLVER_RANKS=96 RUN_STEPS=80 EXPECTED_INFERENCE_CALLS=3 ML_CUBE_OVERLAP=4
export MODEL=$ARTIFACT_DIR/input/mmcp_test_mlp_m5_cpu.pt
export PHYDLL_MODE=blocking NP_DL=96 SS_CONFIG=SS_TPQ96_I1_B96 SS_DB_BATCH=0
export ML_STEP_TIMING_PRINT=1 SOLUTION_INTERVAL=10000
export CASE_TIMEOUT_SECONDS=1800
export SCRATCH_ROOT=$PROJECT_DIR/mmcp_benchmark/scratch/steady_profile_$SLURM_JOB_ID
[[ ${SLURM_JOB_PARTITION:-} == c23mm ]] || exit 2
grep -Fxq PROFILE_SCOPE_VERSION=steady-v1 "$MAIA_BUILD_DIR/BUILD_READY"
mkdir -p "$SCRATCH_ROOT"
cp "$MAIA_BUILD_DIR/BUILD_READY" "$SCRATCH_ROOT/BUILD_READY"
status=0
for PROVIDER in AIX PHYDLL SMARTSIM; do
    export PROVIDER CASE_NAME=steady_smoke_${PROVIDER}_mlp_ov4_96p
    if bash "$PROJECT_DIR/mmcp_benchmark/slurm/cpu_engine.sh" > "$SCRATCH_ROOT/$PROVIDER.engine.log" 2>&1; then
        printf 'PROVIDER=%s\nENGINE_AND_VALIDATOR_STATUS=passed\nEXPECTED_STEADY_CALLS=1\n' "$PROVIDER" > "$SCRATCH_ROOT/$PROVIDER.PASSED"
    else
        rc=$?
        printf 'PROVIDER=%s\nEXIT_STATUS=%s\n' "$PROVIDER" "$rc" > "$SCRATCH_ROOT/$PROVIDER.FAILED"
        status=1
    fi
done
if (( status == 0 )); then
    printf 'PROFILE_SCOPE_VERSION=steady-v1\nWARMUP_CALLS=2\nEXPECTED_STEADY_CALLS=1\n' > "$SCRATCH_ROOT/SMOKE_PASSED"
fi
printf 'Smoke results: %s (status=%s)\n' "$SCRATCH_ROOT" "$status"
exit "$status"

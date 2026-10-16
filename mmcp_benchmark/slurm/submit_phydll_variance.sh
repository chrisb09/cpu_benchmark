#!/usr/bin/env bash
# Diagnostic profiles only; never feeds the publication selection.
set -euo pipefail
dry_run="${1:-}"
[[ -z "$dry_run" || "$dry_run" == --dry-run ]] || exit 2
export PROJECT_DIR="$(realpath "$(dirname "$0")/../..")"
export ARTIFACT_DIR=/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference
export MAIA_BUILD_DIR="$ARTIFACT_DIR/maia/build_cpu_steady_20261014_scorep"
export CAMPAIGN_DIR="${CAMPAIGN_DIR:-$PROJECT_DIR/diagnostics/phydll_variance_20261016}"
export PROVIDER=PHYDLL MEASUREMENT=profile PROFILE_SCOPE_VERSION=steady-v1
export SOLVER_RANKS=96 NP_DL=96 CMI_SCOREP_WARMUP_CALLS=2
export PHYDLL_DL_COUNT=2 PHYDLL_DL_FIELD_COUNT=2
export RUN_STEPS=200 EXPECTED_INFERENCE_CALLS=7 SOLUTION_INTERVAL=10000 ML_STEP_TIMING_PRINT=1
export MAIA_AIX_BATCHSIZE=50000 SS_DB_BATCH=0 CASE_TIMEOUT_SECONDS=14400
unset CMI_SAMPLER_SCRIPT CMI_NUMA_PIN_SCRIPT HET_SURROGATE_HIDE_GPU
export CMI_NUMA_PIN=0
[[ -x "$MAIA_BUILD_DIR/bin/maia" && -x "$MAIA_BUILD_DIR/bin/phydll_dl_client" ]] || exit 2
for marker in PROFILE_SCOPE_VERSION=steady-v1 PROFILE_WARMUP_CALLS=2; do
    grep -Fxq "$marker" "$MAIA_BUILD_DIR/BUILD_READY"
done
if [[ -z "$dry_run" ]]; then
    [[ ! -e "$CAMPAIGN_DIR" ]] || { printf 'Refusing duplicate submission: %s exists\n' "$CAMPAIGN_DIR" >&2; exit 2; }
    mkdir -p "$CAMPAIGN_DIR/logs" "$CAMPAIGN_DIR/provenance" "$CAMPAIGN_DIR/runs"
    cp "$PROJECT_DIR/mmcp_benchmark/slurm/cpu_engine.sh" "$CAMPAIGN_DIR/provenance/cpu_engine.sh"
    cp "$MAIA_BUILD_DIR/BUILD_READY" "$CAMPAIGN_DIR/provenance/BUILD_READY"
    cp "$ARTIFACT_DIR/slurm/pin_cmi_rank.sh" "$CAMPAIGN_DIR/provenance/pin_cmi_rank.sh"
    cp "$ARTIFACT_DIR/setup_env_claix23.sh" "$CAMPAIGN_DIR/provenance/setup_env_claix23.sh"
    sha256sum "$ARTIFACT_DIR/slurm/pin_cmi_rank.sh" "$ARTIFACT_DIR/setup_env_claix23.sh" \
        "$MAIA_BUILD_DIR/bin/maia" "$MAIA_BUILD_DIR/bin/phydll_dl_client" \
        > "$CAMPAIGN_DIR/provenance/runtime.sha256"
    printf 'job,workload,variant,rep,log\n' > "$CAMPAIGN_DIR/jobs.csv"
fi
for rep in 2 3 4 5; do
    for workload in transformer mlp_ov4; do
        export ML_CUBE_OVERLAP=0
        export MODEL="$ARTIFACT_DIR/input/transformer_inference_scripted_fw2.pt"
        if [[ "$workload" == mlp_ov4 ]]; then
            export MODEL="$ARTIFACT_DIR/input/mmcp_test_mlp_m5_cpu.pt" ML_CUBE_OVERLAP=4
        fi
        modes="blocking readiness"
        (( rep % 2 == 0 )) || modes="readiness blocking"
        for variant in $modes; do
            export PHYDLL_MODE="$variant" CASE_NAME="diag_${workload}_${variant}_r${rep}"
            if [[ -n "$dry_run" ]]; then
                printf 'CELL %s model=%s overlap=%s partition=c23mm\n' "$CASE_NAME" "$MODEL" "$ML_CUBE_OVERLAP"
                continue
            fi
            job="$(sbatch --parsable --account=rwth2150 --partition=c23mm --nodes=1 \
                --ntasks=96 --cpus-per-task=1 --exclusive --mem=0 --time=05:00:00 --export=ALL \
                --job-name="$CASE_NAME" --output="$CAMPAIGN_DIR/logs/${CASE_NAME}_%j.out" \
                --error="$CAMPAIGN_DIR/logs/${CASE_NAME}_%j.err" \
                "$PROJECT_DIR/mmcp_benchmark/slurm/phydll_variance.sbatch")"
            job="${job%%;*}"
            printf '%s,%s,%s,%s,%s\n' "$job" "$workload" "$variant" "$rep" \
                "$CAMPAIGN_DIR/logs/${CASE_NAME}_${job}.out" >> "$CAMPAIGN_DIR/jobs.csv"
            printf 'SUBMITTED %s %s\n' "$job" "$CASE_NAME"
        done
    done
done

#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=220G
#SBATCH --time=01:00:00
#SBATCH --job-name=mmcp-profile
#SBATCH --output=logs/mmcp-profile_%j.out
#SBATCH --error=logs/mmcp-profile_%j.err

# Score-P profiling smoke: AIX transformer, one rank count (default 48), with
# user-region profiling enabled (no tracing), plus ML_STEP_TIMING.
# devel uses the user's default account (no --account directive).
#
# Overrides:
#   RANKS_LIST="48"   rank counts (<=48 to fit devel)
#   STEPS=300
#   MODEL_PATH=...    transformer (default) or the staged MLP
set -euo pipefail

source "${CHAIN_HOOK:-/hpcwork/ro092286/smartsim/cpu_benchmark/mmcp_benchmark/slurm/chain_hook.sh}"
chain_maybe_submit_next

project_dir="${PROJECT_DIR:-/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference}"
maia_build="${MAIA_BUILD_DIR:-${project_dir}/maia/build_gnu_production_cmi_scorep}"
maia="${maia_build}/bin/maia"
[[ -x "${maia}" ]] || { printf 'Score-P MAIA not executable: %s\n' "${maia}" >&2; exit 1; }

ranks_list="${RANKS_LIST:-48}"
steps="${STEPS:-300}"
provider="${PROVIDER:-AIX}"
model_path="${MODEL_PATH:-${project_dir}/input/transformer_inference_scripted_fw2.pt}"
(( steps >= 51 )) || { printf 'STEPS must be >=51.\n' >&2; exit 2; }

out_root="${OUT_ROOT:-$(pwd)/results/profile_${SLURM_JOB_ID}}"
mkdir -p "${out_root}" logs

source "${project_dir}/setup_env_claix23.sh"
cuda_lib=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib
aix_prebuilt="${AIXELERATOR_PREBUILT_INSTALL_PREFIX:-${project_dir}/CPP-ML-Interface/extern/AIxeleratorService/INSTALL-SCOREP}"
export LD_LIBRARY_PATH="${aix_prebuilt}/lib:${maia_build}/lib:${project_dir}/CPP-ML-Interface/extern/phydll/build/lib:${cuda_lib}/stubs:${cuda_lib}:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=""
export CPP_ML_INTERFACE_PROVIDER_ENV="${provider}"
export CPP_ML_INTERFACE_DEVICE=CPU
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MLCOUPLING_INTRA_OP_THREADS=1 MLCOUPLING_INTER_OP_THREADS=1
export MAIA_ML_STEP_TIMING_SYNC=1
export MAIA_ML_STEP_TIMING_PRINT=1
export OMPI_MCA_pmix="^s1,s2"
unset OMPI_MCA_ess
export SCOREP_ENABLE_PROFILING=true
export SCOREP_ENABLE_TRACING=false
export SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY=true
unset FLOW_DEBUG_DUMP_DIR MAIA_SNAPSHOT_DIR MLCOUPLING_DEBUG_EXPORT AIX_P2P_TIMELINE_DIR

for ranks in ${ranks_list}; do
    run_dir="${out_root}/rank${ranks}"
    mkdir -p "${run_dir}/out" "${run_dir}/auxdata" "${run_dir}/boxes" "${run_dir}/planes" "${run_dir}/tmp"
    cd "${run_dir}"
    export TMPDIR="${run_dir}/tmp"
    export SCOREP_EXPERIMENT_DIRECTORY="${run_dir}/scorep"
    ln -sfn "${project_dir}/input" input
    ln -sfn "${project_dir}/input/grid_les_medium.hdf5" grid_les_medium.hdf5
    cp --reflink=auto "${project_dir}/input/restart_les_init_medium.hdf5" out/restart_les_ref_medium.hdf5
    cp "${project_dir}/input/properties_run_les_ref_medium.toml" properties.toml
    sed -i \
        -e "s|^timeSteps *=.*|timeSteps = ${steps}|" \
        -e "s|^hostFraction *=.*|hostFraction = \"1.00\"|" \
        -e "s|^mlInterval *=.*|mlInterval = 5|" \
        -e "s|^mlInputLength *=.*|mlInputLength = 5|" \
        -e "s|^mlStepCoefficient *=.*|mlStepCoefficient = 12|" \
        -e "s|^mlForecastWindow *=.*|mlForecastWindow = 2|" \
        -e "s|^mlScalingFactor *=.*|mlScalingFactor = 1|" \
        -e "s|^mlInputStepDistance *=.*|mlInputStepDistance = 1|" \
        -e "s|^mlCubeD *=.*|mlCubeD = 8|" \
        -e "s|^mlCubeOverlap *=.*|mlCubeOverlap = 0|" \
        -e "s|^modelPath *=.*|modelPath = \"./input/$(basename "${model_path}")\"|" \
        properties.toml
    config_name="config_$(printf '%s' "${provider}" | tr '[:upper:]' '[:lower:]').toml"
    cp "${project_dir}/${config_name}" "${config_name}"
    if [[ "${provider}" == AIX ]]; then
        sed -i '/^[[:space:]]*communication_mode[[:space:]]*=/d' "${config_name}"
        sed -i '/^\[library\]/a communication_mode = "collective"' "${config_name}"
    fi
    printf 'PROVIDER=%s\nSOLVER_RANKS=%s\nSTEPS=%s\nMODEL=%s\nMAIA=%s\n' \
        "${provider}" "${ranks}" "${steps}" "${model_path}" "${maia}" | tee run.info

    start=$(date +%s)
    set +e
    srun --label --mpi=pmix --nodes=1 --ntasks="${ranks}" --ntasks-per-node="${ranks}" \
        --cpus-per-task=1 --cpu-bind=cores --kill-on-bad-exit=1 \
        --mem="${SLURM_MEM_PER_NODE}M" \
        "${maia}" ./properties.toml > solver.log 2>&1
    status=$?
    set -e
    printf 'BENCHMARK_SOLVER_WALL_SECONDS=%s\nSOLVER_EXIT_STATUS=%s\n' \
        "$(( $(date +%s) - start ))" "${status}" | tee -a run.info
    printf 'COMPLETE rank%s status=%s wall=%ss\n' "${ranks}" "${status}" "$(( $(date +%s) - start ))"
    [[ "${status}" -eq 0 ]] || { printf 'Run failed at rank%s\n' "${ranks}" >&2; exit "${status}"; }
done

printf 'PROFILE_DONE out=%s\n' "${out_root}"

#!/usr/bin/env bash
# CPU benchmark engine for m-aia solver with AIX, PhyDLL, and SmartSim providers.
#
# Runs inside a single-node Slurm allocation that requests the node's full core
# count (--ntasks=96). PhyDLL oversubscription (np_solver + np_dl ranks) is
# realized with `srun --overcommit`, matching slurm/run_cmi_smoke_gpu.sh and
# slurm/run_phydll_cpp_300_devel.sh.
#
# Required env:
#   SOLVER_RANKS       number of solver MPI ranks
#   PROVIDER           AIX | PHYDLL | SMARTSIM
# Optional env (with defaults for the compact 200-step campaign):
#   MODEL              path to model file (default: transformer)
#   ML_CUBE_OVERLAP    cube overlap (default: 0)
#   MAIA_AIX_BATCHSIZE AIX batch size (default: unset)
#   RUN_STEPS          solver timesteps (default: 200)
#   SOLUTION_INTERVAL  HDF5 output interval (default: 10000, i.e. disabled)
#   ML_STEP_TIMING_PRINT  emit ML_STEP_TIMING lines (default: 1)
#   PHYDLL_MODE        blocking | readiness (default: blocking)
#   NP_DL              PhyDLL DL client ranks (default: SOLVER_RANKS)
#   SS_CONFIG          SmartSim config token (default: SS_DEFAULT)
#   SS_DB_BATCH        SmartSim DB batch_size (default: 0)
#   SS_TIMEOUT_SECONDS SmartSim command/socket/model wait (default: 3600)
#   CASE_TIMEOUT_SECONDS  solver timeout (default: 1800)
#   READY_TIMEOUT_SECONDS SmartSim DB readiness timeout (default: 180)
set -euo pipefail

project_dir="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
project_dir="$(realpath "${project_dir}")"

provider="${PROVIDER:?PROVIDER must be set to AIX, PHYDLL, or SMARTSIM}"
solver_ranks="${SOLVER_RANKS:?SOLVER_RANKS must be set}"
run_steps="${RUN_STEPS:-200}"
solution_interval="${SOLUTION_INTERVAL:-10000}"
timeout_s="${CASE_TIMEOUT_SECONDS:-1800}"
ready_s="${READY_TIMEOUT_SECONDS:-180}"
ss_timeout_s="${SS_TIMEOUT_SECONDS:-3600}"
ml_model_file="${MODEL:-./input/transformer_inference_scripted_fw2.pt}"
ml_cube_overlap="${ML_CUBE_OVERLAP:-0}"
ml_step_timing_print="${ML_STEP_TIMING_PRINT:-1}"
measurement="${MEASUREMENT:-timing}"
case "${measurement}" in
    profile|memory|timing) ;;
    *) printf 'Unknown MEASUREMENT: %s\n' "${measurement}" >&2; exit 2 ;;
esac

for v in "${solver_ranks}" "${run_steps}" "${solution_interval}" "${timeout_s}" "${ready_s}" "${ss_timeout_s}"; do
    [[ "${v}" =~ ^[0-9]+$ ]] || { printf 'Expected a non-negative integer: %s\n' "${v}" >&2; exit 2; }
done
(( solver_ranks > 0 && run_steps > 0 )) || exit 2
(( ss_timeout_s > 0 && ss_timeout_s <= 2147483 )) || exit 2
ss_timeout_ms=$((ss_timeout_s * 1000))
[[ "${ml_cube_overlap}" =~ ^[0-7]$ ]] || { printf 'ML_CUBE_OVERLAP must be 0..7.\n' >&2; exit 2; }

artifact_dir="${ARTIFACT_DIR:-/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference}"
artifact_dir="$(realpath "${artifact_dir}")"
maia_build_dir="${MAIA_BUILD_DIR:?Set MAIA_BUILD_DIR to the plain or Score-P build}"
maia_build_dir="$(realpath "${maia_build_dir}")"
maia="${maia_build_dir}/bin/maia"
[[ -x "${maia}" ]] || { printf 'MAIA not executable: %s\n' "${maia}" >&2; exit 1; }

source "${artifact_dir}/setup_env_claix23.sh"
cuda_lib=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib
cuda_stubs="${cuda_lib}/stubs"
export LD_LIBRARY_PATH="${maia_build_dir}/lib:${artifact_dir}/CPP-ML-Interface/extern/phydll/build/lib:${cuda_lib}:${cuda_stubs}:${LD_LIBRARY_PATH:-}"
# Resolve the AIX library from this build, rather than mixing plain and Score-P.
aix_lib_dir="$(sed -n 's/^AIXELERATOR_PREBUILT_LIB_DIR:PATH=//p; s/^AIXELERATOR_PREBUILT_LIB_DIR:STATIC=//p' "${maia_build_dir}/CMakeCache.txt")"
[[ -z "${aix_lib_dir}" ]] || export LD_LIBRARY_PATH="${aix_lib_dir}:${LD_LIBRARY_PATH}"

export CPP_ML_INTERFACE_DEVICE=CPU
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MLCOUPLING_INTRA_OP_THREADS=1 MLCOUPLING_INTER_OP_THREADS=1
export MAIA_ML_STEP_TIMING_SYNC=1
export MAIA_ML_STEP_TIMING_PRINT="${ml_step_timing_print}"
export PHYDLL_DL_COUNT="${PHYDLL_DL_COUNT:-2}" PHYDLL_DL_FIELD_COUNT="${PHYDLL_DL_FIELD_COUNT:-2}"
export OMPI_MCA_pmix="^s1,s2"
unset OMPI_MCA_ess
export SCOREP_ENABLE_TRACING=false SCOREP_ENABLE_PROFILING=false
unset SCOREP_EXPERIMENT_DIRECTORY SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY
if [[ "${measurement}" == profile ]]; then
    export SCOREP_ENABLE_PROFILING=true
fi
# Check the binaries, not just a speculative build-directory name.
check_instrumentation() {
    local symbols
    symbols="$(nm "$1" 2>/dev/null)" || { printf 'Cannot inspect binary: %s\n' "$1" >&2; exit 1; }
    if [[ "${symbols}" == *SCOREP_InitMeasurement* ]]; then
        [[ "${measurement}" == profile ]] || { printf 'Plain run requires an uninstrumented binary: %s\n' "$1" >&2; exit 1; }
    else
        [[ "${measurement}" != profile ]] || { printf 'Score-P instrumentation missing: %s\n' "$1" >&2; exit 1; }
    fi
}
check_instrumentation "${maia}"
unset FLOW_DEBUG_DUMP_DIR MAIA_SNAPSHOT_DIR MLCOUPLING_DEBUG_EXPORT
unset MLCOUPLING_DEBUG_ALL_RANKS MLCOUPLING_DEBUG_EXPORT_DIR MLCOUPLING_DEBUG_RANK
unset MLCOUPLING_DEBUG_MAX_INFERENCES DUMP_TENSOR_DIR AIX_P2P_TIMELINE_DIR
unset SSDB AIX_DIAGNOSTICS AIX_DIAGNOSTIC_BARRIERS AIX_P2P_INITIAL_CREDITS

# Provider-specific setup
case "${provider}" in
    AIX)
        export CPP_ML_INTERFACE_PROVIDER_ENV=AIX
        config_name="config_aix.toml"
        ;;
    PHYDLL)
        export CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL
        phydll_mode="${PHYDLL_MODE:-blocking}"
        np_dl="${NP_DL:-${solver_ranks}}"
        [[ "${phydll_mode}" == blocking || "${phydll_mode}" == readiness ]] || exit 2
        [[ "${np_dl}" =~ ^[1-9][0-9]*$ ]] || exit 2
        config_name="config_phydll.toml"
        ;;
    SMARTSIM)
        export CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM
        ss_config="${SS_CONFIG:-SS_DEFAULT}"
        ss_db_batch="${SS_DB_BATCH:-0}"
        config_name="config_smartsim.toml"
        ;;
    *)
        printf 'Unknown provider: %s\n' "${provider}" >&2
        exit 2
        ;;
esac

cpp_ml_config="${artifact_dir}/${config_name}"
[[ -r "${cpp_ml_config}" ]] || { printf 'Provider config not readable: %s\n' "${cpp_ml_config}" >&2; exit 1; }

scratch_root="${SCRATCH_ROOT:-${project_dir}/scratch/cpu_bench}"
mkdir -p "${scratch_root}"
scratch_root="$(realpath "${scratch_root}")"

case_name="${CASE_NAME:-${provider}_${solver_ranks}p}"
if [[ "${provider}" == PHYDLL ]]; then
    case_name="${case_name}_${phydll_mode}"
elif [[ "${provider}" == SMARTSIM ]]; then
    case_name="${case_name}_${ss_config}"
fi

run_dir="$(mktemp -d "${scratch_root}/${case_name}_${SLURM_JOB_ID}_XXXXXX")"
cd "${run_dir}"
if [[ "${measurement}" == profile ]]; then
    export SCOREP_EXPERIMENT_DIRECTORY="${run_dir}/scorep"
    export SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY=false
fi
mkdir out auxdata boxes planes tmp
export TMPDIR="${run_dir}/tmp"
ln -s "${artifact_dir}/input" input
ln -s "${artifact_dir}/input/grid_les_medium.hdf5" grid_les_medium.hdf5
cp --reflink=auto "${artifact_dir}/input/restart_les_init_medium.hdf5" out/restart_les_ref_medium.hdf5
cp "${artifact_dir}/input/properties_run_les_ref_medium.toml" properties.toml

# Patch properties.toml with benchmark settings
sed -i \
    -e "s/^timeSteps *=.*/timeSteps = ${run_steps}/" \
    -e 's/^hostFraction *=.*/hostFraction = "1.00"/' \
    -e 's/^mlInterval *=.*/mlInterval = 5/' \
    -e 's/^mlInputLength *=.*/mlInputLength = 5/' \
    -e 's/^mlStepCoefficient *=.*/mlStepCoefficient = 12/' \
    -e 's/^mlForecastWindow *=.*/mlForecastWindow = 2/' \
    -e 's/^mlScalingFactor *=.*/mlScalingFactor = 1/' \
    -e 's/^mlInputStepDistance *=.*/mlInputStepDistance = 1/' \
    -e "s|^modelPath *=.*|modelPath = \"${ml_model_file}\"|" \
    -e "s/^mlCubeOverlap *=.*/mlCubeOverlap = ${ml_cube_overlap}/" \
    -e "s/^solutionInterval *=.*/solutionInterval = ${solution_interval}/" properties.toml

model_path="${ml_model_file}"
[[ -r "${model_path}" ]] || { printf 'Input model not readable: %s\n' "${model_path}" >&2; exit 1; }

# Copy and patch provider config
cp "${cpp_ml_config}" "${config_name}"
if [[ "${provider}" == PHYDLL ]]; then
    sed -i '/^[[:space:]]*solver_readiness_wait[[:space:]]*=/d' "${config_name}"
    if [[ "${phydll_mode}" == readiness ]]; then
        sed -i '/^\[library\]/a solver_readiness_wait = true' "${config_name}"
    else
        sed -i '/^\[library\]/a solver_readiness_wait = false' "${config_name}"
    fi
elif [[ "${provider}" == SMARTSIM ]]; then
    sed -i \
        -e "s/^batch_size *=.*/batch_size = ${ss_db_batch}/" \
        -e "s/^min_batch_size *=.*/min_batch_size = 0/" \
        -e "s/^min_batch_timeout *=.*/min_batch_timeout = 0/" \
        -e "s/^command_timeout *=.*/command_timeout = ${ss_timeout_s}/" \
        -e "s/^socket_timeout *=.*/socket_timeout = ${ss_timeout_ms}/" \
        -e "s/^model_timeout *=.*/model_timeout = ${ss_timeout_ms}/" "${config_name}"
fi

# Write run metadata
{
    printf 'CASE=%s\n' "${case_name}"
    printf 'RUN_DIR=%s\n' "${run_dir}"
    printf 'MAIA=%s\n' "${maia}"
    printf 'PROVIDER=%s\n' "${provider}"
    printf 'MEASUREMENT=%s\n' "${measurement}"
    printf 'WARMUP_CALLS=2\nEXPECTED_INFERENCE_CALLS=%s\n' "${EXPECTED_INFERENCE_CALLS:-$(( (run_steps + 13) / 29 ))}"
    printf 'SCOREP_EXPERIMENT_DIRECTORY=%s\n' "${SCOREP_EXPERIMENT_DIRECTORY:-disabled}"
    printf 'MODEL=%s\n' "${model_path}"
    printf 'SOLVER_RANKS=%s\n' "${solver_ranks}"
    printf 'RUN_STEPS=%s\n' "${run_steps}"
    printf 'SOLUTION_INTERVAL=%s\n' "${solution_interval}"
    printf 'ML_CUBE_OVERLAP=%s\n' "${ml_cube_overlap}"
    printf 'MAIA_AIX_BATCHSIZE=%s\n' "${MAIA_AIX_BATCHSIZE:-default}"
    printf 'MAIA_ML_STEP_TIMING_PRINT=%s\n' "${ml_step_timing_print}"
    if [[ "${provider}" == PHYDLL ]]; then
        printf 'PHYDLL_MODE=%s\n' "${phydll_mode}"
        printf 'DL_RANKS=%s\n' "${np_dl}"
        printf 'PHYDLL_DL_COUNT=%s\n' "${PHYDLL_DL_COUNT}"
    elif [[ "${provider}" == SMARTSIM ]]; then
        printf 'SS_CONFIG=%s\n' "${ss_config}"
        printf 'SS_DB_BATCH=%s\n' "${ss_db_batch}"
        printf 'SS_TIMEOUT_SECONDS=%s\n' "${ss_timeout_s}"
        printf 'SS_COMMAND_TIMEOUT_SECONDS=%s\n' "${ss_timeout_s}"
        printf 'SS_SOCKET_TIMEOUT_MS=%s\n' "${ss_timeout_ms}"
        printf 'SS_MODEL_TIMEOUT_MS=%s\n' "${ss_timeout_ms}"
    fi
    printf 'SLURM_CPUS_ON_NODE=%s\n' "${SLURM_CPUS_ON_NODE:-unknown}"
} | tee run.info

# Host-memory peak for the whole job cgroup. Compute nodes are cgroup v1 (memory
# controller), where the kernel tracks the peak in memory.max_usage_in_bytes; we
# also sample the current usage every 2 s as a fallback.
mem_pid=""
mem_peak_file=""
if [[ "${measurement}" == memory ]]; then
mem_file=".mem_peak"
echo 0 > "${mem_file}"
uid="$(id -u)"
job_cg="/sys/fs/cgroup/memory/slurm/uid_${uid}/job_${SLURM_JOB_ID}"
mem_cur=""; mem_peak_file=""
if [[ -r "${job_cg}/memory.usage_in_bytes" ]]; then
    mem_cur="${job_cg}/memory.usage_in_bytes"
    mem_peak_file="${job_cg}/memory.max_usage_in_bytes"
else
    cg_path="$(sed -n 's/^0:://p' /proc/self/cgroup)"
    cg_mem="$(awk -F: '$2 ~ /(^|,)memory(,|$)/ {print $3}' /proc/self/cgroup)"
    # Only accept the job ancestor, never a task cgroup or the node root.
    for base in "/sys/fs/cgroup/memory${cg_mem}" "/sys/fs/cgroup${cg_path}"; do
        while [[ "${base}" != / && "${base}" != /sys/fs/cgroup ]]; do
            if [[ "${base##*/}" == "job_${SLURM_JOB_ID}" ]]; then
                for cand in "${base}/memory.usage_in_bytes" "${base}/memory.current"; do
                    [[ ! -r "${cand}" ]] || { mem_cur="${cand}"; break; }
                done
                break
            fi
            base="${base%/*}"
        done
        [[ -z "${mem_cur}" ]] || break
    done
    case "${mem_cur}" in
        *.usage_in_bytes) mem_peak_file="${mem_cur%usage_in_bytes}max_usage_in_bytes" ;;
        */memory.current) mem_peak_file="${mem_cur%current}peak" ;;
    esac
fi
printf 'MEM_CGROUP_PATH=%s\n' "${mem_cur:-unreadable}" >> run.info
printf 'MEM_CGROUP_PEAK_PATH=%s\n' "${mem_peak_file:-none}" >> run.info
mem_sampler() {
    [[ -n "${mem_cur}" ]] || return 0
    local peak=0 v
    while true; do
        v="$(cat "${mem_cur}" 2>/dev/null || echo 0)"
        if [[ "${v}" =~ ^[0-9]+$ ]] && (( v > peak )); then peak="${v}"; echo "${peak}" > "${mem_file}"; fi
        sleep 2
    done
}
mem_sampler & mem_pid=$!
printf 'MEMORY_SCOPE=single-cell-job-cgroup-including-startup-and-page-cache\n' >> run.info
fi

controller_pid=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${mem_pid}" ]]; then
        kill "${mem_pid}" 2>/dev/null || true
        wait "${mem_pid}" 2>/dev/null || true
    fi
    if [[ -n "${controller_pid}" ]]; then
        touch .solver_done
        for ((i = 0; i < 30; ++i)); do
            kill -0 "${controller_pid}" 2>/dev/null || break
            sleep 1
        done
        kill -TERM -- "-${controller_pid}" 2>/dev/null || true
        wait "${controller_pid}" 2>/dev/null || true
    fi
    if [[ "${measurement}" == memory ]]; then
    local mem_peak kp
    mem_peak="$(cat "${mem_file}" 2>/dev/null || echo 0)"
    if [[ -n "${mem_peak_file}" && -r "${mem_peak_file}" ]]; then
        kp="$(cat "${mem_peak_file}" 2>/dev/null || echo 0)"
        [[ "${kp}" =~ ^[0-9]+$ ]] && (( kp > mem_peak )) && mem_peak="${kp}"
    fi
    if [[ -z "${mem_cur}" ]]; then
        printf 'MEMORY_PEAK_BYTES=unavailable\nMEMORY_VALID=false\n' >> run.info
        (( status != 0 )) || status=1
    else
        printf 'MEMORY_PEAK_BYTES=%s\nMEMORY_VALID=true\n' "${mem_peak}" >> run.info
    fi
    fi
    printf 'EXIT_STATUS=%s\n' "${status}" >> run.info
    printf 'Preserved bench artifacts: %s\n' "${run_dir}"
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

pin="${artifact_dir}/slurm/pin_cmi_rank.sh"
solver_cmd=(srun --label --mpi=pmix --nodes=1 --overcommit
    --cpu-bind=none --kill-on-bad-exit=1)

if [[ "${provider}" == PHYDLL ]]; then
    dl_client="${maia_build_dir}/CPP-ML-Interface/dl_clients/phydll_dl_client"
    [[ -x "${dl_client}" ]] || dl_client="${maia_build_dir}/bin/phydll_dl_client"
    [[ -x "${dl_client}" ]] || { printf 'C++ PhyDLL client missing in %s\n' "${maia_build_dir}" >&2; exit 1; }
    check_instrumentation "${dl_client}"
    total_tasks=$((solver_ranks + np_dl))
    solver_cmd+=(--ntasks="${total_tasks}" --wait=0)
    export PHYDLL_MPMD_SHUTDOWN_BARRIER=1
    printf '0-%s /bin/bash %s %s ./properties.toml --cpp-ml-config %s\n%s-%s /bin/bash %s %s\n' \
        "$((solver_ranks - 1))" "${pin}" "${maia}" "${config_name}" \
        "${solver_ranks}" "$((total_tasks - 1))" "${pin}" "${dl_client}" > phydll.conf
    solver_cmd+=(--multi-prog ./phydll.conf)
elif [[ "${provider}" == SMARTSIM ]]; then
    smart_env="${SMARTSIM_PYTHON_ENV:-/hpcwork/$(whoami)/smartsim/python/smartsim_cpu/bin/activate}"
    [[ -r "${smart_env}" ]] || { printf 'SmartSim environment missing: %s\n' "${smart_env}" >&2; exit 1; }
    source "${smart_env}"
    # SmartRedis commands use seconds; socket replies and model limits use ms.
    export SR_CMD_TIMEOUT="${ss_timeout_s}" SR_SOCKET_TIMEOUT="${ss_timeout_ms}" SR_MODEL_TIMEOUT="${ss_timeout_ms}"
    export SMARTSIM_MPI_SEQUENTIAL_PUT=0
    export SR_LOG_LEVEL=INFO SR_LOG_FILE="${run_dir}/smartredis.log"

    case "${ss_config}" in
        SS_DEFAULT)        ss_args=(--use-default-cpu-settings) ;;
        SS_TPQ1_I96_B96)   ss_args=(--threads-per-queue 1  --intra-op-threads 96 --inter-op-threads 1 --cpu-cores-per-node "${solver_ranks}") ;;
        SS_TPQ12_I8_B96)   ss_args=(--threads-per-queue 12 --intra-op-threads 8  --inter-op-threads 1 --cpu-cores-per-node "${solver_ranks}") ;;
        SS_TPQ96_I1_B96)   ss_args=(--threads-per-queue 96 --intra-op-threads 1  --inter-op-threads 1 --cpu-cores-per-node "${solver_ranks}") ;;
        *) printf 'Unknown SS_CONFIG: %s\n' "${ss_config}" >&2; exit 2 ;;
    esac

    timeout --signal=TERM --kill-after=10 "$((timeout_s + ready_s + 60))s" \
        python3 "${artifact_dir}/CPP-ML-Interface/dl_clients/smartsim_controller.py" \
        --launcher local --interface lo --db-nodes 1 --auto-port \
        "${ss_args[@]}" \
        --timeout-s "${ready_s}" \
        --endpoint-file .ssdb_endpoint --done-file .solver_done --exp-dir ./ssdb_exp \
        > controller.log 2>&1 &
    controller_pid=$!
    for ((i = 0; i < ready_s; ++i)); do
        [[ ! -s .ssdb_endpoint ]] || break
        kill -0 "${controller_pid}" 2>/dev/null || { printf 'SmartSim controller exited; see controller.log.\n' >&2; exit 1; }
        sleep 1
    done
    [[ -s .ssdb_endpoint ]] || { printf 'SmartSim readiness timeout; see controller.log.\n' >&2; exit 1; }
    IFS= read -r SSDB < .ssdb_endpoint
    export SSDB
    solver_cmd+=(--ntasks="${solver_ranks}" /bin/bash "${pin}" "${maia}" ./properties.toml --cpp-ml-config "${config_name}")
else
    solver_cmd+=(--ntasks="${solver_ranks}" /bin/bash "${pin}" "${maia}" ./properties.toml --cpp-ml-config "${config_name}")
fi

start_seconds=$(date +%s)
set +e
timeout --signal=TERM --kill-after=20 "${timeout_s}s" "${solver_cmd[@]}" > solver.log 2>&1
status=$?
set -e
printf 'BENCHMARK_SOLVER_WALL_SECONDS=%s\nSOLVER_EXIT_STATUS=%s\n' \
    "$(( $(date +%s) - start_seconds ))" "${status}" | tee -a run.info
if (( status == 0 )); then
    python3 "${project_dir}/mmcp_benchmark/verify_cpu_run.py" "${run_dir}" || status=$?
fi
exit "${status}"

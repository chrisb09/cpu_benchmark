#!/bin/bash
#SBATCH --job-name=devel_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --output=/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench/results_devel/logs/devel_sweep_%j.log
#SBATCH --time=01:00:00
#SBATCH --partition=devel
#SBATCH --mem=238G
#SBATCH --account=default

set -euo pipefail

BASE_DIR="/hpcwork/ro092286/smartsim"
BENCH_DIR="${BASE_DIR}/cpu_benchmark/provider_bench"
OUT_DIR="${BENCH_DIR}/results_devel"
LOGS_DIR="${OUT_DIR}/logs"
STATE_DIR="${OUT_DIR}/state"
BUILD_DIR="${PROVIDER_BENCH_BUILD_DIR:-${BENCH_DIR}/build-provider-bench}"
SOLVER_BIN="${BUILD_DIR}/benchmark_solver"
DL_CLIENT_CPP="${PHYDLL_DL_CLIENT:-${BUILD_DIR}/dl-client/phydll_dl_client}"
DL_CLIENT_PY="${BASE_DIR}/CPP-ML-Interface/dl_clients/phydll_dl_client.py"
PYTHON_BIN="${BASE_DIR}/CPP-ML-Interface/extern/python/smartsim_cpu/bin/python"
PHYDLL_LIB_DIR="${BASE_DIR}/CPP-ML-Interface/extern/phydll/build/lib"
CUDA_STUB_SOURCE="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64/libcuda.so"
NVML_STUB_SOURCE="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64/libnvidia-ml.so"
CUDA_STUB_DIR="${BUILD_DIR}/cuda_stubs"

RESULTS_CSV="${OUT_DIR}/provider_sweep.csv"
COMBINATIONS_FILE="${STATE_DIR}/combinations.txt"
INDEX_FILE="${STATE_DIR}/.current_idx"

mkdir -p "${LOGS_DIR}" "${STATE_DIR}" "${CUDA_STUB_DIR}"
ln -sf "${CUDA_STUB_SOURCE}" "${CUDA_STUB_DIR}/libcuda.so"
ln -sf "${CUDA_STUB_SOURCE}" "${CUDA_STUB_DIR}/libcuda.so.1"
ln -sf "${NVML_STUB_SOURCE}" "${CUDA_STUB_DIR}/libnvidia-ml.so"
ln -sf "${NVML_STUB_SOURCE}" "${CUDA_STUB_DIR}/libnvidia-ml.so.1"

cd "${BASE_DIR}/CPP-ML-Interface" && source ./set_env_claix23_cuda12.4.sh

export LD_LIBRARY_PATH="${PHYDLL_LIB_DIR}:${CUDA_STUB_DIR}:${LD_LIBRARY_PATH:-}"
export MLCOUPLING_LOG_LEVEL=DEBUG
export SR_MODEL_TIMEOUT=2000000
export SR_CMD_TIMEOUT=2000000
export SR_SOCKET_TIMEOUT=2000000

get_job_cgroup_mem() {
    local uid="${SLURM_UID:-$(id -u)}"
    local jobid="${SLURM_JOB_ID:-}"
    if [ -n "${jobid}" ]; then
        local cg_path="/sys/fs/cgroup/memory/slurm/uid_${uid}/job_${jobid}/memory.max_usage_in_bytes"
        if [ -f "${cg_path}" ]; then
            local bytes
            bytes=$(cat "${cg_path}")
            echo "$(( bytes / 1024 / 1024 ))"
            return
        fi
    fi
    echo "-1"
}

if [ ! -f "${RESULTS_CSV}" ] || ! grep -q '^label,' "${RESULTS_CSV}"; then
    echo "label,model,provider,tpq,intra_threads,bind_cores,cold_time_s,warm_time_s,solver_rss_mb,job_mem_mb,status" > "${RESULTS_CSV}"
fi

# Build combinations file if missing
if [ ! -f "${COMBINATIONS_FILE}" ]; then
    PROVIDERS=(
        "AIX|N/A|N/A|N/A|none"
        "PHYDLL_CPP_DLR96_I1|96|1|0|phydll_cpp"
        "PHYDLL_PY_DLR96_I1|96|1|0|phydll_py"
        "SS_DEFAULT|N/A|N/A|-1|smartsim_default"
        "SS_TPQ1_I96_B96|1|96|96|smartsim_tpq1_i96_b96"
        "SS_TPQ1_I96_B96_BATCH100|1|96|96|smartsim_tpq1_i96_b96_batch100"
        "SS_TPQ96_I1_B96|96|1|96|smartsim_tpq96_i1_b96"
    )
    MODELS=(
        "watercnn|10000000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/watercnn_cpu.pt|10000000"
        "mmcp_transformer|10000|mmcp|${BASE_DIR}/MMCP_TOM/input/transformer_inference_scripted_fw2.pt|30000"
    )
    rm -f "${COMBINATIONS_FILE}"
    for p_info in "${PROVIDERS[@]}"; do
        PNAME=$(echo "$p_info" | cut -d'|' -f1)
        TPQ=$(echo "$p_info" | cut -d'|' -f2)
        INTRA=$(echo "$p_info" | cut -d'|' -f3)
        BIND=$(echo "$p_info" | cut -d'|' -f4)
        KIND=$(echo "$p_info" | cut -d'|' -f5)
        for m_info in "${MODELS[@]}"; do
            MODEL_NAME=$(echo "$m_info" | cut -d'|' -f1)
            INPUTS=$(echo "$m_info" | cut -d'|' -f2)
            SCHEMA=$(echo "$m_info" | cut -d'|' -f3)
            MODEL_PATH=$(echo "$m_info" | cut -d'|' -f4)
            BATCH_DIM=$(echo "$m_info" | cut -d'|' -f5)
            for rep in $(seq 1 10); do
                echo "${MODEL_NAME}|${INPUTS}|${SCHEMA}|${MODEL_PATH}|${PNAME}|${TPQ}|${INTRA}|${BIND}|${KIND}|${BATCH_DIM}|${rep}" >> "${COMBINATIONS_FILE}"
            done
        done
    done
fi

TOTAL_TASKS=$(wc -l < "${COMBINATIONS_FILE}")
if [ ! -f "${INDEX_FILE}" ]; then echo "1" > "${INDEX_FILE}"; fi
IDX=$(cat "${INDEX_FILE}")

if [ "${IDX}" -gt "${TOTAL_TASKS}" ]; then
    echo "All ${TOTAL_TASKS} benchmark tasks completed!"
    rm -f "${INDEX_FILE}"
    exit 0
fi

COMBO=$(sed -n "${IDX}p" "${COMBINATIONS_FILE}")
IFS='|' read -r MODEL_NAME INPUTS SCHEMA MODEL_PATH PNAME TPQ INTRA BIND KIND BATCH_DIM REP <<< "${COMBO}"

LABEL="${MODEL_NAME}_${PNAME}"
OUTPUT_FILE="${LOGS_DIR}/${LABEL}_rep${REP}_job${SLURM_JOB_ID}.log"

echo "=========================================================="
echo "[Task ${IDX}/${TOTAL_TASKS}] Model: ${MODEL_NAME} | Provider: ${PNAME} | Rep: ${REP}/10"

set +e
RC=0

if [ "${KIND}" = "none" ]; then
    # AIX
    mpirun -n 96 "${SOLVER_BIN}" \
        --provider AIX --model "${MODEL_PATH}" --schema "${SCHEMA}" --inputs "${INPUTS}" \
        > "${OUTPUT_FILE}" 2>&1
    RC=$?

elif [ "${KIND}" = "phydll_cpp" ]; then
    export MLCOUPLING_INTRA_OP_THREADS=1
    export MLCOUPLING_INTER_OP_THREADS=1
    export PHYDLL_DL_COUNT=1
    export PHYDLL_DL_FIELD_COUNT=1
    timeout 600 mpirun --oversubscribe --bind-to none \
        -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
        -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
        -n 96 "${SOLVER_BIN}" \
        --provider PHYDLL --model "${MODEL_PATH}" --schema "${SCHEMA}" --inputs "${INPUTS}" \
        : -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
        -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
        -n 96 "${DL_CLIENT_CPP}" \
        > "${OUTPUT_FILE}" 2>&1
    RC=$?

elif [ "${KIND}" = "phydll_py" ]; then
    export MLCOUPLING_INTRA_OP_THREADS=1
    export MLCOUPLING_INTER_OP_THREADS=1
    export PHYDLL_DL_COUNT=1
    export PHYDLL_DL_FIELD_COUNT=1
    timeout 600 mpirun --oversubscribe --bind-to none \
        -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
        -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
        -n 96 "${SOLVER_BIN}" \
        --provider PHYDLL --model "${MODEL_PATH}" --schema "${SCHEMA}" --inputs "${INPUTS}" \
        : -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
        -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
        -n 96 "${PYTHON_BIN}" "${DL_CLIENT_PY}" \
        > "${OUTPUT_FILE}" 2>&1
    RC=$?

elif [[ "${KIND}" == smartsim_* ]]; then
    ENDPOINT_FILE="${STATE_DIR}/.ssdb_endpoint_${SLURM_JOB_ID}"
    DONE_FILE="${STATE_DIR}/.solver_done_${SLURM_JOB_ID}"
    rm -f "${ENDPOINT_FILE}" "${DONE_FILE}"

    CTRL_ARGS="--auto-port --endpoint-file ${ENDPOINT_FILE} --done-file ${DONE_FILE} --exp-dir ${OUT_DIR}/smartsim_experiments_${SLURM_JOB_ID} --silent"
    SOLVER_MPI_ARGS="-n 96"
    EXTRA_SOLVER_ARGS=""

    if [ "${KIND}" = "smartsim_default" ]; then
        CTRL_ARGS="${CTRL_ARGS} --use-default-cpu-settings"
    elif [ "${KIND}" = "smartsim_tpq1_i96_b96" ]; then
        CTRL_ARGS="${CTRL_ARGS} --threads-per-queue 1 --intra-op-threads 96 --inter-op-threads 1 --cpu-cores-per-node 96"
        SOLVER_MPI_ARGS="--bind-to core --map-by core -n 96"
    elif [ "${KIND}" = "smartsim_tpq1_i96_b96_batch100" ]; then
        CTRL_ARGS="${CTRL_ARGS} --threads-per-queue 1 --intra-op-threads 96 --inter-op-threads 1 --cpu-cores-per-node 96"
        SOLVER_MPI_ARGS="--bind-to core --map-by core -n 96"
        EXTRA_SOLVER_ARGS="--batch-size ${BATCH_DIM} --min-batch-size ${BATCH_DIM} --min-batch-timeout 5000"
    elif [ "${KIND}" = "smartsim_tpq96_i1_b96" ]; then
        CTRL_ARGS="${CTRL_ARGS} --threads-per-queue 96 --intra-op-threads 1 --inter-op-threads 1 --cpu-cores-per-node 96"
        SOLVER_MPI_ARGS="--bind-to core --map-by core -n 96"
    fi

    ${PYTHON_BIN} "${BASE_DIR}/CPP-ML-Interface/dl_clients/smartsim_controller.py" ${CTRL_ARGS} &
    DRIVER_PID=$!

    for _ in $(seq 1 120); do
        if [ -s "${ENDPOINT_FILE}" ]; then break; fi
        sleep 0.5
    done

    if [ ! -s "${ENDPOINT_FILE}" ]; then
        echo "Timed out waiting for SmartSim DB" >&2
        kill $DRIVER_PID 2>/dev/null || true
        RC=124
    else
        export SSDB="$(tr -d '\n' < "${ENDPOINT_FILE}")"
        mpirun ${SOLVER_MPI_ARGS} "${SOLVER_BIN}" \
            --provider SMARTSIM --model "${MODEL_PATH}" --schema "${SCHEMA}" --inputs "${INPUTS}" \
            ${EXTRA_SOLVER_ARGS} \
            > "${OUTPUT_FILE}" 2>&1
        RC=$?
        touch "${DONE_FILE}"
        wait "${DRIVER_PID}" || true
    fi
fi
set -e

JOB_MEM=$(get_job_cgroup_mem)

if [ ${RC} -eq 0 ]; then
    RES_LINE=$(grep '^RESULT:' "${OUTPUT_FILE}" | tail -n 1 | cut -d':' -f2)
    if [ -n "${RES_LINE}" ]; then
        T_COLD=$(echo "${RES_LINE}" | cut -d',' -f1)
        T_WARM=$(echo "${RES_LINE}" | cut -d',' -f2)
        M_RSS=$(echo "${RES_LINE}" | cut -d',' -f3)
        echo "${LABEL},${MODEL_NAME},${PNAME},${TPQ},${INTRA},${BIND},${T_COLD},${T_WARM},${M_RSS},${JOB_MEM},SUCCESS" >> "${RESULTS_CSV}"
        echo "  -> SUCCESS: cold=${T_COLD}s, warm=${T_WARM}s | solver_rss=${M_RSS}MB, job_mem=${JOB_MEM}MB"
    else
        echo "${LABEL},${MODEL_NAME},${PNAME},${TPQ},${INTRA},${BIND},-1,-1,-1,${JOB_MEM},FAILED_PARSE" >> "${RESULTS_CSV}"
        echo "  -> FAILED_PARSE"
    fi
else
    echo "${LABEL},${MODEL_NAME},${PNAME},${TPQ},${INTRA},${BIND},-1,-1,-1,${JOB_MEM},FAILED_RC_${RC}" >> "${RESULTS_CSV}"
    echo "  -> FAILED with RC ${RC}"
fi

NEXT_IDX=$(( IDX + 1 ))
echo "${NEXT_IDX}" > "${INDEX_FILE}"

if [ "${NEXT_IDX}" -le "${TOTAL_TASKS}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "Scheduling successor task ${NEXT_IDX}/${TOTAL_TASKS}..."
        ( cd "${BENCH_DIR}" && sbatch --dependency=afterany:${SLURM_JOB_ID} \
            --partition=devel \
            --account=default \
            --mem=238G \
            --time=01:00:00 \
            "${OUT_DIR}/scripts/run_devel_sweep.sh" ) || true
    fi
else
    echo "All ${TOTAL_TASKS} tasks completed successfully."
    rm -f "${INDEX_FILE}"
fi

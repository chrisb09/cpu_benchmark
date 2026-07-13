#!/bin/bash
set -euo pipefail

BASE_DIR="/hpcwork/ro092286/smartsim"
BENCH_DIR="/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench"
OUT_DIR="${BENCH_DIR}/results_c23mm"
LOGS_DIR="${OUT_DIR}/logs"
BUILD_DIR="${PROVIDER_BENCH_BUILD_DIR:-${BENCH_DIR}/build-provider-bench}"
SOLVER_BIN="${BUILD_DIR}/benchmark_solver"
DL_CLIENT_CPP="${PHYDLL_DL_CLIENT:-${BUILD_DIR}/dl-client/phydll_dl_client}"
DL_CLIENT_PY="${BASE_DIR}/CPP-ML-Interface/dl_clients/phydll_dl_client.py"
PYTHON_BIN="${BASE_DIR}/CPP-ML-Interface/extern/python/smartsim_cpu/bin/python"
PHYDLL_LIB_DIR="${BASE_DIR}/CPP-ML-Interface/extern/phydll/build/lib"
CUDA_STUB_SOURCE="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64/libcuda.so"
NVML_STUB_SOURCE="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64/libnvidia-ml.so"
CUDA_STUB_DIR="${BUILD_DIR}/cuda_stubs"
PNAME="${PNAME:?must set PNAME}"
CLIENT_KIND="${CLIENT_KIND:?must set CLIENT_KIND}"
RESULTS_CSV="${RESULTS_CSV:-${OUT_DIR}/${PNAME}.csv}"
NP_SOLVER="${NP_SOLVER:-96}"
NP_DL="${NP_DL:-${NP_SOLVER}}"
PHYDLL_TIMEOUT="${PHYDLL_TIMEOUT:-600}"
REQUIRED_SUCCESSFUL_RUNS="${PHYDLL_REQUIRED_SUCCESSFUL_RUNS:-10}"
MAX_ATTEMPTS="${PHYDLL_MAX_ATTEMPTS:-10}"

mkdir -p "${LOGS_DIR}" "${CUDA_STUB_DIR}"
ln -sf "${CUDA_STUB_SOURCE}" "${CUDA_STUB_DIR}/libcuda.so"
ln -sf "${CUDA_STUB_SOURCE}" "${CUDA_STUB_DIR}/libcuda.so.1"
ln -sf "${NVML_STUB_SOURCE}" "${CUDA_STUB_DIR}/libnvidia-ml.so"
ln -sf "${NVML_STUB_SOURCE}" "${CUDA_STUB_DIR}/libnvidia-ml.so.1"

cd "${BASE_DIR}/CPP-ML-Interface" && source ./set_env_claix23_cuda12.4.sh

export LD_LIBRARY_PATH="${PHYDLL_LIB_DIR}:${CUDA_STUB_DIR}:${LD_LIBRARY_PATH:-}"
export MLCOUPLING_LOG_LEVEL=DEBUG
export MLCOUPLING_INTRA_OP_THREADS=1
export MLCOUPLING_INTER_OP_THREADS=1
export PHYDLL_DL_COUNT=1
export PHYDLL_DL_FIELD_COUNT=1

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

if [ ! -x "${SOLVER_BIN}" ] || [ ! -x "${DL_CLIENT_CPP}" ]; then
    echo "Provider-bench binaries are missing." >&2
    echo "Run ${BENCH_DIR}/build.sh before submitting this benchmark." >&2
    exit 1
fi

if [ ! -f "${RESULTS_CSV}" ] || ! grep -q '^label,' "${RESULTS_CSV}"; then
    echo "label,model,provider,tpq,intra_threads,bind_cores,cold_time_s,warm_time_s,max_rss_mb,job_mem_mb,status" > "${RESULTS_CSV}"
fi

if grep -q ',RUNNING$' "${RESULTS_CSV}" 2>/dev/null; then
    sed -i 's/,RUNNING$/,TIMEOUT/' "${RESULTS_CSV}"
fi

MODELS=(
    "giant|100000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/giant_cpu.pt"
    "mmcp_transformer|10000|mmcp|${BASE_DIR}/MMCP_TOM/input/transformer_inference_scripted_fw2.pt"
    "transformer|1000000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/transformer_cpu.pt"
    "perfect|100000000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/perfect_cpu.pt"
    "watercnn|10000000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/watercnn_cpu.pt"
)

MODELS_FILTER="${PHYDLL_MODELS_FILTER:-}"

for m_data in "${MODELS[@]}"; do
    if [ -n "${MODELS_FILTER}" ]; then
        MODEL_NAME_PEEK=$(echo "$m_data" | cut -d'|' -f1)
        [[ ",${MODELS_FILTER}," == *",${MODEL_NAME_PEEK},"* ]] || continue
    fi
    MODEL_NAME=$(echo "$m_data" | cut -d'|' -f1)
    INPUTS=$(echo "$m_data" | cut -d'|' -f2)
    SCHEMA=$(echo "$m_data" | cut -d'|' -f3)
    MODEL_PATH=$(echo "$m_data" | cut -d'|' -f4)

    successful_runs=$(awk -F, -v m="$MODEL_NAME" -v p="$PNAME" 'NR>1 && $2==m && $3==p && $11=="SUCCESS" {c++} END {print c+0}' "${RESULTS_CSV}")
    attempted_runs=$(awk -F, -v m="$MODEL_NAME" -v p="$PNAME" 'NR>1 && $2==m && $3==p {c++} END {print c+0}' "${RESULTS_CSV}")
    if [ "${successful_runs}" -ge "${REQUIRED_SUCCESSFUL_RUNS}" ]; then
        echo "Skipping ${MODEL_NAME} for ${PNAME}: already have ${successful_runs} successful runs"
        continue
    fi
    if [ "${attempted_runs}" -ge "${MAX_ATTEMPTS}" ]; then
        echo "Skipping ${MODEL_NAME} for ${PNAME}: attempt limit reached (${attempted_runs}/${MAX_ATTEMPTS})" >&2
        INCOMPLETE=1
        continue
    fi

    for run in $(seq $((attempted_runs + 1)) "${MAX_ATTEMPTS}"); do
        if [ "${successful_runs}" -ge "${REQUIRED_SUCCESSFUL_RUNS}" ]; then
            break
        fi
        LABEL="${MODEL_NAME}_${PNAME}"
        OUTPUT_FILE="${LOGS_DIR}/${MODEL_NAME}_${PNAME}_run${run}.log"
        RUN_ROW="${LABEL},${MODEL_NAME},${PNAME},96,1,0,-1,-1,-1,-1,RUNNING"
        echo "${RUN_ROW}" >> "${RESULTS_CSV}"
        LINE_NUM=$(wc -l < "${RESULTS_CSV}")

        echo "=========================================================="
        echo "Model: ${MODEL_NAME} | Provider: ${PNAME} | Run: ${run}/10"

        set +e
        if [ "${CLIENT_KIND}" = "cpp" ]; then
            timeout "${PHYDLL_TIMEOUT}" mpirun --oversubscribe --bind-to none \
                -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
                -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
                -n ${NP_SOLVER} "${SOLVER_BIN}" \
                --provider PHYDLL --model "${MODEL_PATH}" --schema "${SCHEMA}" --inputs "${INPUTS}" \
                : -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
                -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
                -n ${NP_DL} "${DL_CLIENT_CPP}" \
                > "${OUTPUT_FILE}" 2>&1
        else
            timeout "${PHYDLL_TIMEOUT}" mpirun --oversubscribe --bind-to none \
                -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
                -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
                -n ${NP_SOLVER} "${SOLVER_BIN}" \
                --provider PHYDLL --model "${MODEL_PATH}" --schema "${SCHEMA}" --inputs "${INPUTS}" \
                : -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
                -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
                -n ${NP_DL} "${PYTHON_BIN}" "${DL_CLIENT_PY}" \
                > "${OUTPUT_FILE}" 2>&1
        fi
        RC=$?
        set -e

        RUN_SUCCEEDED=0
        if [ ${RC} -eq 0 ]; then
            RES_LINE=$(grep '^RESULT:' "${OUTPUT_FILE}" | tail -n 1 | cut -d':' -f2)
            if [ -n "${RES_LINE}" ]; then
                T_COLD=$(echo "${RES_LINE}" | cut -d',' -f1)
                T_WARM=$(echo "${RES_LINE}" | cut -d',' -f2)
                M_RSS=$(echo "${RES_LINE}" | cut -d',' -f3)
                M_JOB=$(get_job_cgroup_mem)
                FINAL_ROW="${LABEL},${MODEL_NAME},${PNAME},96,1,0,${T_COLD},${T_WARM},${M_RSS},${M_JOB},SUCCESS"
                echo "  -> SUCCESS: cold=${T_COLD}s, warm=${T_WARM}s | solver_rss=${M_RSS}MB, job_mem=${M_JOB}MB"
                RUN_SUCCEEDED=1
            else
                FINAL_ROW="${LABEL},${MODEL_NAME},${PNAME},96,1,0,-1,-1,-1,-1,FAILED_PARSE"
                echo "  -> FAILED: could not parse RESULT line"
            fi
        else
            FINAL_ROW="${LABEL},${MODEL_NAME},${PNAME},96,1,0,-1,-1,-1,-1,FAILED_RC_${RC}"
            echo "  -> FAILED with RC ${RC}"
        fi

        sed -i "${LINE_NUM}s|.*|${FINAL_ROW}|" "${RESULTS_CSV}"
        if [ "${RUN_SUCCEEDED}" -eq 1 ]; then
            successful_runs=$((successful_runs + 1))
        fi
    done

    if [ "${successful_runs}" -lt "${REQUIRED_SUCCESSFUL_RUNS}" ]; then
        echo "${MODEL_NAME} for ${PNAME}: ${successful_runs}/${REQUIRED_SUCCESSFUL_RUNS} successful runs after ${MAX_ATTEMPTS} attempts" >&2
        INCOMPLETE=1
    fi
done

if [ "${INCOMPLETE:-0}" -ne 0 ]; then
    echo "Benchmark incomplete: inspect ${RESULTS_CSV} and logs." >&2
    exit 1
fi

echo "Benchmark completed successfully."

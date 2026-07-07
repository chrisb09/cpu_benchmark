#!/bin/bash
set -euo pipefail

BASE_DIR="/hpcwork/ro092286/smartsim"
BENCH_DIR="/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench"
OUT_DIR="${BENCH_DIR}/results_c23mm"
LOGS_DIR="${OUT_DIR}/logs"
SOLVER_BIN="${BENCH_DIR}/build/benchmark_solver"
DL_CLIENT_CPP="${BASE_DIR}/CPP-ML-Interface/dl_clients/build/phydll_dl_client"
DL_CLIENT_PY="${BASE_DIR}/CPP-ML-Interface/dl_clients/phydll_dl_client.py"
PYTHON_BIN="${BASE_DIR}/CPP-ML-Interface/extern/python/smartsim_cpu/bin/python"
PHYDLL_LIB_DIR="${BASE_DIR}/CPP-ML-Interface/extern/phydll/build/lib"
CUDA_STUB_SOURCE="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64/libcuda.so"
NVML_STUB_SOURCE="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64/libnvidia-ml.so"
CUDA_STUB_DIR="${BENCH_DIR}/build/cuda_stubs"
PNAME="${PNAME:?must set PNAME}"
CLIENT_KIND="${CLIENT_KIND:?must set CLIENT_KIND}"
RESULTS_CSV="${RESULTS_CSV:-${OUT_DIR}/${PNAME}.csv}"
NP_SOLVER="${NP_SOLVER:-96}"
NP_DL="${NP_DL:-${NP_SOLVER}}"
PHYDLL_TIMEOUT="${PHYDLL_TIMEOUT:-600}"

mkdir -p "${LOGS_DIR}" "${CUDA_STUB_DIR}"
ln -sf "${CUDA_STUB_SOURCE}" "${CUDA_STUB_DIR}/libcuda.so"
ln -sf "${CUDA_STUB_SOURCE}" "${CUDA_STUB_DIR}/libcuda.so.1"
ln -sf "${NVML_STUB_SOURCE}" "${CUDA_STUB_DIR}/libnvidia-ml.so"
ln -sf "${NVML_STUB_SOURCE}" "${CUDA_STUB_DIR}/libnvidia-ml.so.1"

cd "${BASE_DIR}/CPP-ML-Interface" && source ./install.sh cpu

export LD_LIBRARY_PATH="${PHYDLL_LIB_DIR}:${CUDA_STUB_DIR}:${LD_LIBRARY_PATH:-}"
export MLCOUPLING_LOG_LEVEL=DEBUG
export MLCOUPLING_INTRA_OP_THREADS=1
export PHYDLL_DL_COUNT="${NP_DL}"

if [ ! -x "${SOLVER_BIN}" ]; then
    echo "benchmark_solver not found: ${SOLVER_BIN}" >&2
    exit 1
fi

if [ ! -f "${RESULTS_CSV}" ] || ! grep -q '^label,' "${RESULTS_CSV}"; then
    echo "label,model,provider,tpq,intra_threads,bind_cores,time_s,max_rss_mb,status" > "${RESULTS_CSV}"
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

for m_data in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "$m_data" | cut -d'|' -f1)
    INPUTS=$(echo "$m_data" | cut -d'|' -f2)
    SCHEMA=$(echo "$m_data" | cut -d'|' -f3)
    MODEL_PATH=$(echo "$m_data" | cut -d'|' -f4)

    existing_runs=$(awk -F, -v m="$MODEL_NAME" -v p="$PNAME" 'NR>1 && $2==m && $3==p {c++} END {print c+0}' "${RESULTS_CSV}")
    if [ "${existing_runs}" -ge 10 ]; then
        echo "Skipping ${MODEL_NAME} for ${PNAME}: already have ${existing_runs} rows"
        continue
    fi

    for run in $(seq $((existing_runs + 1)) 10); do
        LABEL="${MODEL_NAME}_${PNAME}"
        OUTPUT_FILE="${LOGS_DIR}/${MODEL_NAME}_${PNAME}_run${run}.log"
        RUN_ROW="${LABEL},${MODEL_NAME},${PNAME},96,1,0,-1,-1,RUNNING"
        echo "${RUN_ROW}" >> "${RESULTS_CSV}"
        LINE_NUM=$(wc -l < "${RESULTS_CSV}")

        echo "=========================================================="
        echo "Model: ${MODEL_NAME} | Provider: ${PNAME} | Run: ${run}/10"

        set +e
        if [ "${CLIENT_KIND}" = "cpp" ]; then
            timeout "${PHYDLL_TIMEOUT}" mpirun --oversubscribe --bind-to none \
                -x LD_LIBRARY_PATH -n ${NP_SOLVER} "${SOLVER_BIN}" \
                --provider PHYDLL --model "${MODEL_PATH}" --schema "${SCHEMA}" --inputs "${INPUTS}" \
                : -x LD_LIBRARY_PATH -n ${NP_DL} "${DL_CLIENT_CPP}" \
                > "${OUTPUT_FILE}" 2>&1
        else
            timeout "${PHYDLL_TIMEOUT}" mpirun --oversubscribe --bind-to none \
                -x LD_LIBRARY_PATH -n ${NP_SOLVER} "${SOLVER_BIN}" \
                --provider PHYDLL --model "${MODEL_PATH}" --schema "${SCHEMA}" --inputs "${INPUTS}" \
                : -x LD_LIBRARY_PATH -n ${NP_DL} "${PYTHON_BIN}" "${DL_CLIENT_PY}" \
                > "${OUTPUT_FILE}" 2>&1
        fi
        RC=$?
        set -e

        if [ ${RC} -eq 0 ]; then
            RES_LINE=$(grep '^RESULT:' "${OUTPUT_FILE}" | tail -n 1 | cut -d':' -f2)
            if [ -n "${RES_LINE}" ]; then
                T_S=$(echo "${RES_LINE}" | cut -d',' -f1)
                M_MB=$(echo "${RES_LINE}" | cut -d',' -f2)
                FINAL_ROW="${LABEL},${MODEL_NAME},${PNAME},96,1,0,${T_S},${M_MB},SUCCESS"
                echo "  -> SUCCESS: ${T_S}s | ${M_MB}MB"
            else
                FINAL_ROW="${LABEL},${MODEL_NAME},${PNAME},96,1,0,-1,-1,FAILED_PARSE"
                echo "  -> FAILED: could not parse RESULT line"
            fi
        else
            FINAL_ROW="${LABEL},${MODEL_NAME},${PNAME},96,1,0,-1,-1,FAILED_RC_${RC}"
            echo "  -> FAILED with RC ${RC}"
        fi

        sed -i "${LINE_NUM}s|.*|${FINAL_ROW}|" "${RESULTS_CSV}"
    done
done

echo "Benchmark completed successfully."

#!/bin/bash
#SBATCH --job-name=SS_TPQ96_I1_B96_MULTI_giant_c23ml
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --output=/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench/results_c23ml/logs/SS_TPQ96_I1_B96_MULTI_giant_%j.log
#SBATCH --time=05:00:00
#SBATCH --partition=c23ml
#SBATCH --exclusive
#SBATCH --account=thes2181

set -e
BENCH_DIR="/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench"
BASE_DIR="/hpcwork/ro092286/smartsim"
PYTHON_RUNTIME_ROOT="${BASE_DIR}/CPP-ML-Interface/extern/python"
RUNTIME_DEVICE="smartsim_cpu"
SMARTSIM_PYTHON="${PYTHON_RUNTIME_ROOT}/${RUNTIME_DEVICE}/bin/python"

cd "${BASE_DIR}/CPP-ML-Interface" && source ./install.sh cpu

RUNTIME_EXTRA_LIB_DIR="${PYTHON_RUNTIME_ROOT}/${RUNTIME_DEVICE}/runtime_libs"
if [ -d "${RUNTIME_EXTRA_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${RUNTIME_EXTRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
PHYDLL_LIB_DIR="${BASE_DIR}/CPP-ML-Interface/extern/phydll/build/lib"
export LD_LIBRARY_PATH="${PHYDLL_LIB_DIR}:${LD_LIBRARY_PATH:-}"

export MLCOUPLING_LOG_LEVEL=DEBUG
export SR_MODEL_TIMEOUT=2000000
export SR_CMD_TIMEOUT=2000000
export SR_SOCKET_TIMEOUT=2000000

export MLCOUPLING_MULTI_MODEL=1

cd $BENCH_DIR/build

RESULTS_CSV="/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench/results_c23ml/SS_TPQ96_I1_B96_MULTI_giant.csv"
echo "label,model,provider,tpq,intra_threads,bind_cores,time_s,max_rss_mb,status" > "$RESULTS_CSV"

PNAME="SS_TPQ96_I1_B96_MULTI_giant"
TPQ="96"
INTRA="1"
BIND="96"
ss_args="--threads-per-queue 96 --intra-op-threads 1 --cpu-cores-per-node 96"

ENDPOINT_FILE="/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench/results_c23ml/.ssdb_endpoint_${PNAME}"
DONE_FILE="/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench/results_c23ml/.solver_done_${PNAME}"

MODELS=(
    "giant|100000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/giant_cpu.pt"
)

for m_data in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "$m_data" | cut -d'|' -f1)
    INPUTS=$(echo "$m_data" | cut -d'|' -f2)
    SCHEMA=$(echo "$m_data" | cut -d'|' -f3)
    MODEL_PATH=$(echo "$m_data" | cut -d'|' -f4)
    LABEL="${MODEL_NAME}_${PNAME}_c23ml"

    echo "Starting database for model: ${MODEL_NAME}"
    rm -f "${ENDPOINT_FILE}" "${DONE_FILE}"

    ${SMARTSIM_PYTHON} "${BASE_DIR}/CPP-ML-Interface/dl_clients/smartsim_controller.py" \
        --auto-port \
        --endpoint-file "${ENDPOINT_FILE}" \
        --done-file "${DONE_FILE}" \
        --exp-dir "/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench/results_c23ml/smartsim_experiments_${PNAME}_${MODEL_NAME}" \
        --silent \
        ${ss_args} &
    DRIVER_PID=$!

    for _ in $(seq 1 120); do
        if [ -s "${ENDPOINT_FILE}" ]; then break; fi
        sleep 0.5
    done
    if [ ! -s "${ENDPOINT_FILE}" ]; then
        echo "Timed out waiting for SmartSim DB for model ${MODEL_NAME}"
        kill $DRIVER_PID 2>/dev/null || true
        continue
    fi
    export SSDB="$(tr -d '\n' < "${ENDPOINT_FILE}")"

    for run in {1..10}; do
        echo "=========================================================="
        echo "Model: $MODEL_NAME | Provider: $PNAME | Run: $run/10"

        OUTPUT_FILE="/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench/results_c23ml/logs/${MODEL_NAME}_${PNAME}_run${run}.log"

        set +e
        mpirun -n 96 ./benchmark_solver \
            --provider SMARTSIM --model "$MODEL_PATH" --schema "$SCHEMA" --inputs "$INPUTS" \
            > "$OUTPUT_FILE" 2>&1
        RC=$?
        set -e

        if [ $RC -eq 0 ]; then
            RES_LINE=$(grep "^RESULT:" "$OUTPUT_FILE" | tail -n 1 | cut -d':' -f2)
            if [ -n "$RES_LINE" ]; then
                T_S=$(echo "$RES_LINE" | cut -d',' -f1)
                M_MB=$(echo "$RES_LINE" | cut -d',' -f2)
                echo "$LABEL,$MODEL_NAME,$PNAME,$TPQ,$INTRA,$BIND,$T_S,$M_MB,SUCCESS" >> "$RESULTS_CSV"
                echo "Success: ${T_S}s | ${M_MB}MB"
            else
                echo "$LABEL,$MODEL_NAME,$PNAME,$TPQ,$INTRA,$BIND,-1,-1,FAILED_PARSE" >> "$RESULTS_CSV"
                echo "Failed to parse result line"
            fi
        else
            echo "$LABEL,$MODEL_NAME,$PNAME,$TPQ,$INTRA,$BIND,-1,-1,FAILED_RC_${RC}" >> "$RESULTS_CSV"
            echo "Failed with exit code ${RC}"
        fi
    done

    touch "${DONE_FILE}"
    wait "${DRIVER_PID}" || true
    echo "Tore down database for model: ${MODEL_NAME}"
done

echo "Benchmark completed."

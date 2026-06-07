#!/bin/bash
#SBATCH --job-name=provider_bench
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/provider_bench_%j.log


##SBATCH --partition=devel
##SBATCH --mem=220G

#SBATCH --partition=c23mm
#SBATCH --exclusive
#SBATCH --account=thes2181

#get core count
CORE_COUNT=$(nproc)

set -e

BENCH_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
BASE_DIR="$(cd "${BENCH_DIR}/../.." && pwd)"
PYTHON_RUNTIME_ROOT="${BASE_DIR}/CPP-ML-Interface/extern/python"

echo "Running provider benchmarks with the following configuration:"
echo "Base Directory: ${BASE_DIR}"
echo "Python Runtime Root: ${PYTHON_RUNTIME_ROOT}"
echo "Script Directory: ${BENCH_DIR}"

# Load environment using portable . (dot) instead of source
#. "${BASE_DIR}/CPP-ML-Interface/set_env_claix23_cuda12.4.sh"
cd "/hpcwork/ro092286/smartsim/CPP-ML-Interface" && source ./install.sh cpu

export SR_MODEL_TIMEOUT=900000
export SR_CMD_TIMEOUT=900000
export SR_SOCKET_TIMEOUT=900000

RUNTIME_DEVICE="smartsim_cpu"
SMARTSIM_PYTHON="${PYTHON_RUNTIME_ROOT}/${RUNTIME_DEVICE}/bin/python"
PY_ENV="${PYTHON_RUNTIME_ROOT}/${RUNTIME_DEVICE}"

# Staging runtime libs if they exist
RUNTIME_EXTRA_LIB_DIR="${PY_ENV}/runtime_libs"
if [ -d "${RUNTIME_EXTRA_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${RUNTIME_EXTRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
PHYDLL_LIB_DIR="$(cd "${BASE_DIR}/CPP-ML-Interface/extern/phydll/build/lib" && pwd)"
export LD_LIBRARY_PATH="${PHYDLL_LIB_DIR}:${LD_LIBRARY_PATH:-}"

mkdir -p "$BENCH_DIR/build" "$BENCH_DIR/logs"
cd "$BENCH_DIR/build"
cmake .. -DSMARTSIM_PYTHON="${SMARTSIM_PYTHON}"
make -j

RESULTS_CSV="$BENCH_DIR/provider_results.csv"
if [ ! -f "$RESULTS_CSV" ]; then
    echo "model,provider,time_s,max_rss_mb,status" > "$RESULTS_CSV"
fi

# commented out for now since it might cause more issues

MODELS="
giant|100000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/giant_cpu.pt
mmcp_transformer|10000|mmcp|${BASE_DIR}/MMCP_TOM/input/transformer_inference_scripted_fw2.pt
transformer|1000000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/transformer_cpu.pt
perfect|100000000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/perfect_cpu.pt
watercnn|10000000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/watercnn_cpu.pt
"


# just a quick smoke test to make sure the providers are working before running the full benchmarks with the large iteration counts
MODELS_test="
transformer|1000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/transformer_cpu.pt
perfect|10000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/perfect_cpu.pt
watercnn|10000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/watercnn_cpu.pt
giant|10000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/giant_cpu.pt
mmcp_transformer|10000|mmcp|${BASE_DIR}/MMCP_TOM/input/transformer_inference_scripted_fw2.pt
"


# Phydll might have some issue, we have to investigate later
#PROVIDERS="AIX SMARTSIM PHYDLL_CPP PHYDLL_PYTHON"
PROVIDERS="AIX SMARTSIM"
NP_SOLVER=96
NP_DL=96

for m_data in $MODELS; do
    MODEL_NAME=$(echo "$m_data" | cut -d'|' -f1)
    INPUTS=$(echo "$m_data" | cut -d'|' -f2)
    SCHEMA=$(echo "$m_data" | cut -d'|' -f3)
    MODEL_PATH=$(echo "$m_data" | cut -d'|' -f4)
    
    for PROVIDER in $PROVIDERS; do
        echo "=========================================================="
        echo "Running Model: $MODEL_NAME, Provider: $PROVIDER"
        
        OUTPUT_FILE="$BENCH_DIR/logs/${MODEL_NAME}_${PROVIDER}.log"
        
        set +e
        # 1. AIX
        if [ "$PROVIDER" = "AIX" ]; then
            mpirun -n ${NP_SOLVER} ./benchmark_solver --provider AIX --model "$MODEL_PATH" --schema "$SCHEMA" --inputs "$INPUTS" > "$OUTPUT_FILE" 2>&1
            RC=$?
            
        # 2. SMARTSIM
        elif [ "$PROVIDER" = "SMARTSIM" ]; then
            ENDPOINT_FILE="$BENCH_DIR/.ssdb_endpoint"
            DONE_FILE="$BENCH_DIR/.solver_done"
            rm -f "${ENDPOINT_FILE}" "${DONE_FILE}"
            
            # Start DB
            "${SMARTSIM_PYTHON}" "${BASE_DIR}/module_test/driver.py" --endpoint-file "${ENDPOINT_FILE}" --done-file "${DONE_FILE}" --port 6780 &
            DRIVER_PID=$!
            
            echo "Waiting for SmartSim database to start..."
            for _ in $(seq 1 120); do
                if [ -s "${ENDPOINT_FILE}" ]; then break; fi
                sleep 0.5
            done
            if [ ! -s "${ENDPOINT_FILE}" ]; then 
                echo "Timed out waiting for SmartSim DB"
                kill $DRIVER_PID
                continue
            fi
            export SSDB="$(tr -d '\n' < "${ENDPOINT_FILE}")"
            
            mpirun -n ${NP_SOLVER} ./benchmark_solver --provider SMARTSIM --model "$MODEL_PATH" --schema "$SCHEMA" --inputs "$INPUTS" > "$OUTPUT_FILE" 2>&1
            RC=$?
            
            touch "${DONE_FILE}"
            wait "${DRIVER_PID}" || true
            
        # 3. PHYDLL (C++)
        elif [ "$PROVIDER" = "PHYDLL_CPP" ]; then
            PHYDLL_DL_CLIENT="${BASE_DIR}/CPP-ML-Interface/dl_clients/build/phydll_dl_client"
            export PHYDLL_DL_COUNT=${NP_DL}
            mpirun --oversubscribe -n ${NP_SOLVER} ./benchmark_solver --provider PHYDLL --model "$MODEL_PATH" --schema "$SCHEMA" --inputs "$INPUTS" : -n ${NP_DL} "${PHYDLL_DL_CLIENT}" > "$OUTPUT_FILE" 2>&1
            RC=$?
            
        # 4. PHYDLL (Python)
        elif [ "$PROVIDER" = "PHYDLL_PYTHON" ]; then
            PHYDLL_DL_CLIENT_PY="${BASE_DIR}/CPP-ML-Interface/dl_clients/phydll_dl_client.py"
            export PHYDLL_DL_COUNT=${NP_DL}
            mpirun --oversubscribe -n ${NP_SOLVER} ./benchmark_solver --provider PHYDLL --model "$MODEL_PATH" --schema "$SCHEMA" --inputs "$INPUTS" : -n ${NP_DL} "${SMARTSIM_PYTHON}" "${PHYDLL_DL_CLIENT_PY}" > "$OUTPUT_FILE" 2>&1
            RC=$?
        fi
        set -e
        
        # Parse result
        if [ $RC -eq 0 ]; then
            RES_LINE=$(grep "^RESULT:" "$OUTPUT_FILE" | tail -n 1 | cut -d':' -f2)
            if [ -n "$RES_LINE" ]; then
                T_S=$(echo "$RES_LINE" | cut -d',' -f1)
                M_MB=$(echo "$RES_LINE" | cut -d',' -f2)
                echo "$MODEL_NAME,$PROVIDER,$T_S,$M_MB,SUCCESS" >> "$RESULTS_CSV"
                echo "Success: ${T_S}s | ${M_MB}MB"
            else
                echo "$MODEL_NAME,$PROVIDER,-1,-1,FAILED_PARSE" >> "$RESULTS_CSV"
                echo "Failed to parse output."
            fi
        else
            echo "$MODEL_NAME,$PROVIDER,-1,-1,FAILED_RC_${RC}" >> "$RESULTS_CSV"
            echo "Failed with RC $RC"
        fi
    done
done

echo "Provider benchmarks complete!"

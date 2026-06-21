#!/bin/bash
#SBATCH --job-name=provider_bench_single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --output=logs/provider_bench_single_%j.log

#SBATCH --time=01:00:00
#SBATCH --partition=devel
#SBATCH --mem=238G

##SBATCH --time=04:00:00
##SBATCH --partition=c23mm
##SBATCH --exclusive
##SBATCH --account=thes2181

set -e

BENCH_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
BASE_DIR="$(cd "${BENCH_DIR}/../.." && pwd)"
PYTHON_RUNTIME_ROOT="${BASE_DIR}/CPP-ML-Interface/extern/python"

echo "Running SINGLE-RANK provider benchmarks"
echo "Base Directory: ${BASE_DIR}"
echo "Script Directory: ${BENCH_DIR}"

cd "/hpcwork/ro092286/smartsim/CPP-ML-Interface" && source ./install.sh cpu

export SR_MODEL_TIMEOUT=2000000
export SR_CMD_TIMEOUT=2000000
export SR_SOCKET_TIMEOUT=2000000

RUNTIME_DEVICE="smartsim_cpu"
SMARTSIM_PYTHON="${PYTHON_RUNTIME_ROOT}/${RUNTIME_DEVICE}/bin/python"
PY_ENV="${PYTHON_RUNTIME_ROOT}/${RUNTIME_DEVICE}"

RUNTIME_EXTRA_LIB_DIR="${PY_ENV}/runtime_libs"
if [ -d "${RUNTIME_EXTRA_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${RUNTIME_EXTRA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
PHYDLL_LIB_DIR="$(cd "${BASE_DIR}/CPP-ML-Interface/extern/phydll/build/lib" && pwd)"
export LD_LIBRARY_PATH="${PHYDLL_LIB_DIR}:${LD_LIBRARY_PATH:-}"

mkdir -p "$BENCH_DIR/build" "$BENCH_DIR/logs"
cd "$BENCH_DIR/build"
cmake .. -DSMARTSIM_PYTHON="${SMARTSIM_PYTHON}" -DTorch_DIR="${BASE_DIR}/CPP-ML-Interface/extern/libtorch/share/cmake/Torch"
make -j

RESULTS_CSV="$BENCH_DIR/single_rank_results.csv"
if [ "${1:-}" = "--clean" ] || [ ! -f "$RESULTS_CSV" ]; then
    echo "model,provider,intra_threads,time_s,max_rss_mb,status" > "$RESULTS_CSV"
fi

if grep -q ",RUNNING$" "$RESULTS_CSV" 2>/dev/null; then
    sed -i 's/,RUNNING$/,TIMEOUT/' "$RESULTS_CSV"
fi

MODELS="
giant|100000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/giant_cpu.pt
mmcp_transformer|10000|mmcp|${BASE_DIR}/MMCP_TOM/input/transformer_inference_scripted_fw2.pt
transformer|1000000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/transformer_cpu.pt
perfect|100000000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/perfect_cpu.pt
watercnn|10000000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/watercnn_cpu.pt
"

# Test configurations:
# 1. SMARTSIM_NOBIND: no set_cpus at all (should auto-inherit 96 cpus-per-task)
# 2. SMARTSIM_BIND96: explicit set_cpus(96)
# 3. SMARTSIM_DEFAULT: SmartSim's own default settings
# 4. SMARTSIM_TPQ48_INTRA2_BIND96: best multi-rank config for comparison  
# 5. SMARTSIM_TPQ96_INTRA1_BIND96: max parallelism config
PROVIDERS="SMARTSIM_NOBIND SMARTSIM_BIND96 SMARTSIM_DEFAULT SMARTSIM_TPQ48_INTRA2_BIND96 SMARTSIM_TPQ96_INTRA1_BIND96"

NP_SOLVER=1
export MLCOUPLING_LOG_LEVEL=DEBUG

COMBO_FILE="$BENCH_DIR/single_combinations.txt"
if [ "${1:-}" = "--clean" ] || [ ! -f "$COMBO_FILE" ]; then
    rm -f "$COMBO_FILE"
    for m_data in $MODELS; do
        MODEL_NAME=$(echo "$m_data" | cut -d'|' -f1)
        INPUTS=$(echo "$m_data" | cut -d'|' -f2)
        SCHEMA=$(echo "$m_data" | cut -d'|' -f3)
        MODEL_PATH=$(echo "$m_data" | cut -d'|' -f4)
        for PROVIDER in $PROVIDERS; do
            echo "$MODEL_NAME|$INPUTS|$SCHEMA|$MODEL_PATH|$PROVIDER" >> "$COMBO_FILE"
        done
    done
fi

STARTED_FILE="$BENCH_DIR/single_started.txt"
if [ "${1:-}" = "--clean" ]; then
    rm -f "$STARTED_FILE"
fi
if [ ! -f "$STARTED_FILE" ]; then echo "0" > "$STARTED_FILE"; fi
TOTAL_TASKS=$(wc -l < "$COMBO_FILE")
STARTED=$(cat "$STARTED_FILE")

if [ "$STARTED" -ge "$TOTAL_TASKS" ]; then
    echo "All single-rank tasks completed ($STARTED/$TOTAL_TASKS)."
    exit 0
fi

# Queue successor
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "Work remaining ($STARTED/$TOTAL_TASKS). Scheduling successor..."
    CUR_PARTITION="${SLURM_JOB_PARTITION:-devel}"
    CUR_ACCOUNT="${SLURM_JOB_ACCOUNT:-default}"
    MEM_ARGS=()
    [ -n "$SLURM_MEM_PER_NODE" ] && MEM_ARGS=(--mem="$SLURM_MEM_PER_NODE")
    [ -n "$SLURM_MEM_PER_CPU" ] && MEM_ARGS=(--mem-per-cpu="$SLURM_MEM_PER_CPU")
    ( cd "$BENCH_DIR" && sbatch --dependency=afterany:$SLURM_JOB_ID \
        --partition="$CUR_PARTITION" \
        --account="$CUR_ACCOUNT" \
        "${MEM_ARGS[@]}" \
        --time="01:00:00" \
        --ntasks=1 \
        --cpus-per-task=96 \
        run_bench_single.sh ) || true
fi

while [ "$STARTED" -lt "$TOTAL_TASKS" ]; do
    if [ "$SECONDS" -ge 1800 ]; then
        echo "Time limit reached (30m). Handoff to successor."
        exit 0
    fi
    
    LINE_NUM=$(( STARTED + 1 ))
    COMBO=$(sed -n "${LINE_NUM}p" "$COMBO_FILE")
    IFS='|' read -r MODEL_NAME INPUTS SCHEMA MODEL_PATH PROVIDER <<< "$COMBO"
    
    # Determine intra_val for CSV
    if [[ "$PROVIDER" =~ ^SMARTSIM_TPQ([0-9]+)_INTRA([0-9]+)_(BIND96|NOBIND)$ ]]; then
        INTRA_VAL="${BASH_REMATCH[2]}_${BASH_REMATCH[3]}"
    elif [ "$PROVIDER" = "SMARTSIM_NOBIND" ]; then
        INTRA_VAL="96_NOBIND"
    elif [ "$PROVIDER" = "SMARTSIM_BIND96" ]; then
        INTRA_VAL="96_BIND96"
    elif [ "$PROVIDER" = "SMARTSIM_DEFAULT" ]; then
        INTRA_VAL="DEFAULT"
    else
        INTRA_VAL="N/A"
    fi
    
    echo "$LINE_NUM" > "$STARTED_FILE"
    STARTED=$LINE_NUM
    
    echo "$MODEL_NAME,$PROVIDER,$INTRA_VAL,-1,-1,RUNNING" >> "$RESULTS_CSV"
    
    echo "=========================================================="
    echo "[SINGLE RANK] Model: $MODEL_NAME, Provider: $PROVIDER"
    
    OUTPUT_FILE="$BENCH_DIR/logs/single_${MODEL_NAME}_${PROVIDER}.log"
    
    set +e
    if [[ "$PROVIDER" == SMARTSIM* ]]; then
        ENDPOINT_FILE="$BENCH_DIR/.ssdb_endpoint_single"
        DONE_FILE="$BENCH_DIR/.solver_done_single"
        rm -f "${ENDPOINT_FILE}" "${DONE_FILE}"
        
        SMARTSIM_ARGS=""
        if [ "$PROVIDER" = "SMARTSIM_BIND96" ]; then
            SMARTSIM_ARGS="--cpu-cores-per-node 96"
        elif [ "$PROVIDER" = "SMARTSIM_NOBIND" ]; then
            SMARTSIM_ARGS="--no-cpu-bind"
        elif [ "$PROVIDER" = "SMARTSIM_DEFAULT" ]; then
            SMARTSIM_ARGS="--use-default-cpu-settings"
        elif [[ "$PROVIDER" =~ ^SMARTSIM_TPQ([0-9]+)_INTRA([0-9]+)_(BIND96|NOBIND)$ ]]; then
            TPQ="${BASH_REMATCH[1]}"
            INTRA="${BASH_REMATCH[2]}"
            BIND_OPT="${BASH_REMATCH[3]}"
            SMARTSIM_ARGS="--threads-per-queue $TPQ --intra-op-threads $INTRA"
            if [ "$BIND_OPT" = "NOBIND" ]; then
                SMARTSIM_ARGS="$SMARTSIM_ARGS --no-cpu-bind"
            elif [ "$BIND_OPT" = "BIND96" ]; then
                SMARTSIM_ARGS="$SMARTSIM_ARGS --cpu-cores-per-node 96"
            fi
        fi
        
        # Start DB
        ${SMARTSIM_PYTHON} "${BASE_DIR}/CPP-ML-Interface/dl_clients/smartsim_controller.py" \
            --auto-port \
            --endpoint-file "${ENDPOINT_FILE}" \
            --done-file "${DONE_FILE}" \
            --exp-dir "$BENCH_DIR/build/smartsim_experiments" \
            --silent \
            ${SMARTSIM_ARGS} &
        DRIVER_PID=$!
        
        for _ in $(seq 1 120); do
            if [ -s "${ENDPOINT_FILE}" ]; then break; fi
            sleep 0.5
        done
        if [ ! -s "${ENDPOINT_FILE}" ]; then 
            echo "Timed out waiting for SmartSim DB"
            kill $DRIVER_PID
            RC=124
            touch "${DONE_FILE}"
        else
            export SSDB="$(tr -d '\n' < "${ENDPOINT_FILE}")"
            
            mpirun -n ${NP_SOLVER} ./benchmark_solver --provider SMARTSIM --model "$MODEL_PATH" --schema "$SCHEMA" --inputs "$INPUTS" > "$OUTPUT_FILE" 2>&1
            RC=$?
            
            touch "${DONE_FILE}"
            wait "${DRIVER_PID}" || true
        fi
    fi
    set -e
    
    if [ $RC -eq 0 ]; then
        RES_LINE=$(grep "^RESULT:" "$OUTPUT_FILE" | tail -n 1 | cut -d':' -f2)
        if [ -n "$RES_LINE" ]; then
            T_S=$(echo "$RES_LINE" | cut -d',' -f1)
            M_MB=$(echo "$RES_LINE" | cut -d',' -f2)
            sed -i "s/^$MODEL_NAME,$PROVIDER,$INTRA_VAL,-1,-1,RUNNING$/$MODEL_NAME,$PROVIDER,$INTRA_VAL,$T_S,$M_MB,SUCCESS/" "$RESULTS_CSV"
            echo "Success: ${T_S}s | ${M_MB}MB"
        else
            sed -i "s/^$MODEL_NAME,$PROVIDER,$INTRA_VAL,-1,-1,RUNNING$/$MODEL_NAME,$PROVIDER,$INTRA_VAL,-1,-1,FAILED_PARSE/" "$RESULTS_CSV"
            echo "Failed to parse output."
        fi
    else
        sed -i "s/^$MODEL_NAME,$PROVIDER,$INTRA_VAL,-1,-1,RUNNING$/$MODEL_NAME,$PROVIDER,$INTRA_VAL,-1,-1,FAILED_RC_${RC}/" "$RESULTS_CSV"
        echo "Failed with RC $RC"
    fi
done

echo "Single-rank provider benchmarks complete!"

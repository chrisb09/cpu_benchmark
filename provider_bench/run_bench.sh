#!/bin/bash
#SBATCH --job-name=provider_bench
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/provider_bench_%j.log

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

echo "Running provider benchmarks"
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

if [ "${1:-}" = "--clean" ]; then
    echo "Cleaning previous benchmark state..."
    rm -f "$BENCH_DIR/started.txt" "$BENCH_DIR/combinations.txt" "$BENCH_DIR/provider_results.csv"
fi

RESULTS_CSV="$BENCH_DIR/provider_results.csv"
if [ ! -f "$RESULTS_CSV" ] || ! grep -q "label" "$RESULTS_CSV"; then
    echo "label,model,provider,tpq,intra_threads,bind_cores,time_s,max_rss_mb,status" > "$RESULTS_CSV"
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

# Provider definitions: NAME|TPQ|INTRA|BIND_CORES|SMARTSIM_ARGS
# BIND_CORES=0 means no explicit binding (--no-cpu-bind)
# BIND_CORES=-1 means use SmartSim default (--use-default-cpu-settings)
# BIND_CORES=N means bind N cores (--cpu-cores-per-node N)
#
# Rule set:
#   AIX                          - baseline AIx CPU inference
#   SS_DEFAULT                   - SmartSim with its own defaults (no explicit thread/bind settings)
#   SS_B96                       - SmartSim default threads but explicitly bound to all 96 cores
#   SS_NOBIND                    - SmartSim default threads, no binding
#   SS_TPQ=N_I=M_B=96            - Explicit TPQ+intra, always bind 96 cores (best practice)
#   SS_TPQ=96_I=1_NOBIND         - intra=1 case: binding 1 core makes no sense, use nobind
PROVIDERS="
AIX|N/A|N/A|N/A|N/A
SS_DEFAULT|N/A|N/A|-1|--use-default-cpu-settings
SS_B96|N/A|N/A|96|--cpu-cores-per-node 96
SS_NOBIND|N/A|N/A|0|--no-cpu-bind
SS_TPQ=2_I=48_B=96|2|48|96|--threads-per-queue 2 --intra-op-threads 48 --cpu-cores-per-node 96
SS_TPQ=4_I=24_B=96|4|24|96|--threads-per-queue 4 --intra-op-threads 24 --cpu-cores-per-node 96
SS_TPQ=12_I=8_B=96|12|8|96|--threads-per-queue 12 --intra-op-threads 8 --cpu-cores-per-node 96
SS_TPQ=24_I=4_B=96|24|4|96|--threads-per-queue 24 --intra-op-threads 4 --cpu-cores-per-node 96
SS_TPQ=48_I=2_B=96|48|2|96|--threads-per-queue 48 --intra-op-threads 2 --cpu-cores-per-node 96
SS_TPQ=96_I=1_NOBIND|96|1|0|--threads-per-queue 96 --intra-op-threads 1 --no-cpu-bind
SS_TPQ=96_I=1_B=96|96|1|96|--threads-per-queue 96 --intra-op-threads 1 --cpu-cores-per-node 96
"

NP_SOLVER=96
export MLCOUPLING_LOG_LEVEL=DEBUG

# Build combinations.txt
if [ ! -f "$BENCH_DIR/combinations.txt" ]; then
    while IFS='|' read -r PNAME TPQ INTRA BIND SS_ARGS; do
        [ -z "$PNAME" ] && continue
        while IFS='|' read -r MODEL_NAME INPUTS SCHEMA MODEL_PATH; do
            [ -z "$MODEL_NAME" ] && continue
            echo "${MODEL_NAME}|${INPUTS}|${SCHEMA}|${MODEL_PATH}|${PNAME}|${TPQ}|${INTRA}|${BIND}|${SS_ARGS}" >> "$BENCH_DIR/combinations.txt"
        done <<< "$(echo "$MODELS" | grep -v '^$')"
    done <<< "$(echo "$PROVIDERS" | grep -v '^$')"
fi

if [ ! -f "$BENCH_DIR/started.txt" ]; then echo "0" > "$BENCH_DIR/started.txt"; fi
TOTAL_TASKS=$(wc -l < "$BENCH_DIR/combinations.txt")
STARTED=$(cat "$BENCH_DIR/started.txt")

if [ "$STARTED" -ge "$TOTAL_TASKS" ]; then
    echo "All tasks already completed ($STARTED/$TOTAL_TASKS)."
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
        --ntasks=96 \
        --cpus-per-task=1 \
        run_bench.sh ) || true
fi

while [ "$STARTED" -lt "$TOTAL_TASKS" ]; do
    if [ "$SECONDS" -ge 1800 ]; then
        echo "Time limit reached (30m). Handoff to successor."
        exit 0
    fi

    LINE_NUM=$(( STARTED + 1 ))
    COMBO=$(sed -n "${LINE_NUM}p" "$BENCH_DIR/combinations.txt")
    IFS='|' read -r MODEL_NAME INPUTS SCHEMA MODEL_PATH PNAME TPQ INTRA BIND SS_ARGS <<< "$COMBO"

    # Build the label: model_providerName
    LABEL="${MODEL_NAME}_${PNAME}"

    echo "$LINE_NUM" > "$BENCH_DIR/started.txt"
    STARTED=$LINE_NUM

    echo "$LABEL,$MODEL_NAME,$PNAME,$TPQ,$INTRA,$BIND,-1,-1,RUNNING" >> "$RESULTS_CSV"

    echo "=========================================================="
    echo "[$LINE_NUM/$TOTAL_TASKS] Model: $MODEL_NAME | Provider: $PNAME"

    OUTPUT_FILE="$BENCH_DIR/logs/${MODEL_NAME}_${PNAME}.log"

    set +e
    RC=0

    if [ "$PNAME" = "AIX" ]; then
        mpirun -n ${NP_SOLVER} ./benchmark_solver \
            --provider AIX --model "$MODEL_PATH" --schema "$SCHEMA" --inputs "$INPUTS" \
            > "$OUTPUT_FILE" 2>&1
        RC=$?

    elif [[ "$PNAME" == SS_* ]]; then
        ENDPOINT_FILE="$BENCH_DIR/.ssdb_endpoint"
        DONE_FILE="$BENCH_DIR/.solver_done"
        rm -f "${ENDPOINT_FILE}" "${DONE_FILE}"

        # Start SmartSim DB with the provider-specific args
        ${SMARTSIM_PYTHON} "${BASE_DIR}/CPP-ML-Interface/dl_clients/smartsim_controller.py" \
            --auto-port \
            --endpoint-file "${ENDPOINT_FILE}" \
            --done-file "${DONE_FILE}" \
            --exp-dir "$BENCH_DIR/build/smartsim_experiments" \
            --silent \
            ${SS_ARGS} &
        DRIVER_PID=$!

        for _ in $(seq 1 120); do
            if [ -s "${ENDPOINT_FILE}" ]; then break; fi
            sleep 0.5
        done

        if [ ! -s "${ENDPOINT_FILE}" ]; then
            echo "Timed out waiting for SmartSim DB"
            kill $DRIVER_PID 2>/dev/null || true
            RC=124
            touch "${DONE_FILE}"
        else
            export SSDB="$(tr -d '\n' < "${ENDPOINT_FILE}")"

            mpirun -n ${NP_SOLVER} ./benchmark_solver \
                --provider SMARTSIM --model "$MODEL_PATH" --schema "$SCHEMA" --inputs "$INPUTS" \
                > "$OUTPUT_FILE" 2>&1
            RC=$?

            touch "${DONE_FILE}"
            wait "${DRIVER_PID}" || true
        fi

    elif [ "$PNAME" = "PHYDLL_CPP" ]; then
        PHYDLL_DL_CLIENT="${BASE_DIR}/CPP-ML-Interface/dl_clients/build/phydll_dl_client"
        export PHYDLL_DL_COUNT=${NP_SOLVER}
        mpirun --oversubscribe -n ${NP_SOLVER} ./benchmark_solver \
            --provider PHYDLL --model "$MODEL_PATH" --schema "$SCHEMA" --inputs "$INPUTS" \
            : -n ${NP_SOLVER} "${PHYDLL_DL_CLIENT}" \
            > "$OUTPUT_FILE" 2>&1
        RC=$?
    fi
    set -e

    if [ $RC -eq 0 ]; then
        RES_LINE=$(grep "^RESULT:" "$OUTPUT_FILE" | tail -n 1 | cut -d':' -f2)
        if [ -n "$RES_LINE" ]; then
            T_S=$(echo "$RES_LINE" | cut -d',' -f1)
            M_MB=$(echo "$RES_LINE" | cut -d',' -f2)
            sed -i "s|^${LABEL},${MODEL_NAME},${PNAME},${TPQ},${INTRA},${BIND},-1,-1,RUNNING$|${LABEL},${MODEL_NAME},${PNAME},${TPQ},${INTRA},${BIND},${T_S},${M_MB},SUCCESS|" "$RESULTS_CSV"
            echo "  -> SUCCESS: ${T_S}s | ${M_MB}MB"
        else
            sed -i "s|^${LABEL},${MODEL_NAME},${PNAME},${TPQ},${INTRA},${BIND},-1,-1,RUNNING$|${LABEL},${MODEL_NAME},${PNAME},${TPQ},${INTRA},${BIND},-1,-1,FAILED_PARSE|" "$RESULTS_CSV"
            echo "  -> FAILED: could not parse RESULT line"
        fi
    else
        sed -i "s|^${LABEL},${MODEL_NAME},${PNAME},${TPQ},${INTRA},${BIND},-1,-1,RUNNING$|${LABEL},${MODEL_NAME},${PNAME},${TPQ},${INTRA},${BIND},-1,-1,FAILED_RC_${RC}|" "$RESULTS_CSV"
        echo "  -> FAILED with RC $RC"
    fi
done

echo "Provider benchmarks complete!"

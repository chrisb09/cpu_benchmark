#!/bin/bash
#SBATCH --job-name=cpu_scaling_wide
#SBATCH --partition=devel
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=220G
#SBATCH --output=logs/scaling_bench_%j.log

# This script tests CPU scaling for ranks vs intra-op threads across a wide range of n.
# It includes rescheduling logic and Discord logging for consistency.

BENCH_ROOT="$(pwd)"
BENCH_DIR="${BENCH_ROOT}/benchmarks/scaling_test_combined"
mkdir -p "$BENCH_DIR/logs" "$BENCH_DIR/runs"
RESULTS_CSV="$BENCH_DIR/results.csv"
COMBINATIONS_FILE="$BENCH_DIR/combinations.txt"
STARTED_FILE="$BENCH_DIR/started.txt"
CONCLUDED_FILE="$BENCH_DIR/concluded.txt"
MSG_ID_FILE="$BENCH_DIR/msg_id.txt"

# Discord Script
DISCORD_SCRIPT="${HOME}/scripts/discord_msg.sh"

send_discord() {
    local content="$1"
    if [ -x "$DISCORD_SCRIPT" ]; then
        if [ -f "$MSG_ID_FILE" ]; then
            local msg_id=$(cat "$MSG_ID_FILE")
            "$DISCORD_SCRIPT" edit "$msg_id" "$content" > /dev/null 2>&1
        else
            local msg_id=$("$DISCORD_SCRIPT" send "$content" | grep -E '^[0-9]+$')
            [ -n "$msg_id" ] && echo "$msg_id" > "$MSG_ID_FILE"
        fi
    fi
}

# Generate combinations if they don't exist
if [ ! -f "$COMBINATIONS_FILE" ]; then
    MODELS=(
        "giant|100000|mini_app|../mini_app/train_models/model_a/giant_cpu.pt"
        "transformer|1000000|mini_app|../mini_app/train_models/model_a/transformer_cpu.pt"
        "mmcp_transformer|10000|mmcp|../MMCP_TOM/input/transformer_inference_scripted_fw2.pt"
        "perfect|100000000|mini_app|../mini_app/train_models/model_a/perfect_cpu.pt"
        "watercnn|10000000|mini_app|../mini_app/train_models/model_a/watercnn_cpu.pt"
    )
    N_VALUES=(1 2 4 8 12 16 24 36 48 72 96)
    
    for m_data in "${MODELS[@]}"; do
        for n in "${N_VALUES[@]}"; do
            echo "${m_data}|rank|$n" >> "$COMBINATIONS_FILE"
            echo "${m_data}|intra|$n" >> "$COMBINATIONS_FILE"
        done
    done
fi

# Initialize state
if [ ! -f "$STARTED_FILE" ]; then echo "0" > "$STARTED_FILE"; fi
if [ ! -f "$CONCLUDED_FILE" ]; then echo "0" > "$CONCLUDED_FILE"; fi
if [ ! -f "$RESULTS_CSV" ]; then echo "model,scaling_type,n,time_s,max_rss_mb,status" > "$RESULTS_CSV"; fi

# Load environment
current_dir=$(pwd)
cd /hpcwork/ro092286/smartsim/CPP-ML-Interface
source ./install.sh cuda-12
cd "$current_dir"

TOTAL_TASKS=$(wc -l < "$COMBINATIONS_FILE")
CONCLUDED=$(cat "$CONCLUDED_FILE")

# Check if work is left
if [ "$CONCLUDED" -ge "$TOTAL_TASKS" ]; then
    echo "All tasks already completed ($CONCLUDED/$TOTAL_TASKS)."
    exit 0
fi

# Queue successor immediately
echo "Work remaining ($CONCLUDED/$TOTAL_TASKS). Scheduling successor..."
sbatch --dependency=afterany:$SLURM_JOB_ID \
    --partition="${SLURM_JOB_PARTITION:-devel}" \
    --time="01:00:00" \
    --mem=220G \
    --ntasks=96 \
    $0

PREV_RESULT="N/A"
CUR_TASK=$CONCLUDED

while [ "$CUR_TASK" -lt "$TOTAL_TASKS" ]; do
    # Time limit check (30 minutes)
    if [ "$SECONDS" -ge 1800 ]; then
        send_discord "⏳ **Scaling Job Handoff** (Progress: $CUR_TASK / $TOTAL_TASKS)"
        echo "Time limit reached (30m). Handoff to successor."
        exit 0
    fi
    
    LINE_NUM=$(( CUR_TASK + 1 ))
    COMBO=$(sed -n "${LINE_NUM}p" "$COMBINATIONS_FILE")
    IFS='|' read -r NAME INPUTS SCHEMA M_PATH S_TYPE N <<< "$COMBO"
    
    # Update started count for monitoring
    echo "$LINE_NUM" > "$STARTED_FILE"
    
    MSG="📊 **Extended Scaling Progress** ($LINE_NUM / $TOTAL_TASKS)
**Previous Run:** $PREV_RESULT
**Current Run:**
• Model: \`$NAME\`
• Scaling: $S_TYPE (n=$N)
Status: 🏃 Running..."
    send_discord "$MSG"

    ranks=1
    intra=1
    cores_per_rank=1
    if [ "$S_TYPE" == "rank" ]; then
        ranks=$N
        intra=1
        cores_per_rank=1
    else
        ranks=1
        intra=$N
        cores_per_rank=$N
    fi

    export OMP_NUM_THREADS=$intra
    OUTPUT_FILE="$BENCH_DIR/runs/${NAME}_${S_TYPE}_${N}.txt"
    
    # Use overcommit for high thread/rank counts on limited physical cores
    srun --exact --overcommit -n $ranks -c $cores_per_rank \
        python3 -u benchmark.py \
            --model "$M_PATH" \
            --num-inputs "$INPUTS" \
            --schema "$SCHEMA" \
            --intra $intra --inter 1 2>&1 | tee "$OUTPUT_FILE"
    
    RC=${PIPESTATUS[0]}
    
    if [ $RC -eq 0 ]; then
        RES_LINE=$(grep "^RESULT:" "$OUTPUT_FILE" | tail -n 1 | cut -d':' -f2)
        if [ -n "$RES_LINE" ]; then
            IFS=',' read -r T_S M_MB <<< "$RES_LINE"
            echo "$NAME,$S_TYPE,$N,$T_S,$M_MB,SUCCESS" >> "$RESULTS_CSV"
            PREV_RESULT="✅ ${T_S}s | ${M_MB}MB"
        else
            echo "$NAME,$S_TYPE,$N,-1,-1,FAILED_PARSE" >> "$RESULTS_CSV"
            PREV_RESULT="❌ Parse Failure"
        fi
    else
        echo "$NAME,$S_TYPE,$N,-1,-1,FAILED_RC_${RC}" >> "$RESULTS_CSV"
        PREV_RESULT="❌ Error (RC $RC)"
    fi
    
    # Task successfully finished or recorded as failed, increment concluded
    echo "$LINE_NUM" > "$CONCLUDED_FILE"
    CUR_TASK=$LINE_NUM
done

send_discord "🏁 **Extended Scaling Complete!**
• Total Tasks: $TOTAL_TASKS
• Results: \`$RESULTS_CSV\`"

echo "Benchmark complete!"

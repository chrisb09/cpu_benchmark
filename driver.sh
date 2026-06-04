#!/bin/bash
#SBATCH --job-name=cpu_ml_bench
#SBATCH --partition=devel
#SBATCH --account=default
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --gres=none
#SBATCH --output=logs/benchmark_%j.log

# Benchmark parameters — set via env vars before calling sbatch:
#   BENCH_MODEL      model name prefix, e.g. "giant" or "perfect"  (default: giant)
#   BENCH_INPUTS     total inference inputs across all ranks        (default: 100000)
#   BENCH_SCHEMA     "mini_app" (default) or "mmcp"
#   BENCH_SEQ_LEN    sequence length for mmcp (default: 5)
#   BENCH_FEAT_DIM   feature dimension for mmcp (default: 512)
export BENCH_MODEL="${BENCH_MODEL:-giant}"
export BENCH_INPUTS="${BENCH_INPUTS:-100000}"
export BENCH_SCHEMA="${BENCH_SCHEMA:-mini_app}"
export BENCH_SEQ_LEN="${BENCH_SEQ_LEN:-5}"
export BENCH_FEAT_DIM="${BENCH_FEAT_DIM:-512}"

# Use SLURM_SUBMIT_DIR to anchor the output to the workspace.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    BENCH_ROOT="${SLURM_SUBMIT_DIR}"
else
    BENCH_ROOT="$(cd "$(dirname "$0")" && pwd)"
fi
cd "$BENCH_ROOT" || exit 1

BENCH_DIR="${BENCH_ROOT}/benchmarks/${BENCH_MODEL}_${BENCH_SCHEMA}_${BENCH_INPUTS}"
mkdir -p "$BENCH_DIR/logs" "$BENCH_DIR/runs"
echo "Benchmark directory: $BENCH_DIR  (model=$BENCH_MODEL, schema=$BENCH_SCHEMA, inputs=$BENCH_INPUTS)"

# Optional: reset previous run state/results
if [ "${1:-}" = "--clean" ]; then
    echo "Cleaning previous benchmark state/results..."
    rm -f "$BENCH_DIR/started.txt" "$BENCH_DIR/concluded.txt" \
          "$BENCH_DIR/results.csv" "$BENCH_DIR/combinations.txt" "$BENCH_DIR/msg_id.txt" \
          "$BENCH_DIR/total_start_timestamp.txt"
    rm -rf "$BENCH_DIR/runs"
    mkdir -p "$BENCH_DIR/runs"
    date +%s > "$BENCH_DIR/total_start_timestamp.txt"
fi

# Load cluster environment
current_dir=$(pwd)
cd /hpcwork/ro092286/smartsim/CPP-ML-Interface
source ./install.sh cuda-12
cd "$current_dir"

DISCORD_SCRIPT="${HOME}/scripts/discord_msg.sh"

# Resolve model path
if [[ "${BENCH_SCHEMA}" == "mmcp" ]]; then
    MODEL_CPU="../MMCP_TOM/input/transformer_inference_scripted_fw2.pt"
else
    MODEL_CUDA="../mini_app/train_models/model_a/${BENCH_MODEL}_cuda.pt"
    MODEL_CPU="../mini_app/train_models/model_a/${BENCH_MODEL}_cpu.pt"
    if [ ! -f "$MODEL_CPU" ] && [ -f "$MODEL_CUDA" ]; then
        echo "Converting ${BENCH_MODEL}_cuda.pt to ${BENCH_MODEL}_cpu.pt..."
        python3 -c "import torch; model = torch.jit.load('${MODEL_CUDA}', map_location='cpu'); model.eval(); model.save('${MODEL_CPU}')"
    fi
fi

if [ ! -f "$MODEL_CPU" ]; then
    echo "Error: Model not found at $MODEL_CPU"
    exit 1
fi

# --- Memory Analysis Parameters ---
MODEL_FILE_SIZE_B=$(stat -c%s "$MODEL_CPU" 2>/dev/null || echo 0)
MODEL_FILE_SIZE_MB=$(echo "scale=2; $MODEL_FILE_SIZE_B / 1048576" | bc)

if [[ "${BENCH_SCHEMA}" == "mmcp" ]]; then
    TOTAL_INPUT_DATA_B=$(echo "$BENCH_INPUTS * $BENCH_SEQ_LEN * $BENCH_FEAT_DIM * 4" | bc)
else
    TOTAL_INPUT_DATA_B=$(echo "$BENCH_INPUTS * 18 * 4" | bc)
fi
TOTAL_INPUT_DATA_MB=$(echo "scale=2; $TOTAL_INPUT_DATA_B / 1048576" | bc)

send_discord() {
    local content="$1"
    if [ "${#content}" -gt 1990 ]; then content="${content:0:1980}... (truncated, see summary.txt)"; fi
    if [ -x "$DISCORD_SCRIPT" ]; then
        if [ -f "$BENCH_DIR/msg_id.txt" ]; then
            local msg_id=$(cat "$BENCH_DIR/msg_id.txt")
            "$DISCORD_SCRIPT" edit "$msg_id" "$content" > /dev/null 2>&1
        else
            local msg_id=$("$DISCORD_SCRIPT" send "$content" | grep -E '^[0-9]+$')
            [ -n "$msg_id" ] && echo "$msg_id" > "$BENCH_DIR/msg_id.txt"
        fi
    fi
}

# Generate combinations list
if [ ! -f "$BENCH_DIR/combinations.txt" ]; then
    for r in 1 2 4 8 16; do
        for intra in 1 2 4 8 16; do
            for inter in 1 2 4 8 16; do
                echo "$r,$intra,$inter" >> "$BENCH_DIR/combinations.txt"
            done
        done
    done
fi

# Initialize state files
if [ ! -f "$BENCH_DIR/started.txt" ]; then echo "0" > "$BENCH_DIR/started.txt"; fi
if [ ! -f "$BENCH_DIR/concluded.txt" ]; then echo "0" > "$BENCH_DIR/concluded.txt"; fi
if [ ! -f "$BENCH_DIR/results.csv" ]; then echo "run_id,ranks,intra,inter,time_s,max_rss_mb,status" > "$BENCH_DIR/results.csv"; fi

if grep -q ",RUNNING$" "$BENCH_DIR/results.csv" 2>/dev/null; then
    sed -i 's/,TBD,TBD,RUNNING$/,-1,-1,TIMEOUT/' "$BENCH_DIR/results.csv"
fi

TOTAL_TASKS=$(wc -l < "$BENCH_DIR/combinations.txt")
STARTED=$(cat "$BENCH_DIR/started.txt")

# Chaining logic
if [ "$STARTED" -lt "$TOTAL_TASKS" ]; then
    echo "Scheduling next job in the chain..."
    CUR_PARTITION="${SLURM_JOB_PARTITION:-devel}"
    CUR_ACCOUNT="${SLURM_JOB_ACCOUNT:-default}"
    MEM_ARGS=()
    [ -n "$SLURM_MEM_PER_NODE" ] && MEM_ARGS=(--mem="$SLURM_MEM_PER_NODE")
    [ -n "$SLURM_MEM_PER_CPU" ] && MEM_ARGS=(--mem-per-cpu="$SLURM_MEM_PER_CPU")

    sbatch --dependency=afterany:$SLURM_JOB_ID \
        --partition="$CUR_PARTITION" \
        --account="$CUR_ACCOUNT" \
        "${MEM_ARGS[@]}" \
        --export=ALL,BENCH_MODEL="$BENCH_MODEL",BENCH_INPUTS="$BENCH_INPUTS",BENCH_SCHEMA="$BENCH_SCHEMA",BENCH_SEQ_LEN="$BENCH_SEQ_LEN",BENCH_FEAT_DIM="$BENCH_FEAT_DIM" \
        $0
fi

PREV_RESULT="N/A"

while [ "$STARTED" -lt "$TOTAL_TASKS" ]; do
    [ "$SECONDS" -ge 2400 ] && send_discord "⏳ **Job Handoff** (Progress: $STARTED / $TOTAL_TASKS)" && exit 0
    
    LINE_NUM=$(( STARTED + 1 ))
    COMBO=$(sed -n "${LINE_NUM}p" "$BENCH_DIR/combinations.txt")
    IFS=',' read -r RANKS INTRA INTER <<< "$COMBO"
    echo "$LINE_NUM" > "$BENCH_DIR/started.txt"
    RUN_FOLDER="$BENCH_DIR/runs/run_$STARTED"
    mkdir -p "$RUN_FOLDER"

    TOTAL_MODEL_COPIES_MB=$(echo "scale=2; $MODEL_FILE_SIZE_MB * $RANKS" | bc)
    MSG="📊 **CPU Benchmark Progress** ($LINE_NUM / $TOTAL_TASKS)
**Previous Run:** $PREV_RESULT
**Current Run:**
• Model: \`$BENCH_MODEL\` [Schema: $BENCH_SCHEMA] (${MODEL_FILE_SIZE_MB} MB file)
• Layout: $RANKS ranks, $INTRA intra, $INTER inter
• Workload: $BENCH_INPUTS total inputs (${TOTAL_INPUT_DATA_MB} MB total data)
• Theoretical Node Baseline: $(echo "$TOTAL_MODEL_COPIES_MB + $TOTAL_INPUT_DATA_MB" | bc) MB ($RANKS model copies + data)
• Allocation: 16 cores ($((16/RANKS)) per rank)
Status: 🏃 Running..."
    send_discord "$MSG"
    
    echo "$STARTED,$RANKS,$INTRA,$INTER,TBD,TBD,RUNNING" >> "$BENCH_DIR/results.csv"
    CORES_PER_RANK=$(( 16 / RANKS ))
    [ $CORES_PER_RANK -lt 1 ] && CORES_PER_RANK=1
    export OMP_NUM_THREADS=$INTRA
    OUTPUT_FILE="$RUN_FOLDER/output.txt"
    PYTHONUNBUFFERED=1 srun --exact -n $RANKS -c $CORES_PER_RANK \
        python3 -u benchmark.py \
            --model "$MODEL_CPU" \
            --num-inputs "$BENCH_INPUTS" \
            --schema "$BENCH_SCHEMA" \
            --seq-len "$BENCH_SEQ_LEN" \
            --feature-dim "$BENCH_FEAT_DIM" \
            --intra $INTRA --inter $INTER 2>&1 | tee "$OUTPUT_FILE"
    RC=${PIPESTATUS[0]}
    
    if [ $RC -eq 0 ]; then
        RES_LINE=$(grep "^RESULT:" "$OUTPUT_FILE" | tail -n 1 | cut -d':' -f2)
        if [ -n "$RES_LINE" ]; then
            if [[ "$RES_LINE" == "EXPECTED_TIMEOUT"* ]]; then
                sed -i "s/^$STARTED,.*/$STARTED,$RANKS,$INTRA,$INTER,-2,-2,EXPECTED_TIMEOUT/" "$BENCH_DIR/results.csv"
                PREV_RESULT="⚠️ Skipped (Expected Timeout)"
            else
                sed -i "s/^$STARTED,.*/$STARTED,$RANKS,$INTRA,$INTER,$RES_LINE,SUCCESS/" "$BENCH_DIR/results.csv"
                IFS=',' read -r T_S M_MB <<< "$RES_LINE"
                PREV_RESULT="✅ ${T_S}s | ${M_MB}MB"
            fi
        else
            sed -i "s/^$STARTED,.*/$STARTED,$RANKS,$INTRA,$INTER,-1,-1,FAILED_PARSE/" "$BENCH_DIR/results.csv"
            PREV_RESULT="❌ Parse Failure"
        fi
    else
        sed -i "s/^$STARTED,.*/$STARTED,$RANKS,$INTRA,$INTER,-1,-1,FAILED_RC_${RC}/" "$BENCH_DIR/results.csv"
        PREV_RESULT="❌ Error (RC $RC)"
    fi
    echo "$LINE_NUM" > "$BENCH_DIR/concluded.txt"
    STARTED=$LINE_NUM
done

# --- Final Analysis ---
TOTAL_END_TS=$(date +%s)
TOTAL_START_TS=$(cat "$BENCH_DIR/total_start_timestamp.txt" 2>/dev/null || echo "$TOTAL_END_TS")
TOTAL_ELAPSED=$((TOTAL_END_TS - TOTAL_START_TS))
fmt_duration() { local s=$1; printf '%dh %dm %ds' $((s/3600)) $((s%3600/60)) $((s%60)); }
HUMAN_DURATION=$(fmt_duration $TOTAL_ELAPSED)

ANALYSIS_SUMMARY="*No analysis performed.*"
if [ -f "$BENCH_DIR/results.csv" ]; then
    echo "Running Gemini analysis on results.csv..."
    RESULTS_DATA=$(cat "$BENCH_DIR/results.csv")
    QUERY="Analyze the following PyTorch CPU benchmark results for a model called '${BENCH_MODEL}' (Schema: ${BENCH_SCHEMA}) with ${BENCH_INPUTS} inputs total.
The benchmarks were run on a 16-core CPU node.
Hardware/Model Context:
- Model File Size: ${MODEL_FILE_SIZE_MB} MB
- Total Inputs: ${BENCH_INPUTS} (distributed across all ranks)
- Theoretical Minimum Data Size: ${TOTAL_INPUT_DATA_MB} MB
- Each rank loads its own copy of the model.

Results Data (CSV):
\`\`\`csv
${RESULTS_DATA}
\`\`\`

Please perform a TWO-PART analysis:
1. **Detailed Analysis**: Examination of results, optimal configuration, memory scaling, and oversubscription impact.
2. **Concise Summary**: Short summary (3-5 bullet points) for Discord. Use '###' for headers.
Provide response as JSON object with keys 'detailed' and 'summary'."

    GEMINI_CMD="/home/ro092286/.npm-global/bin/gemini"
    ANALYSIS_TEMP=$(mktemp)
    module load foss/2024a nodejs/20.13.1 > /dev/null 2>&1
    "$GEMINI_CMD" -p "$QUERY" --output-format json > "$ANALYSIS_TEMP" 2>/dev/null
    
    if [ -s "$ANALYSIS_TEMP" ]; then
        ANALYSIS_DETAILED=$(python3 -c "import sys, json; data = json.load(sys.stdin); resp = data.get('response', ''); start = resp.find('{'); end = resp.rfind('}') + 1; print(json.loads(resp[start:end])['detailed'])" < "$ANALYSIS_TEMP" 2>/dev/null)
        ANALYSIS_SUMMARY=$(python3 -c "import sys, json; data = json.load(sys.stdin); resp = data.get('response', ''); start = resp.find('{'); end = resp.rfind('}') + 1; print(json.loads(resp[start:end])['summary'])" < "$ANALYSIS_TEMP" 2>/dev/null)
        if [ -n "$ANALYSIS_DETAILED" ]; then
            ANALYSIS_FILE="${BENCH_DIR}/analysis.md"
            echo -e "# CPU Benchmark Analysis: ${BENCH_MODEL} [${BENCH_SCHEMA}]\n\n${ANALYSIS_DETAILED}" > "$ANALYSIS_FILE"
            UPLOAD_BASE_URL="https://christian-f-brinkmann.de/uploads"
            UPLOAD_CRED_FILE="${HOME}/.swatch_upload"
            if [ -f "$UPLOAD_CRED_FILE" ]; then
                CREDS=$(cat "$UPLOAD_CRED_FILE"); TS=$(date +%Y%m%d_%H%M%S); REMOTE_NAME="bench_${BENCH_MODEL}_${TS}.md"
                curl -s -u "$CREDS" -T "$ANALYSIS_FILE" "${UPLOAD_BASE_URL}/${REMOTE_NAME}" > /dev/null 2>&1
                MD_URL="${UPLOAD_BASE_URL}/${REMOTE_NAME}"
                PDF_RESP=$(curl -s -u "$CREDS" -H "Content-Type: application/json" -X POST -d "{\"filename\": \"${REMOTE_NAME}\"}" "${UPLOAD_BASE_URL}/convert" 2>/dev/null)
                PDF_URL=$(echo "$PDF_RESP" | jq -r '.pdf_filename // empty' 2>/dev/null)
                [ -n "$PDF_URL" ] && LINKS_STR="\n🔗 [Detailed Analysis](${MD_URL}) | [PDF Version](${UPLOAD_BASE_URL}/${PDF_URL})" || LINKS_STR="\n🔗 [Detailed Analysis](${MD_URL})"
            fi
        fi
    fi
    [ -z "$ANALYSIS_SUMMARY" ] || [ "$ANALYSIS_SUMMARY" == "null" ] && ANALYSIS_SUMMARY="*(Gemini analysis failed)*"
    rm -f "$ANALYSIS_TEMP"
fi

FINISH_MSG="🏁 **Benchmark Complete!**
**Config:** \`$BENCH_MODEL\` [$BENCH_SCHEMA] | $BENCH_INPUTS inputs
• Total Tasks: $TOTAL_TASKS
• Total Time: $HUMAN_DURATION
• Started: $(date -d "@$TOTAL_START_TS" '+%Y-%m-%d %H:%M:%S')
• Finished: $(date -d "@$TOTAL_END_TS" '+%Y-%m-%d %H:%M:%S')

**Summary:**
$ANALYSIS_SUMMARY
$LINKS_STR

\`Full Report: $BENCH_DIR/analysis.md\`"
send_discord "$FINISH_MSG"
echo "Benchmark complete! Total time: $HUMAN_DURATION"
echo "$FINISH_MSG" > "$BENCH_DIR/summary.txt"

#!/bin/bash

echo "Starting GPU Benchmarks..."

# Configuration
# Based on PyTorch device list:
# Device 0: NVIDIA H100 | Memory: 93.10 GB  (Likely GPU 0)
# Device 1: NVIDIA H100 | Memory: 93.10 GB  (Likely GPU 1)
# Device 2: NVIDIA H100 | Memory: 93.10 GB  (Likely GPU 3 - Exclusive)
# Device 3: NVIDIA H100 MIG 1g.12gb         (Likely MIG from GPU 2)
GPU_ID=2

run_bench() {
    local model_name=$1
    local inputs=$2
    local schema=$3
    local model_path=$4

    echo "----------------------------------------------------------"
    echo "Model: $model_path"
    echo "Inputs: $inputs | Schema: $schema"
    
    python3 benchmark_gpu.py \
        --gpu $GPU_ID \
        --model "$model_path" \
        --num-inputs $inputs \
        --schema $schema
}

# Define models to test
MODELS=(
    "giant,100000,mini_app,../mini_app/train_models/model_a/giant"
    "transformer,1000000,mini_app,../mini_app/train_models/model_a/transformer"
    "mmcp_transformer,10000,mmcp,../MMCP_TOM/input/transformer_inference_scripted_fw2.pt"
    "perfect,100000000,mini_app,../mini_app/train_models/model_a/perfect"
    "watercnn,10000000,mini_app,../mini_app/train_models/model_a/watercnn"
)

for m in "${MODELS[@]}"; do
    IFS=',' read -r NAME INPUTS SCHEMA BASE_PATH <<< "$m"
    echo "=========================================================="
    echo "BENCHMARKING: $NAME"
    
    # Check for _cpu.pt and _cuda.pt variants if it's not a direct file path
    if [[ "$BASE_PATH" == *".pt" ]]; then
        run_bench "$NAME" "$INPUTS" "$SCHEMA" "$BASE_PATH"
    else
        if [ -f "${BASE_PATH}_cpu.pt" ]; then
            run_bench "${NAME} (CPU-scripted)" "$INPUTS" "$SCHEMA" "${BASE_PATH}_cpu.pt"
        fi
        if [ -f "${BASE_PATH}_cuda.pt" ]; then
            run_bench "${NAME} (CUDA-scripted)" "$INPUTS" "$SCHEMA" "${BASE_PATH}_cuda.pt"
        fi
    fi
done

echo "=========================================================="
echo "GPU Benchmarks completed."

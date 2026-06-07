BASE_DIR="/test"
MODELS="
transformer|1000000|mini_app|${BASE_DIR}/mini_app/train_models/model_a/transformer_cpu.pt
mmcp_transformer|10000|mmcp|${BASE_DIR}/MMCP_TOM/input/transformer_inference_scripted_fw2.pt
"
for m in $MODELS; do
    echo "[$m]"
    MODEL_NAME=$(echo "$m" | cut -d'|' -f1)
    echo "MODEL_NAME=[$MODEL_NAME]"
done

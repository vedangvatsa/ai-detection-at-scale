#!/usr/bin/env bash
# Cloud training runner for 2026 models with QLoRA on a single A100 80GB.
# Works on RunPod, Vast.ai, Lambda Labs, or any CUDA Ubuntu instance.
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/run_cloud_qlora.sh Qwen/Qwen3.6-27B
#
# Notes:
#   - Default: Qwen/Qwen3.6-27B (released 2026-04). Fast and accurate on a single A100 80GB.
#   - For 70B+ models (Llama-3.3, Qwen3.5/3.6 MoE, etc.) use 2x A100 80GB or H100 and multi-GPU settings.

set -euo pipefail

MODEL_NAME="${1:-Qwen/Qwen3.6-27B}"
REPO_DIR="/workspace/ai-detection-at-scale"
OUT_DIR="/workspace/models/turingbench_$(echo "$MODEL_NAME" | tr '/-' '_' | tr '[:upper:]' '[:lower:]')"
HUB_USERNAME="${HF_USERNAME:-vedangvatsa123}"
HUB_MODEL_ID="${HUB_USERNAME}/vedang-turingbench-$(echo "$MODEL_NAME" | tr '/-' '_' | tr '[:upper:]' '[:lower:]')"

# Load HF_TOKEN from .env if not already set
if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO_DIR/.env" ]; then
    export HF_TOKEN=$(grep "^HF_TOKEN=" "$REPO_DIR/.env" | cut -d'=' -f2-)
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Provide it as env var or in $REPO_DIR/.env"
    exit 1
fi

echo "Model: $MODEL_NAME"
echo "Output: $OUT_DIR"
echo "Hub: $HUB_MODEL_ID"

# Clone repo if not present, otherwise pull latest
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/vedangvatsa/ai-detection-at-scale.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull origin main

# Install dependencies if needed
pip install -q -r requirements-qlora.txt

# Run QLoRA training
python3 scripts/train_qlora_classifier.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUT_DIR" \
    --hub_model_id "$HUB_MODEL_ID" \
    --max_length 512 \
    --epochs 1 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --seed 42

#!/usr/bin/env bash
# Cloud training runner for RunPod, Vast.ai, Lambda, or any CUDA Ubuntu instance.
# Reads HF_TOKEN from environment or .env so it is never hardcoded.
# Usage:
#   HF_TOKEN=hf_... bash scripts/run_finetune_cloud.sh roberta-large
#   HF_TOKEN=hf_... bash scripts/run_finetune_cloud.sh microsoft/deberta-v3-large

set -euo pipefail

MODEL_NAME="${1:-roberta-large}"
REPO_DIR="/workspace/ai-detection-at-scale"
OUT_DIR="/workspace/models/turingbench_$(echo "$MODEL_NAME" | tr '/-' '_')"

if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO_DIR/.env" ]; then
    export HF_TOKEN=$(grep "^HF_TOKEN=" "$REPO_DIR/.env" | cut -d'=' -f2-)
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Provide it as env var or in .env"
    exit 1
fi

HUB_SUFFIX="$(echo "$MODEL_NAME" | tr '/-' '_' | tr '[:upper:]' '[:lower:]')"
HUB_MODEL_ID="${HF_USERNAME:-vedangvatsa}/vedang-turingbench-${HUB_SUFFIX}"

# Resume from Hub if a checkpoint exists
RESUME_ARG=""
python3 - <<PY
import os
from huggingface_hub import list_repo_refs
try:
    refs = list(list_repo_refs(repo_id="$HUB_MODEL_ID").branches)
    if refs:
        with open("/tmp/resume_from.txt", "w") as f:
            f.write("$HUB_MODEL_ID")
        print(f"Resuming from Hub checkpoint: $HUB_MODEL_ID")
except Exception as e:
    print(f"No existing checkpoint on Hub: {e}")
PY

if [ -f /tmp/resume_from.txt ]; then
    RESUME_ARG="--resume_from_checkpoint $(cat /tmp/resume_from.txt)"
fi

cd "$REPO_DIR"
python3 scripts/33_finetune_turingbench.py \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUT_DIR" \
    --max_length 256 \
    --epochs 1 \
    --batch_size 48 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --hub_model_id "$HUB_MODEL_ID" \
    $RESUME_ARG \
    --seed 42

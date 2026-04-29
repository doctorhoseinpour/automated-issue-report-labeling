#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_11k_debias_qwen.sh — Debiased RAGTAG on 11-project benchmark (Qwen models)
# ============================================================================
# Runs debiased RAGTAG (margin=3) for Qwen-14B and Qwen-32B on all 11 projects,
# k=1,3,6,9. Expects baseline RAGTAG and neighbor files to already exist.
#
# Usage:
#   bash run_11k_debias_qwen.sh
#
# Estimated runtime: ~10-12 hours on RTX 4090 (14B ~4-5h, 32B ~5-6h)
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate

RESULTS="results/issues11k"
KS="1,3,6,9"
CTX=8192
MARGIN=3

PROJECTS=(
    "ansible_ansible" "bitcoin_bitcoin" "dart-lang_sdk"
    "dotnet_roslyn" "facebook_react" "flutter_flutter"
    "kubernetes_kubernetes" "microsoft_TypeScript" "microsoft_vscode"
    "opencv_opencv" "tensorflow_tensorflow"
)

MODELS=(
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit|unsloth_Qwen2_5_3B_Instruct_bnb_4bit"
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit|unsloth_Qwen2_5_7B_Instruct_bnb_4bit"
    "unsloth/Qwen2.5-14B-Instruct-bnb-4bit|unsloth_Qwen2_5_14B_Instruct_bnb_4bit"
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit|unsloth_Qwen2_5_32B_Instruct_bnb_4bit"
)

for MODEL_ENTRY in "${MODELS[@]}"; do
    MODEL="${MODEL_ENTRY%%|*}"
    TAG="${MODEL_ENTRY##*|}"

    echo ""
    echo "============================================================"
    echo "  Debiased RAGTAG — $MODEL"
    echo "  margin=$MARGIN, k=$KS, ctx=$CTX"
    echo "  Started: $(date)"
    echo "============================================================"

    for PROJ in "${PROJECTS[@]}"; do
        PRED_DIR="$RESULTS/project_specific/$PROJ/$TAG/ragtag_debias_m3/predictions"
        echo ""
        echo "=== $PROJ ==="
        python llm_labeler.py \
            --model "$MODEL" \
            --neighbors_dir "$RESULTS/project_specific/$PROJ/neighbors" \
            --top_ks "$KS" \
            --output_dir "$PRED_DIR" \
            --max_seq_length "$CTX" \
            --max_new_tokens 50 \
            --debias_retrieval \
            --debias_margin "$MARGIN"
    done

    echo ""
    echo "============================================================"
    echo "  $MODEL complete — $(date)"
    echo "============================================================"
done

echo ""
echo "============================================================"
echo "  All Qwen debias runs complete — $(date)"
echo "============================================================"

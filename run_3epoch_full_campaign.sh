#!/bin/bash
# Local 3-epoch FT campaign for Qwen-3B and Qwen-7B
# Covers 24 cells: 2 models × (1 agnostic + 11 project-specific)
# Idempotent: skips a cell if eval_finetune_fixed.csv already exists.
set -e
cd /home/ahosein/llm-labler

LOG_DIR="logs/local_3epoch_campaign"
mkdir -p "$LOG_DIR"

PROJECTS=(
    "ansible/ansible:ansible_ansible"
    "bitcoin/bitcoin:bitcoin_bitcoin"
    "dart-lang/sdk:dart-lang_sdk"
    "dotnet/roslyn:dotnet_roslyn"
    "facebook/react:facebook_react"
    "flutter/flutter:flutter_flutter"
    "kubernetes/kubernetes:kubernetes_kubernetes"
    "microsoft/TypeScript:microsoft_TypeScript"
    "microsoft/vscode:microsoft_vscode"
    "opencv/opencv:opencv_opencv"
    "tensorflow/tensorflow:tensorflow_tensorflow"
)

MODELS=(
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit:unsloth_Qwen2_5_3B_Instruct_bnb_4bit"
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit:unsloth_Qwen2_5_7B_Instruct_bnb_4bit"
)

run_ft() {
    local model="$1"
    local tag="$2"
    local out_dir="$3"
    local train_csv="$4"
    local test_csv="$5"

    # Skip if predictions already exist (eval CSV is generated post-hoc)
    if [[ -f "$out_dir/preds_finetune_fixed.csv" ]]; then
        echo "    SKIP: $out_dir/preds_finetune_fixed.csv already exists"
        return 0
    fi

    mkdir -p "$out_dir"
    echo "    Output: $out_dir"
    python fixed_fine-tune.py \
        --model "$model" \
        --dataset "issues11k.csv" \
        --train_csv "$train_csv" \
        --test_csv "$test_csv" \
        --max_seq_length 2048 \
        --max_new_tokens 50 \
        --inference_batch_size 1 \
        --output_dir "$out_dir"

    if [[ -f "$out_dir/preds_finetune_fixed.csv" ]]; then
        echo "    DONE: preds saved"
    else
        echo "    FAIL: no preds produced"
        return 1
    fi
}

START_TS=$(date +%s)
echo "=== Local 3-epoch campaign starting at $(date) ==="

for entry in "${MODELS[@]}"; do
    model="${entry%%:*}"
    tag="${entry##*:}"

    echo ""
    echo "############################################"
    echo "## MODEL: $model"
    echo "############################################"

    # PA
    echo ""
    echo "[PA] $tag"
    run_ft "$model" "$tag" \
        "results/issues11k/agnostic/$tag/finetune_fixed" \
        "results/issues11k/agnostic/neighbors/train_split.csv" \
        "results/issues11k/agnostic/neighbors/test_split.csv"

    # PS × 11
    for proj_entry in "${PROJECTS[@]}"; do
        proj="${proj_entry%%:*}"
        proj_tag="${proj_entry##*:}"
        echo ""
        echo "[PS-$proj_tag] $tag"
        run_ft "$model" "$tag" \
            "results/issues11k/project_specific/$proj_tag/$tag/finetune_fixed" \
            "results/issues11k/project_specific/$proj_tag/neighbors/train_split.csv" \
            "results/issues11k/project_specific/$proj_tag/neighbors/test_split.csv"
    done
done

END_TS=$(date +%s)
echo ""
echo "=== Done at $(date) (elapsed: $(( (END_TS - START_TS) / 60 )) min) ==="
echo ""
echo "=== Summary of new 3-epoch eval F1 scores ==="
for entry in "${MODELS[@]}"; do
    tag="${entry##*:}"
    short="${tag#unsloth_}"
    short="${short%_Instruct_bnb_4bit}"
    pa="results/issues11k/agnostic/$tag/finetune_fixed/eval_finetune_fixed.csv"
    if [[ -f "$pa" ]]; then
        echo "  $short PA:  $(awk -F, 'NR==2{print $21}' "$pa")"
    fi
    for proj_entry in "${PROJECTS[@]}"; do
        proj_tag="${proj_entry##*:}"
        ps_eval="results/issues11k/project_specific/$proj_tag/$tag/finetune_fixed/eval_finetune_fixed.csv"
        if [[ -f "$ps_eval" ]]; then
            echo "  $short PS-$proj_tag:  $(awk -F, 'NR==2{print $21}' "$ps_eval")"
        fi
    done
done

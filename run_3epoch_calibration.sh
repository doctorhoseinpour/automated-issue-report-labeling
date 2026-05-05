#!/bin/bash
# 3-epoch FT calibration vs existing 1-epoch baselines.
# Runs Qwen-3B and Qwen-7B FT on PA (3,300 train) and PS-facebook_react (300 train).
# Outputs to results/issues11k_3epoch/ to avoid overwriting 1-epoch results.
set -e
cd /home/ahosein/llm-labler

OUT_BASE="results/issues11k_3epoch"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR"

run_ft() {
    local model="$1"
    local tag="$2"
    local config="$3"
    local proj_tag="${4:-}"

    if [[ "$config" == "agnostic" ]]; then
        local train="results/issues11k/agnostic/neighbors/train_split.csv"
        local test="results/issues11k/agnostic/neighbors/test_split.csv"
        local out="$OUT_BASE/agnostic/$tag/finetune_fixed"
        local log="$LOG_DIR/PA_${tag}.log"
        echo ">>> [PA] $tag"
    else
        local train="results/issues11k/project_specific/${proj_tag}/neighbors/train_split.csv"
        local test="results/issues11k/project_specific/${proj_tag}/neighbors/test_split.csv"
        local out="$OUT_BASE/project_specific/${proj_tag}/$tag/finetune_fixed"
        local log="$LOG_DIR/PS_${proj_tag}_${tag}.log"
        echo ">>> [PS-${proj_tag}] $tag"
    fi

    mkdir -p "$out"
    python fixed_fine-tune.py \
        --model "$model" \
        --dataset "issues11k.csv" \
        --train_csv "$train" \
        --test_csv "$test" \
        --max_seq_length 2048 \
        --max_new_tokens 50 \
        --inference_batch_size 1 \
        --output_dir "$out" \
        > "$log" 2>&1
    local f1
    f1=$(awk -F, 'NR==2{print $21}' "$out/eval_finetune_fixed.csv" 2>/dev/null || echo "N/A")
    echo "    f1_macro=$f1"
}

echo "=== 3-epoch FT calibration starting at $(date) ==="

run_ft "unsloth/Qwen2.5-3B-Instruct-bnb-4bit" "unsloth_Qwen2_5_3B_Instruct_bnb_4bit" "agnostic"
run_ft "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" "unsloth_Qwen2_5_7B_Instruct_bnb_4bit" "agnostic"
run_ft "unsloth/Qwen2.5-3B-Instruct-bnb-4bit" "unsloth_Qwen2_5_3B_Instruct_bnb_4bit" "specific" "facebook_react"
run_ft "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" "unsloth_Qwen2_5_7B_Instruct_bnb_4bit" "specific" "facebook_react"

echo ""
echo "=== Done at $(date) ==="
echo ""
echo "=== 3-epoch results ==="
for f in "$OUT_BASE"/agnostic/*/finetune_fixed/eval_finetune_fixed.csv \
         "$OUT_BASE"/project_specific/*/*/finetune_fixed/eval_finetune_fixed.csv; do
    [[ -f "$f" ]] || continue
    f1=$(awk -F, 'NR==2{print $21}' "$f")
    echo "  ${f}: f1_macro=$f1"
done

echo ""
echo "=== 1-epoch baselines (for comparison) ==="
echo "  PA Qwen-3B: 0.652"
echo "  PA Qwen-7B: 0.7411"
echo "  PS-react Qwen-3B: 0.77"
echo "  PS-react Qwen-7B: 0.4889"

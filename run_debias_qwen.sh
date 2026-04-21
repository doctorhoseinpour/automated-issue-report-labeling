#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_debias_qwen.sh — Debiased retrieval for Qwen-14B and Qwen-32B
# ============================================================================
# Extends run_debias.sh (which covers Llama-3B/8B) to the Qwen models.
# k=9 for both Qwen models, margin=3 only.
#
# Usage:
#   LOCAL  (RTX 4090, 3k only):   bash run_debias_qwen.sh --skip_30k
#   REMOTE (A100, 30k only):      bash run_debias_qwen.sh --skip_3k --nrp
#   BOTH:                         bash run_debias_qwen.sh
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON_BIN:-venv/bin/python}"
MAX_NEW_TOKENS=50
BATCH_SIZE=1
CTX=8192
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
DEBIAS_MARGIN=3
K=9

MODEL_14B="unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
MODEL_32B="unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
TAG_14B="unsloth_Qwen2_5_14B_Instruct_bnb_4bit"
TAG_32B="unsloth_Qwen2_5_32B_Instruct_bnb_4bit"

# --- Parse CLI flags ---
SKIP_3K=0
SKIP_30K=0
NRP=0
MODEL_CACHE_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_3k)   SKIP_3K=1;   shift ;;
        --skip_30k)  SKIP_30K=1;  shift ;;
        --nrp)       NRP=1;       shift ;;
        *)  echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ "$NRP" -eq 1 ]]; then
    MODEL_CACHE_DIR="hf_cache"
    mkdir -p "$MODEL_CACHE_DIR"
    export HF_HOME="$MODEL_CACHE_DIR"
    export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
fi

CACHE_ARGS=()
if [[ -n "$MODEL_CACHE_DIR" ]]; then
    CACHE_ARGS=(--cache_dir "$MODEL_CACHE_DIR")
fi

echo "============================================================"
echo "  Debiased Retrieval — Qwen-14B & Qwen-32B"
echo "  k=${K}, margin=${DEBIAS_MARGIN}, ctx=${CTX}"
echo "  skip_3k=${SKIP_3K}, skip_30k=${SKIP_30K}, nrp=${NRP}"
echo "============================================================"

# ============================================================
# Step 0: Verify/rebuild neighbor files
# ============================================================

NB_DIR_3K="results/issues3k_debias/neighbors"
NB_DIR_30K="results/issues30k/neighbors"

if [[ "$SKIP_3K" -eq 0 ]]; then
    if [[ -f "${NB_DIR_3K}/neighbors_k${K}.csv" ]]; then
        echo "[OK] 3k neighbors exist: ${NB_DIR_3K}/neighbors_k${K}.csv"
    elif [[ -f "issues3k.csv" ]]; then
        echo "[0] Building 3k FAISS index + neighbors (k=${K})..."
        MC_ARG=""
        if [[ -n "$MODEL_CACHE_DIR" ]]; then
            MC_ARG="--model_cache_dir $MODEL_CACHE_DIR"
        fi
        $PYTHON build_and_query_index.py \
            --dataset issues3k.csv \
            --output_dir "$NB_DIR_3K" \
            --top_ks "$K" \
            --test_size 0.5 \
            --embedding_model "$EMBEDDING_MODEL" \
            $MC_ARG
    else
        echo "[WARN] issues3k.csv not found and 3k neighbors don't exist. Skipping 3k."
        SKIP_3K=1
    fi
fi

if [[ "$SKIP_30K" -eq 0 ]]; then
    if [[ -f "${NB_DIR_30K}/neighbors_k${K}.csv" ]]; then
        echo "[OK] 30k neighbors exist: ${NB_DIR_30K}/neighbors_k${K}.csv"
    elif [[ -f "issues30k.csv" ]]; then
        echo "[0] Building 30k FAISS index + neighbors (k=${K})..."
        MC_ARG=""
        if [[ -n "$MODEL_CACHE_DIR" ]]; then
            MC_ARG="--model_cache_dir $MODEL_CACHE_DIR"
        fi
        $PYTHON build_and_query_index.py \
            --dataset issues30k.csv \
            --output_dir "$NB_DIR_30K" \
            --top_ks "$K" \
            --test_size 3000 \
            --embedding_model "$EMBEDDING_MODEL" \
            $MC_ARG
    else
        echo "[ERROR] issues30k.csv not found and 30k neighbors don't exist."
        exit 1
    fi
fi

SUFFIX="m${DEBIAS_MARGIN}"

# ============================================================
# Helper: run debias for one model on one dataset
# ============================================================
run_debias() {
    local MODEL="$1"
    local TAG="$2"
    local SHORT_NAME="$3"
    local DATASET_LABEL="$4"    # "3k" or "30k"
    local NB_DIR="$5"

    local PRED_DIR="results/issues${DATASET_LABEL}_debias_${SUFFIX}/${TAG}/ragtag/predictions"
    local PRED_FILE="${PRED_DIR}/preds_k${K}.csv"

    if [[ -f "$PRED_FILE" ]]; then
        echo "  SKIP: ${SHORT_NAME} on ${DATASET_LABEL} — ${PRED_FILE} exists"
        return
    fi

    echo -e "\n  ${SHORT_NAME} on ${DATASET_LABEL}, k=${K}, margin=${DEBIAS_MARGIN}"
    mkdir -p "$PRED_DIR"

    $PYTHON llm_labeler.py \
        --model "$MODEL" \
        --neighbors_dir "$NB_DIR" \
        --output_dir "$PRED_DIR" \
        --top_ks "$K" \
        --max_seq_length "$CTX" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --inference_batch_size "$BATCH_SIZE" \
        --load_in_4bit \
        --debias_retrieval \
        --debias_margin "$DEBIAS_MARGIN" \
        "${CACHE_ARGS[@]}"

    # Evaluate
    local EVAL_DIR="results/issues${DATASET_LABEL}_debias_${SUFFIX}/${TAG}/ragtag/evaluations"
    mkdir -p "$EVAL_DIR"
    for PRED in "$PRED_DIR"/preds_*.csv; do
        [ -f "$PRED" ] || continue
        local BASENAME
        BASENAME=$(basename "$PRED" .csv)
        local K_TAG="${BASENAME#preds_}"
        $PYTHON evaluate.py --preds_csv "$PRED" --output_csv "${EVAL_DIR}/eval_${K_TAG}.csv"
    done
}

# ============================================================
# Run debias for each model on each dataset
# ============================================================

if [[ "$SKIP_3K" -eq 0 ]]; then
    echo -e "\n--- 3k dataset, margin=${DEBIAS_MARGIN} ---"
    run_debias "$MODEL_14B" "$TAG_14B" "Qwen-14B" "3k" "$NB_DIR_3K"
    run_debias "$MODEL_32B" "$TAG_32B" "Qwen-32B" "3k" "$NB_DIR_3K"
fi

if [[ "$SKIP_30K" -eq 0 ]]; then
    echo -e "\n--- 30k dataset, margin=${DEBIAS_MARGIN} ---"
    run_debias "$MODEL_14B" "$TAG_14B" "Qwen-14B" "30k" "$NB_DIR_30K"
    run_debias "$MODEL_32B" "$TAG_32B" "Qwen-32B" "30k" "$NB_DIR_30K"
fi

# ============================================================
# Summary
# ============================================================
echo -e "\n============================================================"
echo "  Debiased Retrieval Results Summary (All Models)"
echo "============================================================"
$PYTHON -c "
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os

LABELS = ['bug', 'feature', 'question']

def eval_f1(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    valid = df[df['predicted_label'].isin(LABELS)]
    if len(valid) == 0:
        return None
    p,r,f,_ = precision_recall_fscore_support(valid['ground_truth'], valid['predicted_label'], labels=LABELS, average=None, zero_division=0)
    _,_,f1m,_ = precision_recall_fscore_support(valid['ground_truth'], valid['predicted_label'], labels=LABELS, average='macro', zero_division=0)
    acc = accuracy_score(valid['ground_truth'], valid['predicted_label'])
    inv = (~df['predicted_label'].isin(LABELS)).mean()
    return f1m, acc, inv, {l: {'P':p[i],'R':r[i],'F1':f[i]} for i,l in enumerate(LABELS)}

print(f\"{'Model':>12} {'Data':>4} {'Type':>10} {'F1_macro':>8} {'Acc':>7} {'Inv%':>6} | {'F1_bug':>7} {'F1_feat':>7} {'F1_ques':>8}\")
print('-' * 90)

configs = [
    # Llama models (from run_debias.sh — existing results)
    ('Llama-3B', '3k', 'k3', 'unsloth_Llama_3_2_3B_Instruct'),
    ('Llama-8B', '3k', 'k9', 'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit'),
    ('Qwen-14B', '3k', 'k9', 'unsloth_Qwen2_5_14B_Instruct_bnb_4bit'),
    ('Qwen-32B', '3k', 'k9', 'unsloth_Qwen2_5_32B_Instruct_bnb_4bit'),
    ('Llama-3B', '30k', 'k3', 'unsloth_Llama_3_2_3B_Instruct'),
    ('Llama-8B', '30k', 'k9', 'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit'),
    ('Qwen-14B', '30k', 'k9', 'unsloth_Qwen2_5_14B_Instruct_bnb_4bit'),
    ('Qwen-32B', '30k', 'k9', 'unsloth_Qwen2_5_32B_Instruct_bnb_4bit'),
]

for name, ds, k_label, tag in configs:
    # Baseline
    if ds == '3k':
        base_path = f'results/issues3k_ctx8192/all_results.csv'
        if os.path.exists(base_path):
            try:
                agg = pd.read_csv(base_path)
                k_val = int(k_label.replace('k',''))
                model_full = tag.replace('_', '/', 1).replace('_', '-').replace('unsloth/', 'unsloth/')
                # Just try to find baseline from prediction files instead
            except: pass
        # Try prediction file directly
        base_pred = f'results/issues3k_debias/neighbors/../{tag}/ragtag/predictions/preds_{k_label}.csv'
    else:
        base_pred = f'results/issues30k/{tag}/ragtag/predictions/preds_{k_label}.csv'
        if not os.path.exists(base_pred):
            base_pred = f'results/issues30k_k_study/{tag}/ragtag/predictions/preds_{k_label}.csv'

    if ds == '3k':
        # Try ctx8192 results
        base_pred = f'results/issues3k_ctx8192/{tag}/ragtag/predictions/preds_{k_label}.csv'

    r = eval_f1(base_pred) if os.path.exists(base_pred) else None
    if r:
        f1m, acc, inv, pc = r
        print(f\"  {name:>10} {ds:>4} {'baseline':>10} {f1m:>8.4f} {acc:>7.4f} {inv:>6.1%} | {pc['bug']['F1']:>7.3f} {pc['feature']['F1']:>7.3f} {pc['question']['F1']:>8.3f}\")

    # Debiased
    debias_path = f'results/issues{ds}_debias_m3/{tag}/ragtag/predictions/preds_{k_label}.csv'
    r = eval_f1(debias_path)
    if r:
        f1m, acc, inv, pc = r
        print(f\"  {name:>10} {ds:>4} {'debias_m3':>10} {f1m:>8.4f} {acc:>7.4f} {inv:>6.1%} | {pc['bug']['F1']:>7.3f} {pc['feature']['F1']:>7.3f} {pc['question']['F1']:>8.3f}\")

    # FT (30k only)
    if ds == '30k':
        ft_path = f'results/issues30k/{tag}/finetune_fixed/preds_finetune_fixed.csv'
        r = eval_f1(ft_path)
        if r:
            f1m, acc, inv, pc = r
            print(f\"  {name:>10} {ds:>4} {'FT':>10} {f1m:>8.4f} {acc:>7.4f} {inv:>6.1%} | {pc['bug']['F1']:>7.3f} {pc['feature']['F1']:>7.3f} {pc['question']['F1']:>8.3f}\")
    print()
"

echo -e "\nDone!"

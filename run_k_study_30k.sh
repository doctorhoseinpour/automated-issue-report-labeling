#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_k_study_30k.sh — K-value study on 30k for Llama-3B and Llama-8B
# ============================================================================
# Tests k=1,3,5,9,15 with ctx=8192 on the 30k dataset.
# Results go to results/issues30k_k_study/ to keep existing results clean.
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON_BIN:-venv/bin/python}"
DATASET="issues30k.csv"
TEST_SIZE=3000
CTX=8192
TOP_KS="1,3,5,9,15"
MAX_NEW_TOKENS=50
BATCH_SIZE=1
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

RESULTS_DIR="results/issues30k_k_study"
NEIGHBORS_DIR="${RESULTS_DIR}/neighbors"

echo "============================================================"
echo "  K-value study on 30k dataset"
echo "  K values: ${TOP_KS}"
echo "  Context:  ${CTX}"
echo "============================================================"

# --- Step 1: Build FAISS index + retrieve neighbors ---
TRAIN_CSV="${NEIGHBORS_DIR}/train_split.csv"
TEST_CSV="${NEIGHBORS_DIR}/test_split.csv"

if [ -f "$TRAIN_CSV" ] && [ -f "$TEST_CSV" ]; then
    echo -e "\n[SKIP] Splits already exist. Checking neighbor files..."
    NEED_BUILD=false
    for K in 1 3 5 9 15; do
        if [ ! -f "${NEIGHBORS_DIR}/neighbors_k${K}.csv" ]; then
            echo "  Missing neighbors_k${K}.csv — need to rebuild"
            NEED_BUILD=true
            break
        fi
    done
    if [ "$NEED_BUILD" = false ]; then
        echo "  All neighbor files present, skipping index build"
    fi
else
    NEED_BUILD=true
fi

if [ "$NEED_BUILD" = true ]; then
    echo -e "\n[1/3] Building FAISS index + retrieving neighbors for k=${TOP_KS}..."
    $PYTHON build_and_query_index.py \
        --dataset "$DATASET" \
        --output_dir "$NEIGHBORS_DIR" \
        --top_ks "$TOP_KS" \
        --test_size "$TEST_SIZE" \
        --embedding_model "$EMBEDDING_MODEL"
fi

# --- Step 2: Llama-3B ---
MODEL_3B="unsloth/Llama-3.2-3B-Instruct"
TAG_3B="unsloth_Llama_3_2_3B_Instruct"
PRED_DIR_3B="${RESULTS_DIR}/${TAG_3B}/ragtag/predictions"
EVAL_DIR_3B="${RESULTS_DIR}/${TAG_3B}/ragtag/evaluations"

echo -e "\n[2/3] Llama-3B — RAGTAG k=${TOP_KS}, ctx=${CTX}"
$PYTHON llm_labeler.py \
    --model "$MODEL_3B" \
    --neighbors_dir "$NEIGHBORS_DIR" \
    --output_dir "$PRED_DIR_3B" \
    --top_ks "$TOP_KS" \
    --max_seq_length "$CTX" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --inference_batch_size "$BATCH_SIZE" \
    --load_in_4bit

echo -e "\n  Evaluating Llama-3B predictions..."
mkdir -p "$EVAL_DIR_3B"
for PRED_FILE in "$PRED_DIR_3B"/preds_*.csv; do
    [ -f "$PRED_FILE" ] || continue
    BASENAME=$(basename "$PRED_FILE" .csv)
    K_TAG="${BASENAME#preds_}"
    $PYTHON evaluate.py --preds_csv "$PRED_FILE" --output_csv "${EVAL_DIR_3B}/eval_${K_TAG}.csv"
done

# --- Step 3: Llama-8B ---
MODEL_8B="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
TAG_8B="unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit"
PRED_DIR_8B="${RESULTS_DIR}/${TAG_8B}/ragtag/predictions"
EVAL_DIR_8B="${RESULTS_DIR}/${TAG_8B}/ragtag/evaluations"

echo -e "\n[3/3] Llama-8B — RAGTAG k=${TOP_KS}, ctx=${CTX}"
$PYTHON llm_labeler.py \
    --model "$MODEL_8B" \
    --neighbors_dir "$NEIGHBORS_DIR" \
    --output_dir "$PRED_DIR_8B" \
    --top_ks "$TOP_KS" \
    --max_seq_length "$CTX" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --inference_batch_size "$BATCH_SIZE" \
    --load_in_4bit

echo -e "\n  Evaluating Llama-8B predictions..."
mkdir -p "$EVAL_DIR_8B"
for PRED_FILE in "$PRED_DIR_8B"/preds_*.csv; do
    [ -f "$PRED_FILE" ] || continue
    BASENAME=$(basename "$PRED_FILE" .csv)
    K_TAG="${BASENAME#preds_}"
    $PYTHON evaluate.py --preds_csv "$PRED_FILE" --output_csv "${EVAL_DIR_8B}/eval_${K_TAG}.csv"
done

# --- Summary ---
echo -e "\n============================================================"
echo "  K-study results summary"
echo "============================================================"
$PYTHON -c "
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

LABELS = ['bug', 'feature', 'question']

def eval_f1(path):
    df = pd.read_csv(path)
    valid = df[df['predicted_label'].isin(LABELS)]
    p,r,f,_ = precision_recall_fscore_support(valid['ground_truth'], valid['predicted_label'], labels=LABELS, average=None, zero_division=0)
    _,_,f1m,_ = precision_recall_fscore_support(valid['ground_truth'], valid['predicted_label'], labels=LABELS, average='macro', zero_division=0)
    acc = accuracy_score(valid['ground_truth'], valid['predicted_label'])
    inv = (~df['predicted_label'].isin(LABELS)).mean()
    return f1m, acc, inv, {l: {'P':p[i],'R':r[i],'F1':f[i]} for i,l in enumerate(LABELS)}

print(f\"{'Model':>12} {'k':>3} {'F1_macro':>8} {'Acc':>7} {'Inv%':>6} | {'F1_bug':>7} {'F1_feat':>7} {'F1_ques':>8} | {'R_bug':>6} {'R_ques':>7}\")
print('-' * 90)

import os
for tag, name in [('unsloth_Llama_3_2_3B_Instruct', 'Llama-3B'), ('unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit', 'Llama-8B')]:
    for k_label in ['k1', 'k3', 'k5', 'k9', 'k15']:
        path = f'results/issues30k_k_study/{tag}/ragtag/predictions/preds_{k_label}.csv'
        if not os.path.exists(path):
            continue
        f1m, acc, inv, pc = eval_f1(path)
        print(f\"  {name:>10} {k_label:>3} {f1m:>8.4f} {acc:>7.4f} {inv:>6.1%} | {pc['bug']['F1']:>7.3f} {pc['feature']['F1']:>7.3f} {pc['question']['F1']:>8.3f} | {pc['bug']['R']:>6.3f} {pc['question']['R']:>7.3f}\")
    print()

# Also show FT baseline for comparison
for tag, name, ft_path in [
    ('unsloth_Llama_3_2_3B_Instruct', 'Llama-3B', 'results/issues30k/unsloth_Llama_3_2_3B_Instruct/finetune_fixed/preds_finetune_fixed.csv'),
    ('unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit', 'Llama-8B', 'results/issues30k/unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit/finetune_fixed/preds_finetune_fixed.csv'),
]:
    if os.path.exists(ft_path):
        f1m, acc, inv, pc = eval_f1(ft_path)
        print(f\"  {name:>10}  FT {f1m:>8.4f} {acc:>7.4f} {inv:>6.1%} | {pc['bug']['F1']:>7.3f} {pc['feature']['F1']:>7.3f} {pc['question']['F1']:>8.3f} | {pc['bug']['R']:>6.3f} {pc['question']['R']:>7.3f}\")
"

echo -e "\nDone!"

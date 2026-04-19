#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_debias.sh — Test debiased retrieval on Llama-3B and Llama-8B
# ============================================================================
# Removes bug neighbors when question count is close to or exceeds bug count.
# Tests margins 3 and 2, on both 3k and 30k datasets.
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON_BIN:-venv/bin/python}"
MAX_NEW_TOKENS=50
BATCH_SIZE=1
CTX=8192
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

MODEL_3B="unsloth/Llama-3.2-3B-Instruct"
MODEL_8B="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
TAG_3B="unsloth_Llama_3_2_3B_Instruct"
TAG_8B="unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit"

# ============================================================
# Step 0: Rebuild 3k FAISS index (original neighbor files were cleaned up)
# ============================================================
NB_DIR_3K="results/issues3k_debias/neighbors"

if [ -f "${NB_DIR_3K}/neighbors_k9.csv" ]; then
    echo "[SKIP] 3k neighbors already exist"
else
    echo "[0] Rebuilding 3k FAISS index + neighbors (k=3,9)..."
    $PYTHON build_and_query_index.py \
        --dataset issues3k.csv \
        --output_dir "$NB_DIR_3K" \
        --top_ks "3,9" \
        --test_size 0.5 \
        --embedding_model "$EMBEDDING_MODEL"
fi

NB_DIR_30K="results/issues30k/neighbors"

# ============================================================
# Run both margins
# ============================================================
for DEBIAS_MARGIN in 3 2; do

echo -e "\n============================================================"
echo "  MARGIN = ${DEBIAS_MARGIN}"
echo "============================================================"

SUFFIX="m${DEBIAS_MARGIN}"

# --- 3k ---
echo -e "\n--- 3k dataset, margin=${DEBIAS_MARGIN} ---"

PRED_3B_3K="results/issues3k_debias_${SUFFIX}/${TAG_3B}/ragtag/predictions"
echo -e "\n  Llama-3B on 3k, k=3, margin=${DEBIAS_MARGIN}"
$PYTHON llm_labeler.py \
    --model "$MODEL_3B" \
    --neighbors_dir "$NB_DIR_3K" \
    --output_dir "$PRED_3B_3K" \
    --top_ks "3" \
    --max_seq_length "$CTX" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --inference_batch_size "$BATCH_SIZE" \
    --load_in_4bit \
    --debias_retrieval \
    --debias_margin "$DEBIAS_MARGIN"

EVAL_3B_3K="results/issues3k_debias_${SUFFIX}/${TAG_3B}/ragtag/evaluations"
mkdir -p "$EVAL_3B_3K"
for PRED_FILE in "$PRED_3B_3K"/preds_*.csv; do
    [ -f "$PRED_FILE" ] || continue
    BASENAME=$(basename "$PRED_FILE" .csv)
    K_TAG="${BASENAME#preds_}"
    $PYTHON evaluate.py --preds_csv "$PRED_FILE" --output_csv "${EVAL_3B_3K}/eval_${K_TAG}.csv"
done

PRED_8B_3K="results/issues3k_debias_${SUFFIX}/${TAG_8B}/ragtag/predictions"
echo -e "\n  Llama-8B on 3k, k=9, margin=${DEBIAS_MARGIN}"
$PYTHON llm_labeler.py \
    --model "$MODEL_8B" \
    --neighbors_dir "$NB_DIR_3K" \
    --output_dir "$PRED_8B_3K" \
    --top_ks "9" \
    --max_seq_length "$CTX" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --inference_batch_size "$BATCH_SIZE" \
    --load_in_4bit \
    --debias_retrieval \
    --debias_margin "$DEBIAS_MARGIN"

EVAL_8B_3K="results/issues3k_debias_${SUFFIX}/${TAG_8B}/ragtag/evaluations"
mkdir -p "$EVAL_8B_3K"
for PRED_FILE in "$PRED_8B_3K"/preds_*.csv; do
    [ -f "$PRED_FILE" ] || continue
    BASENAME=$(basename "$PRED_FILE" .csv)
    K_TAG="${BASENAME#preds_}"
    $PYTHON evaluate.py --preds_csv "$PRED_FILE" --output_csv "${EVAL_8B_3K}/eval_${K_TAG}.csv"
done

# --- 30k ---
echo -e "\n--- 30k dataset, margin=${DEBIAS_MARGIN} ---"

PRED_3B_30K="results/issues30k_debias_${SUFFIX}/${TAG_3B}/ragtag/predictions"
echo -e "\n  Llama-3B on 30k, k=3, margin=${DEBIAS_MARGIN}"
$PYTHON llm_labeler.py \
    --model "$MODEL_3B" \
    --neighbors_dir "$NB_DIR_30K" \
    --output_dir "$PRED_3B_30K" \
    --top_ks "3" \
    --max_seq_length "$CTX" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --inference_batch_size "$BATCH_SIZE" \
    --load_in_4bit \
    --debias_retrieval \
    --debias_margin "$DEBIAS_MARGIN"

EVAL_3B_30K="results/issues30k_debias_${SUFFIX}/${TAG_3B}/ragtag/evaluations"
mkdir -p "$EVAL_3B_30K"
for PRED_FILE in "$PRED_3B_30K"/preds_*.csv; do
    [ -f "$PRED_FILE" ] || continue
    BASENAME=$(basename "$PRED_FILE" .csv)
    K_TAG="${BASENAME#preds_}"
    $PYTHON evaluate.py --preds_csv "$PRED_FILE" --output_csv "${EVAL_3B_30K}/eval_${K_TAG}.csv"
done

PRED_8B_30K="results/issues30k_debias_${SUFFIX}/${TAG_8B}/ragtag/predictions"
echo -e "\n  Llama-8B on 30k, k=9, margin=${DEBIAS_MARGIN}"
$PYTHON llm_labeler.py \
    --model "$MODEL_8B" \
    --neighbors_dir "$NB_DIR_30K" \
    --output_dir "$PRED_8B_30K" \
    --top_ks "9" \
    --max_seq_length "$CTX" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --inference_batch_size "$BATCH_SIZE" \
    --load_in_4bit \
    --debias_retrieval \
    --debias_margin "$DEBIAS_MARGIN"

EVAL_8B_30K="results/issues30k_debias_${SUFFIX}/${TAG_8B}/ragtag/evaluations"
mkdir -p "$EVAL_8B_30K"
for PRED_FILE in "$PRED_8B_30K"/preds_*.csv; do
    [ -f "$PRED_FILE" ] || continue
    BASENAME=$(basename "$PRED_FILE" .csv)
    K_TAG="${BASENAME#preds_}"
    $PYTHON evaluate.py --preds_csv "$PRED_FILE" --output_csv "${EVAL_8B_30K}/eval_${K_TAG}.csv"
done

done  # end margin loop

# ============================================================
# Summary
# ============================================================
echo -e "\n============================================================"
echo "  Debiased Retrieval Results Summary"
echo "============================================================"
$PYTHON -c "
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os

LABELS = ['bug', 'feature', 'question']

def eval_f1(path):
    df = pd.read_csv(path)
    valid = df[df['predicted_label'].isin(LABELS)]
    p,r,f,_ = precision_recall_fscore_support(valid['ground_truth'], valid['predicted_label'], labels=LABELS, average=None, zero_division=0)
    _,_,f1m,_ = precision_recall_fscore_support(valid['ground_truth'], valid['predicted_label'], labels=LABELS, average='macro', zero_division=0)
    acc = accuracy_score(valid['ground_truth'], valid['predicted_label'])
    inv = (~df['predicted_label'].isin(LABELS)).mean()
    return f1m, acc, inv, {l: {'P':p[i],'R':r[i],'F1':f[i]} for i,l in enumerate(LABELS)}

print(f\"{'Model':>12} {'Data':>4} {'Type':>10} {'F1_macro':>8} {'Acc':>7} {'Inv%':>6} | {'F1_bug':>7} {'F1_feat':>7} {'F1_ques':>8} | {'R_bug':>6} {'R_feat':>7} {'R_ques':>7} | {'P_bug':>6} {'P_feat':>7} {'P_ques':>7}\")
print('-' * 145)

configs = [
    ('Llama-3B', '3k', 'k3', 'unsloth_Llama_3_2_3B_Instruct', 'unsloth/Llama-3.2-3B-Instruct'),
    ('Llama-8B', '3k', 'k9', 'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit', 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit'),
    ('Llama-3B', '30k', 'k3', 'unsloth_Llama_3_2_3B_Instruct', 'unsloth/Llama-3.2-3B-Instruct'),
    ('Llama-8B', '30k', 'k9', 'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit', 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit'),
]

for name, ds, k_label, tag, model_full in configs:
    # Baseline
    if ds == '3k':
        try:
            agg = pd.read_csv('results/issues3k_ctx8192/all_results.csv')
            k_val = int(k_label.replace('k',''))
            row = agg[(agg['model'] == model_full) & (agg['top_k'] == k_val) & (agg['approach'] == 'ragtag')]
            if len(row):
                r = row.iloc[0]
                print(f\"  {name:>10} {ds:>4} {'baseline':>10} {r['f1_macro']:>8.4f} {r['accuracy']:>7.4f} {r['invalid_rate']:>6.1%} | {r['f1_bug']:>7.3f} {r['f1_feature']:>7.3f} {r['f1_question']:>8.3f} | {r['recall_bug']:>6.3f} {r['recall_feature']:>7.3f} {r['recall_question']:>7.3f} | {r['precision_bug']:>6.3f} {r['precision_feature']:>7.3f} {r['precision_question']:>7.3f}\")
        except: pass
    else:
        base_path = f'results/issues30k/{tag}/ragtag/predictions/preds_{k_label}.csv'
        if not os.path.exists(base_path):
            base_path = f'results/issues30k_k_study/{tag}/ragtag/predictions/preds_{k_label}.csv'
        if os.path.exists(base_path):
            f1m, acc, inv, pc = eval_f1(base_path)
            print(f\"  {name:>10} {ds:>4} {'baseline':>10} {f1m:>8.4f} {acc:>7.4f} {inv:>6.1%} | {pc['bug']['F1']:>7.3f} {pc['feature']['F1']:>7.3f} {pc['question']['F1']:>8.3f} | {pc['bug']['R']:>6.3f} {pc['feature']['R']:>7.3f} {pc['question']['R']:>7.3f} | {pc['bug']['P']:>6.3f} {pc['feature']['P']:>7.3f} {pc['question']['P']:>7.3f}\")

    # Debiased margins
    for margin in [3, 2]:
        debias_path = f'results/issues{ds}_debias_m{margin}/{tag}/ragtag/predictions/preds_{k_label}.csv'
        if os.path.exists(debias_path):
            f1m, acc, inv, pc = eval_f1(debias_path)
            print(f\"  {name:>10} {ds:>4} {'debias_m'+str(margin):>10} {f1m:>8.4f} {acc:>7.4f} {inv:>6.1%} | {pc['bug']['F1']:>7.3f} {pc['feature']['F1']:>7.3f} {pc['question']['F1']:>8.3f} | {pc['bug']['R']:>6.3f} {pc['feature']['R']:>7.3f} {pc['question']['R']:>7.3f} | {pc['bug']['P']:>6.3f} {pc['feature']['P']:>7.3f} {pc['question']['P']:>7.3f}\")

    # FT
    if ds == '30k':
        ft_path = f'results/issues30k/{tag}/finetune_fixed/preds_finetune_fixed.csv'
        if os.path.exists(ft_path):
            f1m, acc, inv, pc = eval_f1(ft_path)
            print(f\"  {name:>10} {ds:>4} {'FT':>10} {f1m:>8.4f} {acc:>7.4f} {inv:>6.1%} | {pc['bug']['F1']:>7.3f} {pc['feature']['F1']:>7.3f} {pc['question']['F1']:>8.3f} | {pc['bug']['R']:>6.3f} {pc['feature']['R']:>7.3f} {pc['question']['R']:>7.3f} | {pc['bug']['P']:>6.3f} {pc['feature']['P']:>7.3f} {pc['question']['P']:>7.3f}\")
    print()
"

echo -e "\nDone!"

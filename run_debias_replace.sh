#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_debias_replace.sh — Test debiased retrieval WITH replacement
# ============================================================================
# Instead of just removing bug neighbors, backfill with non-bug neighbors
# from a larger retrieval pool (k=16 from FAISS, then filter to k=3 or k=9).
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON_BIN:-venv/bin/python}"
MAX_NEW_TOKENS=50
BATCH_SIZE=1
CTX=8192
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

MODEL_3B="unsloth/Llama-3.2-3B-Instruct"
TAG_3B="unsloth_Llama_3_2_3B_Instruct"

# ============================================================
# Step 0: Ensure k=16 neighbor files exist for both datasets
# ============================================================
NB_DIR_3K="results/issues3k_debias/neighbors"
NB_DIR_30K="results/issues30k/neighbors"

# 3k: rebuild with k=16 if needed
if [ -f "${NB_DIR_3K}/neighbors_k16.csv" ]; then
    echo "[SKIP] 3k neighbors_k16.csv already exists"
else
    echo "[0a] Building 3k FAISS index + neighbors (k=3,9,16)..."
    $PYTHON build_and_query_index.py \
        --dataset issues3k.csv \
        --output_dir "$NB_DIR_3K" \
        --top_ks "3,9,16" \
        --test_size 0.5 \
        --embedding_model "$EMBEDDING_MODEL"
fi

# 30k: should already have k=16
if [ ! -f "${NB_DIR_30K}/neighbors_k16.csv" ]; then
    echo "[ERROR] 30k neighbors_k16.csv not found. Run the 30k index build first."
    exit 1
fi

# ============================================================
# Run margin=3 with replacement (backfill from k=16 pool)
# ============================================================
DEBIAS_FLAGS="--debias_retrieval --debias_margin 3"
SUFFIX="m3_replace"

# --- 3k: Llama-3B (k=3) ---
PRED="results/issues3k_debias_${SUFFIX}/${TAG_3B}/ragtag/predictions"
echo -e "\n  Llama-3B on 3k, k=3, mode=replace_m3"
$PYTHON llm_labeler.py \
    --model "$MODEL_3B" \
    --neighbors_dir "$NB_DIR_3K" \
    --output_dir "$PRED" \
    --top_ks "3" \
    --max_seq_length "$CTX" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --inference_batch_size "$BATCH_SIZE" \
    --load_in_4bit \
    $DEBIAS_FLAGS

EVAL="results/issues3k_debias_${SUFFIX}/${TAG_3B}/ragtag/evaluations"
mkdir -p "$EVAL"
for PRED_FILE in "$PRED"/preds_*.csv; do
    [ -f "$PRED_FILE" ] || continue
    BASENAME=$(basename "$PRED_FILE" .csv)
    K_TAG="${BASENAME#preds_}"
    $PYTHON evaluate.py --preds_csv "$PRED_FILE" --output_csv "${EVAL}/eval_${K_TAG}.csv"
done

# --- 30k: Llama-3B (k=3) ---
PRED="results/issues30k_debias_${SUFFIX}/${TAG_3B}/ragtag/predictions"
echo -e "\n  Llama-3B on 30k, k=3, mode=replace_m3"
$PYTHON llm_labeler.py \
    --model "$MODEL_3B" \
    --neighbors_dir "$NB_DIR_30K" \
    --output_dir "$PRED" \
    --top_ks "3" \
    --max_seq_length "$CTX" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --inference_batch_size "$BATCH_SIZE" \
    --load_in_4bit \
    $DEBIAS_FLAGS

EVAL="results/issues30k_debias_${SUFFIX}/${TAG_3B}/ragtag/evaluations"
mkdir -p "$EVAL"
for PRED_FILE in "$PRED"/preds_*.csv; do
    [ -f "$PRED_FILE" ] || continue
    BASENAME=$(basename "$PRED_FILE" .csv)
    K_TAG="${BASENAME#preds_}"
    $PYTHON evaluate.py --preds_csv "$PRED_FILE" --output_csv "${EVAL}/eval_${K_TAG}.csv"
done

# ============================================================
# Summary
# ============================================================
echo -e "\n============================================================"
echo "  Debiased Retrieval with Replacement — Llama-3B Results"
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

print(f\"{'Data':>4} {'Type':>14} {'F1_macro':>8} {'Acc':>7} {'Inv%':>6} | {'F1_bug':>7} {'F1_feat':>7} {'F1_ques':>8} | {'R_bug':>6} {'R_feat':>7} {'R_ques':>7} | {'P_bug':>6} {'P_feat':>7} {'P_ques':>7}\")
print('-' * 130)

tag = 'unsloth_Llama_3_2_3B_Instruct'
model_full = 'unsloth/Llama-3.2-3B-Instruct'

for ds in ['3k', '30k']:
    # Baseline
    if ds == '3k':
        try:
            agg = pd.read_csv('results/issues3k_ctx8192/all_results.csv')
            row = agg[(agg['model'] == model_full) & (agg['top_k'] == 3) & (agg['approach'] == 'ragtag')]
            if len(row):
                r = row.iloc[0]
                print(f\"  {ds:>4} {'baseline':>14} {r['f1_macro']:>8.4f} {r['accuracy']:>7.4f} {r['invalid_rate']:>6.1%} | {r['f1_bug']:>7.3f} {r['f1_feature']:>7.3f} {r['f1_question']:>8.3f} | {r['recall_bug']:>6.3f} {r['recall_feature']:>7.3f} {r['recall_question']:>7.3f} | {r['precision_bug']:>6.3f} {r['precision_feature']:>7.3f} {r['precision_question']:>7.3f}\")
        except: pass
    else:
        for bp in [f'results/issues30k/{tag}/ragtag/predictions/preds_k3.csv', f'results/issues30k_k_study/{tag}/ragtag/predictions/preds_k3.csv']:
            if os.path.exists(bp):
                f1m, acc, inv, pc = eval_f1(bp)
                print(f\"  {ds:>4} {'baseline':>14} {f1m:>8.4f} {acc:>7.4f} {inv:>6.1%} | {pc['bug']['F1']:>7.3f} {pc['feature']['F1']:>7.3f} {pc['question']['F1']:>8.3f} | {pc['bug']['R']:>6.3f} {pc['feature']['R']:>7.3f} {pc['question']['R']:>7.3f} | {pc['bug']['P']:>6.3f} {pc['feature']['P']:>7.3f} {pc['question']['P']:>7.3f}\")
                break

    # Remove m3
    path = f'results/issues{ds}_debias_m3/{tag}/ragtag/predictions/preds_k3.csv'
    if os.path.exists(path):
        f1m, acc, inv, pc = eval_f1(path)
        print(f\"  {ds:>4} {'remove_m3':>14} {f1m:>8.4f} {acc:>7.4f} {inv:>6.1%} | {pc['bug']['F1']:>7.3f} {pc['feature']['F1']:>7.3f} {pc['question']['F1']:>8.3f} | {pc['bug']['R']:>6.3f} {pc['feature']['R']:>7.3f} {pc['question']['R']:>7.3f} | {pc['bug']['P']:>6.3f} {pc['feature']['P']:>7.3f} {pc['question']['P']:>7.3f}\")

    # Replace m3
    path = f'results/issues{ds}_debias_m3_replace/{tag}/ragtag/predictions/preds_k3.csv'
    if os.path.exists(path):
        f1m, acc, inv, pc = eval_f1(path)
        print(f\"  {ds:>4} {'replace_m3':>14} {f1m:>8.4f} {acc:>7.4f} {inv:>6.1%} | {pc['bug']['F1']:>7.3f} {pc['feature']['F1']:>7.3f} {pc['question']['F1']:>8.3f} | {pc['bug']['R']:>6.3f} {pc['feature']['R']:>7.3f} {pc['question']['R']:>7.3f} | {pc['bug']['P']:>6.3f} {pc['feature']['P']:>7.3f} {pc['question']['P']:>7.3f}\")

    # FT
    if ds == '30k':
        ft_path = f'results/issues30k/{tag}/finetune_fixed/preds_finetune_fixed.csv'
        if os.path.exists(ft_path):
            f1m, acc, inv, pc = eval_f1(ft_path)
            print(f\"  {ds:>4} {'FT':>14} {f1m:>8.4f} {acc:>7.4f} {inv:>6.1%} | {pc['bug']['F1']:>7.3f} {pc['feature']['F1']:>7.3f} {pc['question']['F1']:>8.3f} | {pc['bug']['R']:>6.3f} {pc['feature']['R']:>7.3f} {pc['question']['R']:>7.3f} | {pc['bug']['P']:>6.3f} {pc['feature']['P']:>7.3f} {pc['question']['P']:>7.3f}\")
    print()
"

echo -e "\nDone!"

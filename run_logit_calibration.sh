#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_logit_calibration.sh — Test BC, CD, and BC+CD on Llama-3B and Llama-8B
# ============================================================================
# Phase 2 logit-level interventions for correcting parametric bug bias.
#
# Round 1: BC on 3k (fast baseline)
# Round 2: CD on 3k with alpha sweep (0.5, 0.75, 1.0)
# Round 3: Best method on 30k
# Round 4: BC+CD combined on 30k
# Round 5: Stacked with Phase 1 debiased retrieval on 30k
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON_BIN:-venv/bin/python}"
CTX=8192
BATCH_SIZE=4

MODEL_3B="unsloth/Llama-3.2-3B-Instruct"
MODEL_8B="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
TAG_3B="unsloth_Llama_3_2_3B_Instruct"
TAG_8B="unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit"

NB_DIR_3K="results/issues3k_debias/neighbors"
NB_DIR_30K="results/issues30k/neighbors"

# Helper: run logit calibration + evaluate
run_logit() {
    local MODEL="$1" TAG="$2" NB_DIR="$3" DS="$4" K="$5" METHOD="$6"
    local ALPHA="${7:-1.0}" DEBIAS="${8:-}"

    local SUFFIX="${METHOD}"
    if [[ "$METHOD" == "cd" || "$METHOD" == "bc_cd" ]]; then
        SUFFIX="${METHOD}_a${ALPHA}"
    fi
    if [[ -n "$DEBIAS" ]]; then
        SUFFIX="${SUFFIX}_debias_m${DEBIAS}"
    fi

    local PRED="results/issues${DS}_logit_${SUFFIX}/${TAG}/ragtag/predictions"
    local EVAL="results/issues${DS}_logit_${SUFFIX}/${TAG}/ragtag/evaluations"

    # Build command
    local CMD=("$PYTHON" logit_calibration.py \
        --model "$MODEL" \
        --neighbors_dir "$NB_DIR" \
        --output_dir "$PRED" \
        --top_ks "$K" \
        --method "$METHOD" \
        --cd_alpha "$ALPHA" \
        --max_seq_length "$CTX" \
        --inference_batch_size "$BATCH_SIZE" \
        --load_in_4bit \
        --eval_dir "$EVAL" \
        --save_logits)

    if [[ -n "$DEBIAS" ]]; then
        CMD+=(--debias_retrieval --debias_margin "$DEBIAS")
    fi

    echo -e "\n  $(echo "$MODEL" | sed 's|.*/||') on ${DS}, k=${K}, method=${SUFFIX}"
    "${CMD[@]}"

    # Evaluate any prediction files
    mkdir -p "$EVAL"
    for PRED_FILE in "$PRED"/preds_*.csv; do
        [ -f "$PRED_FILE" ] || continue
        BASENAME=$(basename "$PRED_FILE" .csv)
        K_TAG="${BASENAME#preds_}"
        $PYTHON evaluate.py --preds_csv "$PRED_FILE" --output_csv "${EVAL}/eval_${K_TAG}.csv"
    done
}

# ============================================================================
# Round 1: Batch Calibration on 3k
# ============================================================================
echo -e "\n============================================================"
echo "  ROUND 1: Batch Calibration (BC) on 3k"
echo "============================================================"

run_logit "$MODEL_3B" "$TAG_3B" "$NB_DIR_3K" "3k" "3" "bc"
run_logit "$MODEL_8B" "$TAG_8B" "$NB_DIR_3K" "3k" "9" "bc"

# ============================================================================
# Round 2: Contrastive Decoding on 3k (alpha sweep)
# ============================================================================
echo -e "\n============================================================"
echo "  ROUND 2: Contrastive Decoding (CD) on 3k — alpha sweep"
echo "============================================================"

for ALPHA in 1.0 0.5 0.75; do
    echo -e "\n--- alpha = ${ALPHA} ---"
    run_logit "$MODEL_3B" "$TAG_3B" "$NB_DIR_3K" "3k" "3" "cd" "$ALPHA"
    run_logit "$MODEL_8B" "$TAG_8B" "$NB_DIR_3K" "3k" "9" "cd" "$ALPHA"
done

# ============================================================================
# Round 3: Best methods on 30k
# ============================================================================
echo -e "\n============================================================"
echo "  ROUND 3: BC and CD (alpha=1.0) on 30k"
echo "============================================================"

run_logit "$MODEL_3B" "$TAG_3B" "$NB_DIR_30K" "30k" "3" "bc"
run_logit "$MODEL_8B" "$TAG_8B" "$NB_DIR_30K" "30k" "9" "bc"
run_logit "$MODEL_3B" "$TAG_3B" "$NB_DIR_30K" "30k" "3" "cd" "1.0"
run_logit "$MODEL_8B" "$TAG_8B" "$NB_DIR_30K" "30k" "9" "cd" "1.0"

# ============================================================================
# Round 4: BC+CD combined on 30k
# ============================================================================
echo -e "\n============================================================"
echo "  ROUND 4: BC+CD on 30k (alpha=1.0)"
echo "============================================================"

run_logit "$MODEL_3B" "$TAG_3B" "$NB_DIR_30K" "30k" "3" "bc_cd" "1.0"
run_logit "$MODEL_8B" "$TAG_8B" "$NB_DIR_30K" "30k" "9" "bc_cd" "1.0"

# ============================================================================
# Round 5: Stacked with Phase 1 debiased retrieval on 30k
# ============================================================================
echo -e "\n============================================================"
echo "  ROUND 5: Best logit method + debiased retrieval (m=3) on 30k"
echo "============================================================"

run_logit "$MODEL_3B" "$TAG_3B" "$NB_DIR_30K" "30k" "3" "bc" "1.0" "3"
run_logit "$MODEL_8B" "$TAG_8B" "$NB_DIR_30K" "30k" "9" "bc" "1.0" "3"
run_logit "$MODEL_3B" "$TAG_3B" "$NB_DIR_30K" "30k" "3" "cd" "1.0" "3"
run_logit "$MODEL_8B" "$TAG_8B" "$NB_DIR_30K" "30k" "9" "cd" "1.0" "3"
run_logit "$MODEL_3B" "$TAG_3B" "$NB_DIR_30K" "30k" "3" "bc_cd" "1.0" "3"
run_logit "$MODEL_8B" "$TAG_8B" "$NB_DIR_30K" "30k" "9" "bc_cd" "1.0" "3"

# ============================================================================
# Summary
# ============================================================================
echo -e "\n============================================================"
echo "  Logit Calibration Results Summary"
echo "============================================================"
$PYTHON -c "
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os, glob

LABELS = ['bug', 'feature', 'question']

def eval_f1(path):
    df = pd.read_csv(path)
    y_true = df['ground_truth'].astype(str).str.lower().str.strip()
    y_pred = df['predicted_label'].astype(str).str.lower().str.strip()
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    _, _, f1m, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    inv = (~y_pred.isin(LABELS)).mean()
    return f1m, acc, inv, {l: {'P': p[i], 'R': r[i], 'F1': f[i]} for i, l in enumerate(LABELS)}

header = f\"{'Model':>12} {'Data':>4} {'Type':>18} {'F1_macro':>8} {'Acc':>7} {'Inv%':>6} | {'F1_bug':>7} {'F1_feat':>7} {'F1_ques':>8} | {'R_bug':>6} {'R_feat':>7} {'R_ques':>7} | {'P_bug':>6} {'P_feat':>7} {'P_ques':>7}\"
print(header)
print('-' * 155)

configs = [
    ('Llama-3B', '3k', 'k3', '$TAG_3B'),
    ('Llama-8B', '3k', 'k9', '$TAG_8B'),
    ('Llama-3B', '30k', 'k3', '$TAG_3B'),
    ('Llama-8B', '30k', 'k9', '$TAG_8B'),
]

def print_row(name, ds, rtype, f1m, acc, inv, pc):
    print(f\"  {name:>10} {ds:>4} {rtype:>18} {f1m:>8.4f} {acc:>7.4f} {inv:>6.1%} | {pc['bug']['F1']:>7.3f} {pc['feature']['F1']:>7.3f} {pc['question']['F1']:>8.3f} | {pc['bug']['R']:>6.3f} {pc['feature']['R']:>7.3f} {pc['question']['R']:>7.3f} | {pc['bug']['P']:>6.3f} {pc['feature']['P']:>7.3f} {pc['question']['P']:>7.3f}\")

for name, ds, k_label, tag in configs:
    # Baseline (from debias results or 30k k-study)
    base_paths = [
        f'results/issues{ds}_debias_m3/{tag}/ragtag/predictions/preds_{k_label}.csv',
        f'results/issues{ds}_ctx8192/{tag}/ragtag/predictions/preds_{k_label}.csv',
        f'results/issues{ds}/{tag}/ragtag/predictions/preds_{k_label}.csv',
        f'results/issues{ds}_k_study/{tag}/ragtag/predictions/preds_{k_label}.csv',
    ]
    for bp in base_paths:
        if os.path.exists(bp):
            try:
                f1m, acc, inv, pc = eval_f1(bp)
                print_row(name, ds, 'baseline', f1m, acc, inv, pc)
            except: pass
            break

    # Debias m3
    debias_path = f'results/issues{ds}_debias_m3/{tag}/ragtag/predictions/preds_{k_label}.csv'
    if os.path.exists(debias_path):
        try:
            f1m, acc, inv, pc = eval_f1(debias_path)
            print_row(name, ds, 'debias_m3', f1m, acc, inv, pc)
        except: pass

    # Logit calibration results
    for pattern in glob.glob(f'results/issues{ds}_logit_*/{tag}/ragtag/predictions/preds_{k_label}_*.csv'):
        try:
            method_dir = pattern.split(f'issues{ds}_logit_')[1].split('/')[0]
            f1m, acc, inv, pc = eval_f1(pattern)
            print_row(name, ds, f'logit_{method_dir}', f1m, acc, inv, pc)
        except: pass

    # Fine-tune
    if ds == '30k':
        ft_path = f'results/issues30k/{tag}/finetune_fixed/preds_finetune_fixed.csv'
        if os.path.exists(ft_path):
            try:
                f1m, acc, inv, pc = eval_f1(ft_path)
                print_row(name, ds, 'FT', f1m, acc, inv, pc)
            except: pass
    print()
"

echo -e "\nDone!"

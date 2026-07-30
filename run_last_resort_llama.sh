#!/usr/bin/env bash
# =============================================================================
# run_last_resort_llama.sh
#
# Fresh, self-contained experiment run for two Llama models that are NOT in the
# paper's active lineup. All outputs land in a separate folder so the
# paper-archival data under results/issues11k/ is left untouched.
#
# Experiments (for BOTH models):
#   A) Full RAGTAG, project-specific (PS)              k = {0,1,3,6,9,12,15}
#   B) Debiased RAGTAG, project-specific (PS), margin 3 k = {0,1,3,6,9,12,15}
#   C) Fine-tune, project-agnostic (PA), 3 epochs
#
# Every preds_*.csv produced gets a matching eval_*.csv (evaluate.py), exactly
# like run_k12_k15_local_8k.sh and run_11k_experiments.sh.
#
# Run:
#   bash run_last_resort_llama.sh 2>&1 | tee last_resort_llama_run.log
# =============================================================================

set -uo pipefail   # NOT -e: a failed cell is logged, the run continues.

# --- environment -------------------------------------------------------------
cd "$(dirname "$0")"
source venv/bin/activate
export PYTHONUNBUFFERED=1

# --- constants ---------------------------------------------------------------
RESULTS_SRC="results/issues11k"
OUT_ROOT="results/last_resort_llama"
LOG_DIR="$OUT_ROOT/logs"
DATASET="issues11k.csv"

CTX_RAGTAG=8192
CTX_FT=2048
KS="0,1,3,6,9,12,15"
MARGIN=3

mkdir -p "$LOG_DIR"

# model "hf_id|tag" pairs (tag = hf_id with / . - replaced by _)
MODELS=(
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit|unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit"
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit|unsloth_Llama_3_2_3B_Instruct_bnb_4bit"
)

PROJECTS=(
    ansible_ansible bitcoin_bitcoin dart-lang_sdk dotnet_roslyn
    facebook_react flutter_flutter kubernetes_kubernetes microsoft_TypeScript
    microsoft_vscode opencv_opencv tensorflow_tensorflow
)

# k labels used in preds filenames, for idempotent skip checks
PRED_FILES=(preds_zero_shot.csv preds_k1.csv preds_k3.csv preds_k6.csv \
            preds_k9.csv preds_k12.csv preds_k15.csv)

banner() { echo; echo "=============================================================="; \
           echo ">>> $*"; echo "=============================================================="; }

# returns 0 (true) if every expected RAGTAG preds file already exists
ragtag_done() {
    local dir="$1" f
    for f in "${PRED_FILES[@]}"; do
        [[ -f "$dir/$f" ]] || return 1
    done
    return 0
}

START_TS=$(date +%s)
banner "LAST RESORT LLAMA — start $(date)"

for ENTRY in "${MODELS[@]}"; do
    MODEL="${ENTRY%%|*}"
    TAG="${ENTRY##*|}"
    banner "MODEL: $MODEL  (tag: $TAG)"

    # ----------------------------------------------------------------------
    # Phase A — RAGTAG, project-specific
    # ----------------------------------------------------------------------
    for PROJ in "${PROJECTS[@]}"; do
        PRED_DIR="$OUT_ROOT/project_specific/$PROJ/$TAG/ragtag/predictions"
        EVAL_DIR="$OUT_ROOT/project_specific/$PROJ/$TAG/ragtag/evaluations"
        if ragtag_done "$PRED_DIR"; then
            echo "[skip] RAGTAG PS  $TAG / $PROJ  (predictions present)"
            continue
        fi
        mkdir -p "$PRED_DIR" "$EVAL_DIR"
        banner "Phase A — RAGTAG PS  |  $TAG  |  $PROJ"
        python llm_labeler.py \
            --model "$MODEL" \
            --neighbors_dir "$RESULTS_SRC/project_specific/$PROJ/neighbors" \
            --top_ks "$KS" \
            --output_dir "$PRED_DIR" \
            --eval_dir "$EVAL_DIR" \
            --max_seq_length "$CTX_RAGTAG" \
            --max_new_tokens 50 \
            --inference_batch_size 1 \
            --model_name_for_eval "$TAG" \
            2>&1 | tee "$LOG_DIR/${TAG}_ragtag_${PROJ}.log"
    done

    # ----------------------------------------------------------------------
    # Phase B — Debiased RAGTAG, project-specific, margin 3
    # ----------------------------------------------------------------------
    for PROJ in "${PROJECTS[@]}"; do
        PRED_DIR="$OUT_ROOT/project_specific/$PROJ/$TAG/ragtag_debias_m3/predictions"
        EVAL_DIR="$OUT_ROOT/project_specific/$PROJ/$TAG/ragtag_debias_m3/evaluations"
        if ragtag_done "$PRED_DIR"; then
            echo "[skip] DEBIAS PS  $TAG / $PROJ  (predictions present)"
            continue
        fi
        mkdir -p "$PRED_DIR" "$EVAL_DIR"
        banner "Phase B — Debiased RAGTAG PS (m=$MARGIN)  |  $TAG  |  $PROJ"
        python llm_labeler.py \
            --model "$MODEL" \
            --neighbors_dir "$RESULTS_SRC/project_specific/$PROJ/neighbors" \
            --top_ks "$KS" \
            --output_dir "$PRED_DIR" \
            --eval_dir "$EVAL_DIR" \
            --max_seq_length "$CTX_RAGTAG" \
            --max_new_tokens 50 \
            --inference_batch_size 1 \
            --model_name_for_eval "$TAG" \
            --debias_retrieval \
            --debias_margin "$MARGIN" \
            2>&1 | tee "$LOG_DIR/${TAG}_debias_${PROJ}.log"
    done

    # ----------------------------------------------------------------------
    # Phase C — Fine-tune, project-agnostic, 3 epochs
    #   (num_train_epochs=3 is hard-coded in fixed_fine-tune.py)
    # ----------------------------------------------------------------------
    FT_DIR="$OUT_ROOT/agnostic/$TAG/finetune_fixed"
    if [[ -f "$FT_DIR/preds_finetune_fixed.csv" ]]; then
        echo "[skip] FINETUNE PA  $TAG  (predictions present)"
    else
        mkdir -p "$FT_DIR"
        banner "Phase C — Fine-tune PA (3 epochs)  |  $TAG"
        python fixed_fine-tune.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --train_csv "$RESULTS_SRC/agnostic/neighbors/train_split.csv" \
            --test_csv "$RESULTS_SRC/agnostic/neighbors/test_split.csv" \
            --max_seq_length "$CTX_FT" \
            --max_new_tokens 50 \
            --inference_batch_size 1 \
            --output_dir "$FT_DIR" \
            2>&1 | tee "$LOG_DIR/${TAG}_finetune.log"
    fi

    # explicit evaluation of the fine-tune predictions (mirrors run_11k Phase 5)
    if [[ -f "$FT_DIR/preds_finetune_fixed.csv" ]]; then
        banner "Phase C — evaluate fine-tune  |  $TAG"
        python evaluate.py \
            --preds_csv "$FT_DIR/preds_finetune_fixed.csv" \
            --output_csv "$FT_DIR/eval_finetune_fixed.csv" \
            --model_name "$TAG" \
            2>&1 | tee -a "$LOG_DIR/${TAG}_finetune.log"
    else
        echo "[warn] $FT_DIR/preds_finetune_fixed.csv missing — skipping FT eval"
    fi
done

ELAPSED=$(( $(date +%s) - START_TS ))
banner "LAST RESORT LLAMA — done in $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m"
echo "Results: $OUT_ROOT/"
echo "Logs:    $LOG_DIR/"

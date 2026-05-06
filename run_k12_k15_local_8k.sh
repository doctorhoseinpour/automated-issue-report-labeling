#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_k12_k15_local_8k.sh — Extend k-grid to {12, 15} at ctx=8192
# ============================================================================
# Runs RAGTAG (PA + PS) and Debiased RAGTAG (PS only, margin=3) at k=12 and
# k=15 for Qwen-3B, Qwen-7B, Qwen-14B.
# Qwen-32B is run on NRP. 16K context runs are in run_k12_k15_local_16k.sh.
#
# Per cell, llm_labeler.py runs --top_ks "12,15" so each cell produces both
# preds_k12.csv and preds_k15.csv in one process.
#
# Outputs extend the existing 8K dirs:
#   results/issues11k/agnostic/<TAG>/ragtag/predictions/preds_k{12,15}.csv
#   results/issues11k/project_specific/<P>/<TAG>/ragtag/predictions/preds_k{12,15}.csv
#   results/issues11k/project_specific/<P>/<TAG>/ragtag_debias_m3/predictions/preds_k{12,15}.csv
#
# Idempotent: each cell skips if its preds_k15.csv already exists.
#
# Estimated runtime: ~34 hours total (3B ~5h, 7B ~10h, 14B ~19h).
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate

RESULTS="results/issues11k"
KS="12,15"
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
)

# ----------------------------------------------------------------------------
# Helper: run one llm_labeler invocation with idempotent skip
#   $1 = model HF ID
#   $2 = neighbors_dir
#   $3 = output predictions dir
#   $4 = "ragtag" or "debias" (controls --debias_retrieval flag)
# ----------------------------------------------------------------------------
run_cell() {
    local MODEL="$1"
    local NEIGH_DIR="$2"
    local PRED_DIR="$3"
    local METHOD="$4"

    local SKIP_FILE="$PRED_DIR/preds_k15.csv"
    if [[ -f "$SKIP_FILE" ]]; then
        echo "    SKIP: $SKIP_FILE already exists"
        return 0
    fi

    mkdir -p "$PRED_DIR"
    local EVAL_DIR
    EVAL_DIR="$(dirname "$PRED_DIR")/evaluations"
    mkdir -p "$EVAL_DIR"

    local DEBIAS_FLAGS=""
    if [[ "$METHOD" == "debias" ]]; then
        DEBIAS_FLAGS="--debias_retrieval --debias_margin $MARGIN"
    fi

    python llm_labeler.py \
        --model "$MODEL" \
        --neighbors_dir "$NEIGH_DIR" \
        --top_ks "$KS" \
        --output_dir "$PRED_DIR" \
        --eval_dir "$EVAL_DIR" \
        --max_seq_length "$CTX" \
        --max_new_tokens 50 \
        --inference_batch_size 1 \
        --model_name_for_eval "$(basename "$(dirname "$PRED_DIR")")" \
        $DEBIAS_FLAGS
}

# ----------------------------------------------------------------------------
# Main loop: each model, RAGTAG-PA, RAGTAG-PS x 11, Debias-PS x 11
# ----------------------------------------------------------------------------
START_TS=$(date +%s)
echo "============================================================"
echo "  k=12,15 local 8K campaign — start $(date)"
echo "  Models: 3B, 7B, 14B  (32B is on NRP)"
echo "  Ctx:    $CTX"
echo "  Ks:     $KS"
echo "  Debias: PS only (margin=$MARGIN)"
echo "============================================================"

for MODEL_ENTRY in "${MODELS[@]}"; do
    MODEL="${MODEL_ENTRY%%|*}"
    TAG="${MODEL_ENTRY##*|}"

    echo ""
    echo "############################################################"
    echo "## MODEL: $MODEL"
    echo "## TAG:   $TAG"
    echo "############################################################"

    # ---- RAGTAG project-agnostic ----
    echo ""
    echo "[RAGTAG-PA] $TAG"
    run_cell "$MODEL" \
        "$RESULTS/agnostic/neighbors" \
        "$RESULTS/agnostic/$TAG/ragtag/predictions" \
        "ragtag"

    # ---- RAGTAG project-specific (11 projects) ----
    for PROJ in "${PROJECTS[@]}"; do
        echo ""
        echo "[RAGTAG-PS] $TAG  $PROJ"
        run_cell "$MODEL" \
            "$RESULTS/project_specific/$PROJ/neighbors" \
            "$RESULTS/project_specific/$PROJ/$TAG/ragtag/predictions" \
            "ragtag"
    done

    # ---- Debiased RAGTAG project-specific (11 projects) ----
    for PROJ in "${PROJECTS[@]}"; do
        echo ""
        echo "[Debias-PS] $TAG  $PROJ"
        run_cell "$MODEL" \
            "$RESULTS/project_specific/$PROJ/neighbors" \
            "$RESULTS/project_specific/$PROJ/$TAG/ragtag_debias_m3/predictions" \
            "debias"
    done

    echo ""
    echo "## $MODEL complete  ($(date))"
done

END_TS=$(date +%s)
ELAPSED_MIN=$(( (END_TS - START_TS) / 60 ))
echo ""
echo "============================================================"
echo "  k=12,15 local 8K campaign DONE — $(date)"
echo "  Elapsed: $ELAPSED_MIN min"
echo "============================================================"

echo ""
echo "=== F1 macro summary (new k=12, k=15 cells) ==="
for TAG in unsloth_Qwen2_5_3B_Instruct_bnb_4bit unsloth_Qwen2_5_7B_Instruct_bnb_4bit unsloth_Qwen2_5_14B_Instruct_bnb_4bit; do
    SHORT="${TAG#unsloth_}"
    SHORT="${SHORT%_Instruct_bnb_4bit}"
    for K in 12 15; do
        PA="$RESULTS/agnostic/$TAG/ragtag/evaluations/eval_k${K}.csv"
        if [[ -f "$PA" ]]; then
            f1=$(awk -F, 'NR==2{print $21}' "$PA")
            printf "  %s RAGTAG-PA k=%s f1=%s\n" "$SHORT" "$K" "$f1"
        fi
        for PROJ in "${PROJECTS[@]}"; do
            PS="$RESULTS/project_specific/$PROJ/$TAG/ragtag/evaluations/eval_k${K}.csv"
            if [[ -f "$PS" ]]; then
                f1=$(awk -F, 'NR==2{print $21}' "$PS")
                printf "  %s RAGTAG-PS k=%s %-25s f1=%s\n" "$SHORT" "$K" "$PROJ" "$f1"
            fi
            DB="$RESULTS/project_specific/$PROJ/$TAG/ragtag_debias_m3/evaluations/eval_k${K}.csv"
            if [[ -f "$DB" ]]; then
                f1=$(awk -F, 'NR==2{print $21}' "$DB")
                printf "  %s Debias-PS k=%s %-25s f1=%s\n" "$SHORT" "$K" "$PROJ" "$f1"
            fi
        done
    done
done

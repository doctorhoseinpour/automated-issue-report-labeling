#!/usr/bin/env bash
# =============================================================================
# run_encoder_baselines.sh — encoder baselines for the 11k benchmark
# =============================================================================
# Runs the three literature-mined cheap baselines demanded by ESEM Review #126C
# (see docs/SANER_REVISION_PLAN.md P0.1) in both PA and PS settings:
#
#   1. SetFit  body=sentence-transformers/all-mpnet-base-v2   (NLBSE'24 baseline)
#   2. SetFit  body=Collab-uniba/github-issues-mpnet-st-e10   (domain-adapted)
#   3. RoBERTa-base fine-tune (Colavito JSS 2026 recipe)
#
# Configs (single seed 42, per literature):
#   SetFit:  batch 16, num_epochs 1, num_iterations 20, LogisticRegression head
#   RoBERTa: lr 2e-5, batch 16, weight_decay 0.01, 15 epochs, warmup 0.1, seq 512
#
# Idempotent: each runner skips if its preds CSV already exists.
#
# Usage:
#   bash run_encoder_baselines.sh                # everything
#   bash run_encoder_baselines.sh --pa_only      # PA cells only (verification)
#   bash run_encoder_baselines.sh --skip_setfit | --skip_roberta
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_MAIN="${PYTHON_BIN:-venv/bin/python}"
PYTHON_SETFIT="${PYTHON_SETFIT_BIN:-venv-setfit/bin/python}"
RESULTS="results/issues11k"
SEED=42

SETFIT_BODIES=(
  "sentence-transformers/all-mpnet-base-v2"
  "Collab-uniba/github-issues-mpnet-st-e10"
)

PROJECT_TAGS=(
  ansible_ansible bitcoin_bitcoin dart-lang_sdk dotnet_roslyn facebook_react
  flutter_flutter kubernetes_kubernetes microsoft_TypeScript microsoft_vscode
  opencv_opencv tensorflow_tensorflow
)

PA_ONLY=false
SKIP_SETFIT=false
SKIP_ROBERTA=false
for arg in "$@"; do
  case "$arg" in
    --pa_only) PA_ONLY=true ;;
    --skip_setfit) SKIP_SETFIT=true ;;
    --skip_roberta) SKIP_ROBERTA=true ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

banner() {
  echo ""
  echo "============================================================"
  echo "  $1"
  echo "============================================================"
}

run_setfit_cell() {  # $1=body_model $2=train_csv $3=test_csv $4=output_dir
  "$PYTHON_SETFIT" run_setfit.py \
    --body_model "$1" \
    --train_csv "$2" \
    --test_csv "$3" \
    --output_dir "$4" \
    --batch_size 16 --num_epochs 1 --num_iterations 20 --seed "$SEED"
}

run_roberta_cell() {  # $1=train_csv $2=test_csv $3=output_dir
  "$PYTHON_MAIN" run_transformer_ft.py \
    --model roberta-base \
    --train_csv "$1" \
    --test_csv "$2" \
    --output_dir "$3" \
    --epochs 15 --batch_size 16 --lr 2e-5 --weight_decay 0.01 \
    --max_seq_length 512 --warmup_ratio 0.1 --seed "$SEED"
}

PA_TRAIN="$RESULTS/agnostic/neighbors/train_split.csv"
PA_TEST="$RESULTS/agnostic/neighbors/test_split.csv"

# ---------------------------------------------------------------- SetFit
if ! $SKIP_SETFIT; then
  for BODY in "${SETFIT_BODIES[@]}"; do
    TAG="${BODY//\//_}"
    banner "SetFit ($BODY) — PA"
    run_setfit_cell "$BODY" "$PA_TRAIN" "$PA_TEST" "$RESULTS/agnostic/$TAG/setfit"
    if ! $PA_ONLY; then
      for P in "${PROJECT_TAGS[@]}"; do
        banner "SetFit ($BODY) — PS $P"
        run_setfit_cell "$BODY" \
          "$RESULTS/project_specific/$P/neighbors/train_split.csv" \
          "$RESULTS/project_specific/$P/neighbors/test_split.csv" \
          "$RESULTS/project_specific/$P/$TAG/setfit"
      done
    fi
  done
fi

# ---------------------------------------------------------------- RoBERTa
if ! $SKIP_ROBERTA; then
  banner "RoBERTa-base — PA"
  run_roberta_cell "$PA_TRAIN" "$PA_TEST" "$RESULTS/agnostic/roberta-base/finetune_transformer"
  if ! $PA_ONLY; then
    for P in "${PROJECT_TAGS[@]}"; do
      banner "RoBERTa-base — PS $P"
      run_roberta_cell \
        "$RESULTS/project_specific/$P/neighbors/train_split.csv" \
        "$RESULTS/project_specific/$P/neighbors/test_split.csv" \
        "$RESULTS/project_specific/$P/roberta-base/finetune_transformer"
    done
  fi
fi

banner "Encoder baselines complete"

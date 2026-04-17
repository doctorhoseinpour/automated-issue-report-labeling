#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_nrp.sh — Runs on NRP (A6000/A100)
# ============================================================================
# Only the two fine-tune jobs that don't fit on the local 4090:
#   - Qwen2.5-14B fixed fine-tune on issues30k
#   - Qwen2.5-32B fixed fine-tune on issues30k
#
# Prereq: run_local.sh must have completed Phase 2 first (builds FAISS index,
#          creates train/test splits at results/issues30k/neighbors/).
#
# Usage:  nohup bash run_nrp.sh --nrp 2>&1 | tee nrp_experiment.log &
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

NRP_MODE=0
HF_CACHE_DIR="$SCRIPT_DIR/hf_cache"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nrp) NRP_MODE=1; shift ;;
    -h|--help)
      echo "Usage: run_nrp.sh [--nrp]"
      exit 0 ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

CACHE_ARGS=()
if [[ "$NRP_MODE" -eq 1 ]]; then
  echo ">>> NRP mode: HF cache → $HF_CACHE_DIR"
  CACHE_ARGS=(--cache_dir "$HF_CACHE_DIR")
fi

clear_hf_cache() {
  if [[ -d "$HF_CACHE_DIR" ]]; then
    echo ">>> Clearing HF cache at $HF_CACHE_DIR ..."
    rm -rf "$HF_CACHE_DIR"
    echo "  Done."
  fi
}

# Models that need NRP for fine-tuning
NRP_FT_MODELS=(
  "unsloth/Qwen2.5-14B-Instruct-bnb-4bit|Qwen-14B"
  "unsloth/Qwen2.5-32B-Instruct-bnb-4bit|Qwen-32B"
)

DATASET_30K="$SCRIPT_DIR/issues30k.csv"
TEST_SIZE_30K=3000
RESULTS_30K="$SCRIPT_DIR/results/issues30k"
TRAIN_30K="$RESULTS_30K/neighbors/train_split.csv"
TEST_30K="$RESULTS_30K/neighbors/test_split.csv"
MAX_NEW_TOKENS=50
INFERENCE_BATCH_SIZE=1

# --- Verify prereqs ---
if [[ ! -f "$DATASET_30K" ]]; then
  echo "ERROR: issues30k.csv not found at $DATASET_30K" >&2
  exit 1
fi

EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
NEIGHBORS_30K="$RESULTS_30K/neighbors"

if [[ ! -f "$TRAIN_30K" || ! -f "$TEST_30K" ]]; then
  echo ">>> Train/test splits not found. Building FAISS index for issues30k..."
  mkdir -p "$NEIGHBORS_30K"

  # Only need k values for fine-tune (no RAGTAG neighbors needed here),
  # but build with a reasonable k so the splits are compatible
  "$PYTHON_BIN" "$SCRIPT_DIR/build_and_query_index.py" \
    --dataset "$DATASET_30K" \
    --top_ks "3" \
    --test_size "$TEST_SIZE_30K" \
    --embedding_model "$EMBED_MODEL" \
    --output_dir "$NEIGHBORS_30K"

  echo "  Splits built: $TRAIN_30K, $TEST_30K"
fi

echo ""
echo "============================================================"
echo "  NRP Fine-Tuning: Qwen-14B + Qwen-32B on issues30k"
echo "============================================================"
echo "  Splits: train=$TRAIN_30K test=$TEST_30K"

EXPERIMENT_START=$(date +%s)

clear_hf_cache

for ft_cfg in "${NRP_FT_MODELS[@]}"; do
  IFS='|' read -r model short_name <<< "$ft_cfg"
  model_tag="$(echo "$model" | tr -c '[:alnum:]' '_' | sed 's/_*$//')"

  echo ""
  echo "  =========================================================="
  echo "  [FIXED FT] $short_name (issues30k) — NRP"
  echo "  =========================================================="

  clear_hf_cache

  FT_DIR="$RESULTS_30K/$model_tag/finetune_fixed"
  FT_PREDS="$FT_DIR/preds_finetune_fixed.csv"
  FT_EVAL="$FT_DIR/eval_finetune_fixed.csv"

  if [[ -f "$FT_PREDS" ]]; then
    echo "    Predictions exist. SKIPPING."
  else
    echo "    Training + inference..."
    mkdir -p "$FT_DIR"
    STAGE_START=$(date +%s)

    "$PYTHON_BIN" "$SCRIPT_DIR/fixed_fine-tune.py" \
      --model "$model" \
      --dataset "$DATASET_30K" \
      --train_csv "$TRAIN_30K" \
      --test_csv "$TEST_30K" \
      --max_seq_length 2048 \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --inference_batch_size "$INFERENCE_BATCH_SIZE" \
      --test_size "$TEST_SIZE_30K" \
      --output_dir "$FT_DIR" \
      "${CACHE_ARGS[@]}"

    ELAPSED=$(($(date +%s) - STAGE_START))
    echo "    [FIXED FT] Done in ${ELAPSED}s"
  fi

  if [[ -f "$FT_PREDS" && ! -f "$FT_EVAL" ]]; then
    echo "    Evaluating..."
    "$PYTHON_BIN" "$SCRIPT_DIR/evaluate.py" \
      --preds_csv "$FT_PREDS" \
      --top_k 0 \
      --output_csv "$FT_EVAL" \
      --model_name "$model"
  fi

  clear_hf_cache
done

EXPERIMENT_END=$(date +%s)
TOTAL=$((EXPERIMENT_END - EXPERIMENT_START))

echo ""
echo "============================================================"
echo "  NRP fine-tuning complete in ${TOTAL}s"
echo "============================================================"
echo "  Results: $RESULTS_30K/"
echo "============================================================"

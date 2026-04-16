#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_final_experiments.sh — 3k Ablation (random-shot) + 30k Generalization
# ============================================================================
# Phase 1: Random-shot ablation on issues3k (proves retrieval quality matters)
#   - For each model's best RAGTAG config, run with random neighbors (3 seeds)
#   - Results → results/ablation_random_3k/
#
# Phase 2: Generalization on issues30k
#   - RAGTAG best configs, fixed fine-tune, VTAG
#   - Results → results/issues30k/
#
# NRP usage:  nohup bash run_final_experiments.sh --nrp 2>&1 | tee experiment.log &
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- Config ---
NRP_MODE=0
HF_CACHE_DIR="$SCRIPT_DIR/hf_cache"
SKIP_PHASE1=0
SKIP_PHASE2=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nrp)         NRP_MODE=1;      shift ;;
    --skip_phase1) SKIP_PHASE1=1;   shift ;;
    --skip_phase2) SKIP_PHASE2=1;   shift ;;
    -h|--help)
      echo "Usage: run_final_experiments.sh [--nrp] [--skip_phase1] [--skip_phase2]"
      exit 0 ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

CACHE_ARGS=()
if [[ "$NRP_MODE" -eq 1 ]]; then
  echo ">>> NRP mode: HF cache → $HF_CACHE_DIR"
  CACHE_ARGS=(--cache_dir "$HF_CACHE_DIR")
fi

# --- Best configs from 3k analysis ---
# Format: MODEL|SHORT_NAME|BEST_K|CTX
BEST_CONFIGS=(
  "unsloth/Llama-3.2-3B-Instruct|Llama-3B|3|8192"
  "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit|Llama-8B|9|8192"
  "unsloth/Qwen2.5-14B-Instruct-bnb-4bit|Qwen-14B|9|8192"
  "unsloth/Qwen2.5-32B-Instruct-bnb-4bit|Qwen-32B|3|8192"
)

EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS=50
INFERENCE_BATCH_SIZE=1
RANDOM_SEEDS="1,2,3"

# --- Helper: nuke HF cache ---
clear_hf_cache() {
  if [[ -d "$HF_CACHE_DIR" ]]; then
    echo ">>> Clearing HF cache at $HF_CACHE_DIR ..."
    rm -rf "$HF_CACHE_DIR"
    echo "  Done."
  fi
}

EXPERIMENT_START=$(date +%s)

# ============================================================================
# PHASE 1: Random-shot ablation on issues3k
# ============================================================================
if [[ "$SKIP_PHASE1" -eq 0 ]]; then
  echo ""
  echo "============================================================"
  echo "  PHASE 1: Random-Shot Ablation (issues3k)"
  echo "============================================================"

  ABLATION_DIR="$SCRIPT_DIR/results/ablation_random_3k"

  # We need the existing FAISS neighbors dir to get train/test splits.
  # Find it from any existing ctx8192 run.
  EXISTING_NEIGHBORS=""
  for d in "$SCRIPT_DIR/results/issues3k_ctx8192"/run_*/neighbors; do
    if [[ -d "$d" && -f "$d/train_split.csv" && -f "$d/test_split.csv" ]]; then
      EXISTING_NEIGHBORS="$d"
      break
    fi
  done

  # If no existing neighbors dir found, we need to build the index first
  if [[ -z "$EXISTING_NEIGHBORS" ]]; then
    echo ">>> No existing neighbors found. Building FAISS index for issues3k..."
    NEIGHBORS_BUILD_DIR="$ABLATION_DIR/neighbors_faiss"
    mkdir -p "$NEIGHBORS_BUILD_DIR"

    # Find max k across all best configs
    MAX_K=0
    for cfg in "${BEST_CONFIGS[@]}"; do
      IFS='|' read -r _ _ k _ <<< "$cfg"
      if [[ "$k" -gt "$MAX_K" ]]; then MAX_K="$k"; fi
    done

    "$PYTHON_BIN" "$SCRIPT_DIR/build_and_query_index.py" \
      --dataset "$SCRIPT_DIR/issues3k.csv" \
      --top_ks "$MAX_K" \
      --test_size 0.5 \
      --embedding_model "$EMBED_MODEL" \
      --output_dir "$NEIGHBORS_BUILD_DIR"

    EXISTING_NEIGHBORS="$NEIGHBORS_BUILD_DIR"
  fi

  TRAIN_CSV="$EXISTING_NEIGHBORS/train_split.csv"
  TEST_CSV="$EXISTING_NEIGHBORS/test_split.csv"
  echo "  Using splits from: $EXISTING_NEIGHBORS"
  echo "  Train: $TRAIN_CSV"
  echo "  Test:  $TEST_CSV"

  # Generate random neighbor CSVs for all best-k values
  RANDOM_NB_DIR="$ABLATION_DIR/random_neighbors"
  ALL_KS=""
  for cfg in "${BEST_CONFIGS[@]}"; do
    IFS='|' read -r _ _ k _ <<< "$cfg"
    if [[ -z "$ALL_KS" ]]; then
      ALL_KS="$k"
    elif [[ ! "$ALL_KS" =~ (^|,)${k}(,|$) ]]; then
      ALL_KS="$ALL_KS,$k"
    fi
  done

  echo ">>> Generating random neighbor CSVs (k=$ALL_KS, seeds=$RANDOM_SEEDS)..."
  "$PYTHON_BIN" "$SCRIPT_DIR/random_neighbors.py" \
    --train_csv "$TRAIN_CSV" \
    --test_csv "$TEST_CSV" \
    --top_ks "$ALL_KS" \
    --seeds "$RANDOM_SEEDS" \
    --output_dir "$RANDOM_NB_DIR"

  # Run each model at its best config with random neighbors
  clear_hf_cache

  IFS=',' read -ra SEED_ARRAY <<< "$RANDOM_SEEDS"
  for cfg in "${BEST_CONFIGS[@]}"; do
    IFS='|' read -r model short_name best_k ctx <<< "$cfg"
    model_tag="$(echo "$model" | tr -c '[:alnum:]' '_' | sed 's/_*$//')"

    echo ""
    echo "  =========================================================="
    echo "  [ABLATION] $short_name — k=$best_k, ctx=$ctx, ${#SEED_ARRAY[@]} seeds"
    echo "  =========================================================="

    clear_hf_cache

    for seed in "${SEED_ARRAY[@]}"; do
      SEED_NB_DIR="$RANDOM_NB_DIR/seed${seed}"
      PRED_DIR="$ABLATION_DIR/$model_tag/seed${seed}/predictions"
      EVAL_DIR="$ABLATION_DIR/$model_tag/seed${seed}/evaluations"
      LOG_DIR="$ABLATION_DIR/$model_tag/seed${seed}/logs"
      mkdir -p "$PRED_DIR" "$EVAL_DIR" "$LOG_DIR"

      # Check if already done
      if [[ -f "$PRED_DIR/preds_k${best_k}.csv" ]]; then
        echo "    [seed=$seed] Predictions exist. SKIPPING."
        continue
      fi

      echo "    [seed=$seed] Running $short_name with random neighbors..."
      STAGE_START=$(date +%s)

      "$PYTHON_BIN" "$SCRIPT_DIR/llm_labeler.py" \
        --model "$model" \
        --neighbors_dir "$SEED_NB_DIR" \
        --top_ks "$best_k" \
        --output_dir "$PRED_DIR" \
        --log_dir "$LOG_DIR" \
        --eval_dir "$EVAL_DIR" \
        --model_name_for_eval "$model" \
        --max_seq_length "$ctx" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --inference_batch_size "$INFERENCE_BATCH_SIZE" \
        "${CACHE_ARGS[@]}"

      ELAPSED=$(($(date +%s) - STAGE_START))
      echo "    [seed=$seed] Done in ${ELAPSED}s"
    done

    clear_hf_cache
  done

  echo ""
  echo ">>> Phase 1 complete. Results: $ABLATION_DIR/"

fi  # SKIP_PHASE1


# ============================================================================
# PHASE 2: Generalization on issues30k
# ============================================================================
if [[ "$SKIP_PHASE2" -eq 0 ]]; then
  echo ""
  echo "============================================================"
  echo "  PHASE 2: Generalization (issues30k)"
  echo "============================================================"

  DATASET_30K="$SCRIPT_DIR/issues30k.csv"
  TEST_SIZE_30K=3000
  if [[ ! -f "$DATASET_30K" ]]; then
    echo "ERROR: issues30k.csv not found at $DATASET_30K" >&2
    exit 1
  fi

  RESULTS_30K="$SCRIPT_DIR/results/issues30k"
  NEIGHBORS_30K="$RESULTS_30K/neighbors"
  TRAIN_30K="$NEIGHBORS_30K/train_split.csv"
  TEST_30K="$NEIGHBORS_30K/test_split.csv"

  # --- Step 0: Build FAISS index for issues30k ---
  # Need max k across RAGTAG configs + VTAG k=16
  VTAG_K=16
  MAX_K_30K=0
  for cfg in "${BEST_CONFIGS[@]}"; do
    IFS='|' read -r _ _ k _ <<< "$cfg"
    if [[ "$k" -gt "$MAX_K_30K" ]]; then MAX_K_30K="$k"; fi
  done
  if [[ "$VTAG_K" -gt "$MAX_K_30K" ]]; then MAX_K_30K="$VTAG_K"; fi

  ALL_KS_30K=""
  for cfg in "${BEST_CONFIGS[@]}"; do
    IFS='|' read -r _ _ k _ <<< "$cfg"
    if [[ -z "$ALL_KS_30K" ]]; then
      ALL_KS_30K="$k"
    elif [[ ! "$ALL_KS_30K" =~ (^|,)${k}(,|$) ]]; then
      ALL_KS_30K="$ALL_KS_30K,$k"
    fi
  done
  # Add VTAG k
  if [[ ! "$ALL_KS_30K" =~ (^|,)${VTAG_K}(,|$) ]]; then
    ALL_KS_30K="$ALL_KS_30K,$VTAG_K"
  fi

  if [[ -f "$TRAIN_30K" && -f "$TEST_30K" ]]; then
    echo ">>> Splits already exist for issues30k. Checking neighbor files..."
    IFS=',' read -ra K30_ARRAY <<< "$ALL_KS_30K"
    all_nb_exist=1
    for k in "${K30_ARRAY[@]}"; do
      if [[ ! -f "$NEIGHBORS_30K/neighbors_k${k}.csv" ]]; then
        all_nb_exist=0
        break
      fi
    done
    if [[ "$all_nb_exist" -eq 1 ]]; then
      echo "  All neighbor files exist. SKIPPING retrieval."
    else
      echo ">>> Building FAISS index for issues30k (k=$ALL_KS_30K)..."
      "$PYTHON_BIN" "$SCRIPT_DIR/build_and_query_index.py" \
        --dataset "$DATASET_30K" \
        --top_ks "$ALL_KS_30K" \
        --test_size "$TEST_SIZE_30K" \
        --embedding_model "$EMBED_MODEL" \
        --output_dir "$NEIGHBORS_30K"
    fi
  else
    echo ">>> Building FAISS index for issues30k (k=$ALL_KS_30K)..."
    mkdir -p "$NEIGHBORS_30K"
    "$PYTHON_BIN" "$SCRIPT_DIR/build_and_query_index.py" \
      --dataset "$DATASET_30K" \
      --top_ks "$ALL_KS_30K" \
      --test_size "$TEST_SIZE_30K" \
      --embedding_model "$EMBED_MODEL" \
      --output_dir "$NEIGHBORS_30K"
  fi

  echo "  Splits: train=$TRAIN_30K test=$TEST_30K"

  # --- Step 1: VTAG on issues30k ---
  echo ""
  echo "  =========================================================="
  echo "  [VTAG] issues30k — k=$VTAG_K, similarity voting"
  echo "  =========================================================="

  VTAG_EVAL_DIR="$RESULTS_30K/vtag/evaluations"
  if [[ -f "$VTAG_EVAL_DIR/eval_k${VTAG_K}.csv" ]]; then
    echo "    VTAG evaluation exists. SKIPPING."
  else
    mkdir -p "$VTAG_EVAL_DIR"
    "$PYTHON_BIN" "$SCRIPT_DIR/vtag.py" \
      --neighbors_csv "$NEIGHBORS_30K/neighbors_k${VTAG_K}.csv" \
      --voting similarity \
      --ks "$VTAG_K" \
      --eval_dir "$VTAG_EVAL_DIR"
    echo "    [VTAG] Done."
  fi

  # --- Step 2: RAGTAG best configs on issues30k ---
  clear_hf_cache

  for cfg in "${BEST_CONFIGS[@]}"; do
    IFS='|' read -r model short_name best_k ctx <<< "$cfg"
    model_tag="$(echo "$model" | tr -c '[:alnum:]' '_' | sed 's/_*$//')"

    echo ""
    echo "  =========================================================="
    echo "  [RAGTAG] $short_name — k=$best_k, ctx=$ctx (issues30k)"
    echo "  =========================================================="

    clear_hf_cache

    PRED_DIR="$RESULTS_30K/$model_tag/ragtag/predictions"
    EVAL_DIR="$RESULTS_30K/$model_tag/ragtag/evaluations"
    LOG_DIR="$RESULTS_30K/$model_tag/ragtag/logs"
    mkdir -p "$PRED_DIR" "$EVAL_DIR" "$LOG_DIR"

    # Include zero-shot for completeness
    RUN_KS="0,$best_k"

    if [[ -f "$PRED_DIR/preds_k${best_k}.csv" ]]; then
      echo "    Predictions exist. SKIPPING."
    else
      echo "    Running RAGTAG..."
      STAGE_START=$(date +%s)

      "$PYTHON_BIN" "$SCRIPT_DIR/llm_labeler.py" \
        --model "$model" \
        --neighbors_dir "$NEIGHBORS_30K" \
        --top_ks "$RUN_KS" \
        --output_dir "$PRED_DIR" \
        --log_dir "$LOG_DIR" \
        --eval_dir "$EVAL_DIR" \
        --model_name_for_eval "$model" \
        --max_seq_length "$ctx" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --inference_batch_size "$INFERENCE_BATCH_SIZE" \
        "${CACHE_ARGS[@]}"

      ELAPSED=$(($(date +%s) - STAGE_START))
      echo "    [RAGTAG] Done in ${ELAPSED}s"
    fi

    # --- Fixed fine-tune for this model ---
    echo ""
    echo "  =========================================================="
    echo "  [FIXED FT] $short_name (issues30k)"
    echo "  =========================================================="

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

  echo ""
  echo ">>> Phase 2 complete. Results: $RESULTS_30K/"

fi  # SKIP_PHASE2


# ============================================================================
# Summary
# ============================================================================
EXPERIMENT_END=$(date +%s)
TOTAL=$((EXPERIMENT_END - EXPERIMENT_START))

echo ""
echo "============================================================"
echo "  All experiments complete in ${TOTAL}s"
echo "============================================================"
echo "  Phase 1 (3k ablation): results/ablation_random_3k/"
echo "  Phase 2 (30k):         results/issues30k/"
echo "============================================================"

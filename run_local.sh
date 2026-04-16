#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_local.sh — Runs on local RTX 4090 (24GB)
# ============================================================================
# Phase 1: ALL random-shot ablation (3k) — inference only, all models fit
# Phase 2: 30k generalization — everything EXCEPT Qwen-14B/32B fine-tuning
#   - FAISS index build
#   - VTAG
#   - RAGTAG for all 4 models
#   - Fine-tune for Llama-3B and Llama-8B only
#
# Usage:  nohup bash run_local.sh 2>&1 | tee local_experiment.log &
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SKIP_PHASE1=0
SKIP_PHASE2=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip_phase1) SKIP_PHASE1=1;   shift ;;
    --skip_phase2) SKIP_PHASE2=1;   shift ;;
    -h|--help)
      echo "Usage: run_local.sh [--skip_phase1] [--skip_phase2]"
      exit 0 ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

# --- Best configs from 3k analysis ---
# Format: MODEL|SHORT_NAME|BEST_K|CTX
BEST_CONFIGS=(
  "unsloth/Llama-3.2-3B-Instruct|Llama-3B|3|8192"
  "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit|Llama-8B|9|8192"
  "unsloth/Qwen2.5-14B-Instruct-bnb-4bit|Qwen-14B|9|8192"
  "unsloth/Qwen2.5-32B-Instruct-bnb-4bit|Qwen-32B|3|8192"
)

# Models that fit fine-tuning on local 4090
LOCAL_FT_MODELS=(
  "unsloth/Llama-3.2-3B-Instruct|Llama-3B"
  "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit|Llama-8B"
)

EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
MAX_NEW_TOKENS=50
INFERENCE_BATCH_SIZE=1
RANDOM_SEEDS="1"

EXPERIMENT_START=$(date +%s)

# ============================================================================
# PHASE 1: Random-shot ablation on issues3k (ALL models — inference only)
# ============================================================================
if [[ "$SKIP_PHASE1" -eq 0 ]]; then
  echo ""
  echo "============================================================"
  echo "  PHASE 1: Random-Shot Ablation (issues3k) — LOCAL"
  echo "============================================================"

  ABLATION_DIR="$SCRIPT_DIR/results/ablation_random_3k"

  # Find existing splits from ctx8192 runs
  EXISTING_NEIGHBORS=""
  for d in "$SCRIPT_DIR/results/issues3k_ctx8192"/run_*/neighbors; do
    if [[ -d "$d" && -f "$d/train_split.csv" && -f "$d/test_split.csv" ]]; then
      EXISTING_NEIGHBORS="$d"
      break
    fi
  done

  if [[ -z "$EXISTING_NEIGHBORS" ]]; then
    echo ">>> No existing neighbors found. Building FAISS index for issues3k..."
    NEIGHBORS_BUILD_DIR="$ABLATION_DIR/neighbors_faiss"
    mkdir -p "$NEIGHBORS_BUILD_DIR"

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

  # Generate random neighbor CSVs
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

  IFS=',' read -ra SEED_ARRAY <<< "$RANDOM_SEEDS"
  for cfg in "${BEST_CONFIGS[@]}"; do
    IFS='|' read -r model short_name best_k ctx <<< "$cfg"
    model_tag="$(echo "$model" | tr -c '[:alnum:]' '_' | sed 's/_*$//')"

    echo ""
    echo "  =========================================================="
    echo "  [ABLATION] $short_name — k=$best_k, ctx=$ctx, ${#SEED_ARRAY[@]} seeds"
    echo "  =========================================================="

    for seed in "${SEED_ARRAY[@]}"; do
      SEED_NB_DIR="$RANDOM_NB_DIR/seed${seed}"
      PRED_DIR="$ABLATION_DIR/$model_tag/seed${seed}/predictions"
      EVAL_DIR="$ABLATION_DIR/$model_tag/seed${seed}/evaluations"
      LOG_DIR="$ABLATION_DIR/$model_tag/seed${seed}/logs"
      mkdir -p "$PRED_DIR" "$EVAL_DIR" "$LOG_DIR"

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
        --inference_batch_size "$INFERENCE_BATCH_SIZE"

      ELAPSED=$(($(date +%s) - STAGE_START))
      echo "    [seed=$seed] Done in ${ELAPSED}s"
    done
  done

  echo ""
  echo ">>> Phase 1 complete. Results: $ABLATION_DIR/"
fi


# ============================================================================
# PHASE 2: Generalization on issues30k
# ============================================================================
if [[ "$SKIP_PHASE2" -eq 0 ]]; then
  echo ""
  echo "============================================================"
  echo "  PHASE 2: Generalization (issues30k) — LOCAL"
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

  # --- Step 0: Build FAISS index ---
  VTAG_K=16
  ALL_KS_30K=""
  for cfg in "${BEST_CONFIGS[@]}"; do
    IFS='|' read -r _ _ k _ <<< "$cfg"
    if [[ -z "$ALL_KS_30K" ]]; then
      ALL_KS_30K="$k"
    elif [[ ! "$ALL_KS_30K" =~ (^|,)${k}(,|$) ]]; then
      ALL_KS_30K="$ALL_KS_30K,$k"
    fi
  done
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

  # --- Step 1: VTAG ---
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

  # --- Step 2: RAGTAG for ALL models ---
  for cfg in "${BEST_CONFIGS[@]}"; do
    IFS='|' read -r model short_name best_k ctx <<< "$cfg"
    model_tag="$(echo "$model" | tr -c '[:alnum:]' '_' | sed 's/_*$//')"

    echo ""
    echo "  =========================================================="
    echo "  [RAGTAG] $short_name — k=$best_k, ctx=$ctx (issues30k)"
    echo "  =========================================================="

    PRED_DIR="$RESULTS_30K/$model_tag/ragtag/predictions"
    EVAL_DIR="$RESULTS_30K/$model_tag/ragtag/evaluations"
    LOG_DIR="$RESULTS_30K/$model_tag/ragtag/logs"
    mkdir -p "$PRED_DIR" "$EVAL_DIR" "$LOG_DIR"

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
        --inference_batch_size "$INFERENCE_BATCH_SIZE"

      ELAPSED=$(($(date +%s) - STAGE_START))
      echo "    [RAGTAG] Done in ${ELAPSED}s"
    fi
  done

  # --- Step 3: Fine-tune Llama-3B and Llama-8B ONLY ---
  for ft_cfg in "${LOCAL_FT_MODELS[@]}"; do
    IFS='|' read -r model short_name <<< "$ft_cfg"
    model_tag="$(echo "$model" | tr -c '[:alnum:]' '_' | sed 's/_*$//')"

    echo ""
    echo "  =========================================================="
    echo "  [FIXED FT] $short_name (issues30k) — LOCAL"
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
        --output_dir "$FT_DIR"

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
  done

  echo ""
  echo ">>> Phase 2 (local portion) complete. Results: $RESULTS_30K/"
  echo ">>> REMAINING: Qwen-14B and Qwen-32B fine-tuning → run run_nrp.sh on NRP"
fi


EXPERIMENT_END=$(date +%s)
TOTAL=$((EXPERIMENT_END - EXPERIMENT_START))

echo ""
echo "============================================================"
echo "  Local experiments complete in ${TOTAL}s"
echo "============================================================"
echo "  Phase 1 (3k ablation): results/ablation_random_3k/"
echo "  Phase 2 (30k local):   results/issues30k/"
echo "  TODO on NRP:           Qwen-14B + Qwen-32B fine-tuning"
echo "============================================================"

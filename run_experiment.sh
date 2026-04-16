#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_experiment.sh — Unified RAGTAG+ vs Fine-Tune Experiment Orchestrator
# ============================================================================
# For each model × dataset combination:
#   1. Run RAGTAG pipeline (build index, retrieve neighbors, label, evaluate)
#      → produces train/test splits in neighbors dir
#   2. Run flawed fine-tune baseline using SAME train/test splits
#   3. Run fixed fine-tune using SAME train/test splits
#   4. Aggregate all results across approaches
#
# This ensures apples-to-apples comparison: same data, different methods.
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: python3 or python is required." >&2
    exit 1
  fi
fi

usage() {
  cat <<'USAGE'
Usage: run_experiment.sh [options]

Required:
  --datasets "d1.csv,d2.csv"  Comma-separated dataset CSV paths
  --models "m1,m2"            Comma-separated HuggingFace model specs
                              Format: model_id[:max_new_tokens]

Options:
  --top_ks "1,3,5,7,9"       Comma-separated k values for RAGTAG (default: 1,3,5,7,9)
  --test_sizes "d1:0.5,d2:3000"
                              Per-dataset test sizes (default: 0.5 for all)
                              Format: dataset_basename:size,...
  --max_seq_length N          Max context window for RAGTAG+LLM (default: 4096)
  --ft_max_seq_length N       Max context window for fine-tuning (default: same as --max_seq_length)
  --max_new_tokens N          Default max tokens for generation (default: 50)
  --inference_batch_size N    Batch size for inference in RAGTAG and FT (default: 1)
  --embedding_model NAME      HF embedding model (default: sentence-transformers/all-MiniLM-L6-v2)
  --results_dir PATH          Root output directory (default: results/)
  --model_cache_dir PATH      HF model cache directory
  --skip_ragtag               Skip RAGTAG (reuse existing results)
  --skip_flawed_ft            Skip flawed fine-tune baseline
  --skip_fixed_ft             Skip fixed fine-tune
  --nrp                       NRP cluster mode (sets cache dir)
  -h, --help                  Show this help
USAGE
}

# --- Defaults ---
DATASETS_SPEC=""
MODELS_SPEC=""
TOP_KS="1,3,5,7,9"
TEST_SIZES_SPEC=""
MAX_SEQ_LENGTH=16384
FT_MAX_SEQ_LENGTH=""
MAX_NEW_TOKENS=50
INFERENCE_BATCH_SIZE=1
EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
RESULTS_DIR="$SCRIPT_DIR/results"
MODEL_CACHE_DIR=""
SKIP_RAGTAG=0
SKIP_FLAWED_FT=0
SKIP_FIXED_FT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)       DATASETS_SPEC="$2";     shift 2 ;;
    --models)         MODELS_SPEC="$2";        shift 2 ;;
    --top_ks)         TOP_KS="$2";             shift 2 ;;
    --test_sizes)     TEST_SIZES_SPEC="$2";    shift 2 ;;
    --max_seq_length) MAX_SEQ_LENGTH="$2";     shift 2 ;;
    --ft_max_seq_length) FT_MAX_SEQ_LENGTH="$2"; shift 2 ;;
    --max_new_tokens) MAX_NEW_TOKENS="$2";     shift 2 ;;
    --inference_batch_size) INFERENCE_BATCH_SIZE="$2"; shift 2 ;;
    --embedding_model) EMBED_MODEL="$2";       shift 2 ;;
    --results_dir)    RESULTS_DIR="$2";        shift 2 ;;
    --model_cache_dir) MODEL_CACHE_DIR="$2";   shift 2 ;;
    --skip_ragtag)    SKIP_RAGTAG=1;           shift ;;
    --skip_flawed_ft) SKIP_FLAWED_FT=1;       shift ;;
    --skip_fixed_ft)  SKIP_FIXED_FT=1;        shift ;;
    --nrp)
      MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$(pwd)/hf_cache}"
      echo ">>> NRP mode: cache → $MODEL_CACHE_DIR"
      shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage; exit 1 ;;
  esac
done

# --- Validate ---
if [[ -z "$DATASETS_SPEC" ]]; then
  echo "Error: --datasets required" >&2; exit 1
fi
if [[ -z "$MODELS_SPEC" ]]; then
  echo "Error: --models required" >&2; exit 1
fi

if [[ -z "$FT_MAX_SEQ_LENGTH" ]]; then
  FT_MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH"
fi

# --- Parse datasets ---
IFS=',' read -ra DATASETS <<< "$DATASETS_SPEC"

# --- Parse per-dataset test sizes ---
declare -A DATASET_TEST_SIZE
if [[ -n "$TEST_SIZES_SPEC" ]]; then
  IFS=',' read -ra TS_PAIRS <<< "$TEST_SIZES_SPEC"
  for pair in "${TS_PAIRS[@]}"; do
    IFS=':' read -r ds ts <<< "$pair"
    DATASET_TEST_SIZE["$ds"]="$ts"
  done
fi

# --- Parse model specs ---
IFS=',' read -ra MODEL_SPECS <<< "$MODELS_SPEC"
declare -a MODEL_NAMES MODEL_TAGS MODEL_TOKENS
for spec in "${MODEL_SPECS[@]}"; do
  if [[ "$spec" =~ ^(.+):([0-9]+)$ ]]; then
    m="${BASH_REMATCH[1]}"
    t="${BASH_REMATCH[2]}"
  else
    m="$spec"
    t="$MAX_NEW_TOKENS"
  fi
  tag="$(echo "$m" | tr -c '[:alnum:]' '_' | sed 's/_*$//')"
  MODEL_NAMES+=("$m")
  MODEL_TAGS+=("$tag")
  MODEL_TOKENS+=("$t")
done

CACHE_ARGS=()
if [[ -n "$MODEL_CACHE_DIR" ]]; then
  CACHE_ARGS=(--model_cache_dir "$MODEL_CACHE_DIR")
fi

CACHE_ARGS_FT=()
if [[ -n "$MODEL_CACHE_DIR" ]]; then
  CACHE_ARGS_FT=(--cache_dir "$MODEL_CACHE_DIR")
fi

echo ""
echo "============================================================"
echo "  RAGTAG+ vs Fine-Tune Experiment"
echo "============================================================"
echo "  Datasets:            ${DATASETS[*]}"
echo "  Models:              ${MODEL_TAGS[*]}"
echo "  K values:            $TOP_KS"
echo "  max_seq_length:      $MAX_SEQ_LENGTH (RAGTAG)"
echo "  ft_max_seq_length:   $FT_MAX_SEQ_LENGTH (Fine-Tune)"
echo "  max_new_tokens:      $MAX_NEW_TOKENS"
echo "  inference_batch_size:$INFERENCE_BATCH_SIZE"
echo "  Embedding:           $EMBED_MODEL"
echo "  Results:             $RESULTS_DIR"
echo "  Skip RAGTAG:         $SKIP_RAGTAG"
echo "  Skip Flawed FT:      $SKIP_FLAWED_FT"
echo "  Skip Fixed FT:       $SKIP_FIXED_FT"
echo "============================================================"

EXPERIMENT_START=$(date +%s)
TIMING_FILE="$RESULTS_DIR/experiment_timing.csv"
mkdir -p "$RESULTS_DIR"
echo "stage,dataset,model,approach,seconds" > "$TIMING_FILE"

# ============================================================================
# Main Loop: dataset × model
# ============================================================================
for dataset_path in "${DATASETS[@]}"; do
  dataset_basename="$(basename "$dataset_path")"
  dataset_tag="${dataset_basename%.csv}"

  # Resolve dataset path
  if [[ ! -f "$dataset_path" ]]; then
    if [[ -f "$SCRIPT_DIR/$dataset_path" ]]; then
      dataset_path="$SCRIPT_DIR/$dataset_path"
    else
      echo "WARNING: Dataset not found: $dataset_path — skipping."
      continue
    fi
  fi

  # Get test size for this dataset
  test_size="${DATASET_TEST_SIZE[$dataset_basename]:-0.5}"

  DATASET_DIR="$RESULTS_DIR/$dataset_tag"

  echo ""
  echo "============================================================"
  echo "  Dataset: $dataset_basename (test_size=$test_size)"
  echo "============================================================"

  # =====================================================================
  # STEP 0: RAGTAG retrieval (shared across all models for this dataset)
  # =====================================================================
  NEIGHBORS_DIR="$DATASET_DIR/neighbors"
  TRAIN_SPLIT="$NEIGHBORS_DIR/train_split.csv"
  TEST_SPLIT="$NEIGHBORS_DIR/test_split.csv"

  if [[ "$SKIP_RAGTAG" -eq 0 ]]; then
    # Check if retrieval already done
    if [[ -f "$TRAIN_SPLIT" && -f "$TEST_SPLIT" ]]; then
      echo ">>> Retrieval: splits already exist, checking neighbor files..."
      IFS=',' read -ra K_ARRAY <<< "$TOP_KS"
      all_nb_exist=1
      for k in "${K_ARRAY[@]}"; do
        if [[ ! -f "$NEIGHBORS_DIR/neighbors_k${k}.csv" ]]; then
          all_nb_exist=0
          break
        fi
      done
      if [[ "$all_nb_exist" -eq 1 ]]; then
        echo "  All neighbor files exist. SKIPPING retrieval."
      else
        echo ">>> Running FAISS retrieval..."
        STAGE_START=$(date +%s)
        RETRIEVAL_EXTRA_ARGS=()
        if [[ -n "$MODEL_CACHE_DIR" ]]; then
          RETRIEVAL_EXTRA_ARGS+=(--model_cache_dir "$MODEL_CACHE_DIR")
        fi
        "$PYTHON_BIN" "$SCRIPT_DIR/build_and_query_index.py" \
          --dataset "$dataset_path" \
          --top_ks "$TOP_KS" \
          --test_size "$test_size" \
          --embedding_model "$EMBED_MODEL" \
          --output_dir "$NEIGHBORS_DIR" \
          "${RETRIEVAL_EXTRA_ARGS[@]}"
        echo "retrieval,$dataset_tag,-,ragtag,$(($(date +%s) - STAGE_START))" >> "$TIMING_FILE"
      fi
    else
      echo ">>> Running FAISS retrieval..."
      mkdir -p "$NEIGHBORS_DIR"
      STAGE_START=$(date +%s)
      RETRIEVAL_EXTRA_ARGS=()
      if [[ -n "$MODEL_CACHE_DIR" ]]; then
        RETRIEVAL_EXTRA_ARGS+=(--model_cache_dir "$MODEL_CACHE_DIR")
      fi
      "$PYTHON_BIN" "$SCRIPT_DIR/build_and_query_index.py" \
        --dataset "$dataset_path" \
        --top_ks "$TOP_KS" \
        --test_size "$test_size" \
        --embedding_model "$EMBED_MODEL" \
        --output_dir "$NEIGHBORS_DIR" \
        "${RETRIEVAL_EXTRA_ARGS[@]}"
      echo "retrieval,$dataset_tag,-,ragtag,$(($(date +%s) - STAGE_START))" >> "$TIMING_FILE"
    fi
  else
    echo ">>> Retrieval: SKIPPED (--skip_ragtag)"
  fi

  # Verify splits exist
  if [[ ! -f "$TRAIN_SPLIT" || ! -f "$TEST_SPLIT" ]]; then
    echo "ERROR: Train/test splits not found at $NEIGHBORS_DIR. Run without --skip_ragtag first." >&2
    exit 1
  fi

  echo "  Shared splits: train=$TRAIN_SPLIT test=$TEST_SPLIT"

  # =====================================================================
  # Per-model loop
  # =====================================================================
  for mi in "${!MODEL_NAMES[@]}"; do
    model="${MODEL_NAMES[$mi]}"
    tag="${MODEL_TAGS[$mi]}"
    model_max_tokens="${MODEL_TOKENS[$mi]}"

    MODEL_DIR="$DATASET_DIR/$tag"

    echo ""
    echo "  =========================================================="
    echo "  Model: $model  (max_new_tokens=$model_max_tokens)"
    echo "  =========================================================="

    # -----------------------------------------------------------------
    # A. RAGTAG approach
    # -----------------------------------------------------------------
    if [[ "$SKIP_RAGTAG" -eq 0 ]]; then
      RAGTAG_PRED_DIR="$MODEL_DIR/ragtag/predictions"
      RAGTAG_EVAL_DIR="$MODEL_DIR/ragtag/evaluations"
      RAGTAG_LOG_DIR="$MODEL_DIR/ragtag/logs"
      mkdir -p "$RAGTAG_PRED_DIR" "$RAGTAG_EVAL_DIR" "$RAGTAG_LOG_DIR"

      ALL_KS="0,${TOP_KS}"

      # Check if already done
      IFS=',' read -ra ALL_K_ARRAY <<< "$ALL_KS"
      ragtag_done=1
      for ak in "${ALL_K_ARRAY[@]}"; do
        if [[ "$ak" -eq 0 ]]; then
          pf="$RAGTAG_PRED_DIR/preds_zero_shot.csv"
        else
          pf="$RAGTAG_PRED_DIR/preds_k${ak}.csv"
        fi
        if [[ ! -f "$pf" ]]; then
          ragtag_done=0
          break
        fi
      done

      if [[ "$ragtag_done" -eq 1 ]]; then
        echo "    [RAGTAG] All predictions exist. SKIPPING."
      else
        echo "    [RAGTAG] Running LLM labeling (K=${ALL_KS})..."
        STAGE_START=$(date +%s)

        LLM_EXTRA_ARGS=()
        if [[ -n "$MODEL_CACHE_DIR" ]]; then
          LLM_EXTRA_ARGS+=(--cache_dir "$MODEL_CACHE_DIR")
        fi

        "$PYTHON_BIN" "$SCRIPT_DIR/llm_labeler.py" \
          --model "$model" \
          --neighbors_dir "$NEIGHBORS_DIR" \
          --top_ks "$ALL_KS" \
          --output_dir "$RAGTAG_PRED_DIR" \
          --log_dir "$RAGTAG_LOG_DIR" \
          --eval_dir "$RAGTAG_EVAL_DIR" \
          --model_name_for_eval "$model" \
          --max_seq_length "$MAX_SEQ_LENGTH" \
          --max_new_tokens "$model_max_tokens" \
          --inference_batch_size "$INFERENCE_BATCH_SIZE" \
          "${LLM_EXTRA_ARGS[@]}"

        ELAPSED=$(($(date +%s) - STAGE_START))
        echo "ragtag,$dataset_tag,$tag,ragtag,$ELAPSED" >> "$TIMING_FILE"
        echo "    [RAGTAG] Done in ${ELAPSED}s"
      fi
    fi

    # -----------------------------------------------------------------
    # B. Flawed Fine-Tune Baseline
    # -----------------------------------------------------------------
    if [[ "$SKIP_FLAWED_FT" -eq 0 ]]; then
      FLAWED_DIR="$MODEL_DIR/finetune_flawed"
      FLAWED_PREDS="$FLAWED_DIR/preds_finetune_flawed.csv"
      FLAWED_EVAL="$FLAWED_DIR/eval_finetune_flawed.csv"

      if [[ -f "$FLAWED_PREDS" ]]; then
        echo "    [FLAWED FT] Predictions exist. SKIPPING."
      else
        echo "    [FLAWED FT] Training + inference..."
        mkdir -p "$FLAWED_DIR"
        STAGE_START=$(date +%s)

        "$PYTHON_BIN" "$SCRIPT_DIR/baseline_finetune_flawed.py" \
          --model "$model" \
          --dataset "$dataset_path" \
          --train_csv "$TRAIN_SPLIT" \
          --test_csv "$TEST_SPLIT" \
          --max_seq_length "$FT_MAX_SEQ_LENGTH" \
          --max_new_tokens "$model_max_tokens" \
          --inference_batch_size "$INFERENCE_BATCH_SIZE" \
          --test_size "$test_size" \
          --output_dir "$FLAWED_DIR" \
          "${CACHE_ARGS_FT[@]}"

        ELAPSED=$(($(date +%s) - STAGE_START))
        echo "finetune,$dataset_tag,$tag,flawed,$ELAPSED" >> "$TIMING_FILE"
        echo "    [FLAWED FT] Done in ${ELAPSED}s"
      fi

      # Evaluate with standard evaluate.py
      if [[ -f "$FLAWED_PREDS" && ! -f "$FLAWED_EVAL" ]]; then
        echo "    [FLAWED FT] Evaluating..."
        "$PYTHON_BIN" "$SCRIPT_DIR/evaluate.py" \
          --preds_csv "$FLAWED_PREDS" \
          --top_k 0 \
          --output_csv "$FLAWED_EVAL" \
          --model_name "$model"
      fi
    fi

    # -----------------------------------------------------------------
    # C. Fixed Fine-Tune
    # -----------------------------------------------------------------
    if [[ "$SKIP_FIXED_FT" -eq 0 ]]; then
      FIXED_DIR="$MODEL_DIR/finetune_fixed"
      FIXED_PREDS="$FIXED_DIR/preds_finetune_fixed.csv"
      FIXED_EVAL="$FIXED_DIR/eval_finetune_fixed.csv"

      if [[ -f "$FIXED_PREDS" ]]; then
        echo "    [FIXED FT] Predictions exist. SKIPPING."
      else
        echo "    [FIXED FT] Training + inference..."
        mkdir -p "$FIXED_DIR"
        STAGE_START=$(date +%s)

        "$PYTHON_BIN" "$SCRIPT_DIR/fixed_fine-tune.py" \
          --model "$model" \
          --dataset "$dataset_path" \
          --train_csv "$TRAIN_SPLIT" \
          --test_csv "$TEST_SPLIT" \
          --max_seq_length "$FT_MAX_SEQ_LENGTH" \
          --max_new_tokens "$model_max_tokens" \
          --inference_batch_size "$INFERENCE_BATCH_SIZE" \
          --test_size "$test_size" \
          --output_dir "$FIXED_DIR" \
          "${CACHE_ARGS_FT[@]}"

        ELAPSED=$(($(date +%s) - STAGE_START))
        echo "finetune,$dataset_tag,$tag,fixed,$ELAPSED" >> "$TIMING_FILE"
        echo "    [FIXED FT] Done in ${ELAPSED}s"
      fi

      # Evaluate with standard evaluate.py
      if [[ -f "$FIXED_PREDS" && ! -f "$FIXED_EVAL" ]]; then
        echo "    [FIXED FT] Evaluating..."
        "$PYTHON_BIN" "$SCRIPT_DIR/evaluate.py" \
          --preds_csv "$FIXED_PREDS" \
          --top_k 0 \
          --output_csv "$FIXED_EVAL" \
          --model_name "$model"
      fi
    fi

  done  # models
done  # datasets

# ============================================================================
# Aggregate ALL results across datasets, models, approaches
# ============================================================================
echo ""
echo ">>> Aggregating all results..."
"$PYTHON_BIN" - "$RESULTS_DIR" <<'PYAGG'
import sys, os, pandas as pd

results_dir = sys.argv[1]
eval_dfs = []
cost_dfs = []

for root, dirs, files in os.walk(results_dir):
    for f in files:
        fp = os.path.join(root, f)
        # Collect evaluation CSVs
        if f.startswith("eval_") and f.endswith(".csv"):
            try:
                df = pd.read_csv(fp)
                # Infer approach from path
                rel = os.path.relpath(fp, results_dir)
                parts = rel.split(os.sep)
                if "ragtag" in rel:
                    df["approach"] = "ragtag"
                elif "finetune_flawed" in rel:
                    df["approach"] = "finetune_flawed"
                elif "finetune_fixed" in rel:
                    df["approach"] = "finetune_fixed"
                else:
                    df["approach"] = "unknown"
                # Infer dataset from path
                if len(parts) >= 1:
                    df["dataset"] = parts[0]
                eval_dfs.append(df)
            except Exception as e:
                print(f"  WARN: Could not read {fp}: {e}")

        # Collect cost metrics
        if f == "cost_metrics.csv":
            try:
                df = pd.read_csv(fp)
                rel = os.path.relpath(fp, results_dir)
                parts = rel.split(os.sep)
                if "ragtag" in rel:
                    df["approach"] = "ragtag"
                elif "finetune_flawed" in rel:
                    df["approach"] = "finetune_flawed"
                elif "finetune_fixed" in rel:
                    df["approach"] = "finetune_fixed"
                else:
                    df["approach"] = "unknown"
                if len(parts) >= 1:
                    df["dataset"] = parts[0]
                cost_dfs.append(df)
            except Exception as e:
                print(f"  WARN: Could not read {fp}: {e}")

# Write aggregated evaluations
if eval_dfs:
    all_evals = pd.concat(eval_dfs, ignore_index=True)
    out = os.path.join(results_dir, "all_results.csv")
    all_evals.to_csv(out, index=False)
    print(f"  Wrote aggregated results: {out} ({len(all_evals)} rows)")
    print(all_evals.to_string(index=False))
else:
    print("  No evaluation results found.")

# Write aggregated cost metrics
if cost_dfs:
    all_costs = pd.concat(cost_dfs, ignore_index=True)
    out = os.path.join(results_dir, "all_cost_metrics.csv")
    all_costs.to_csv(out, index=False)
    print(f"\n  Wrote aggregated cost metrics: {out} ({len(all_costs)} rows)")
else:
    print("  No cost metrics found.")
PYAGG

EXPERIMENT_END=$(date +%s)
TOTAL=$((EXPERIMENT_END - EXPERIMENT_START))

echo ""
echo "============================================================"
echo "  Experiment complete in ${TOTAL}s"
echo "============================================================"
echo "  Results dir:     $RESULTS_DIR"
echo "  All results:     $RESULTS_DIR/all_results.csv"
echo "  Cost metrics:    $RESULTS_DIR/all_cost_metrics.csv"
echo "  Timing:          $TIMING_FILE"
echo "============================================================"

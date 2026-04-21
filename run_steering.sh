#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_steering.sh — Activation Steering Experiment Runner
# ============================================================================
# Full experiment pipeline:
#   Phase 1: Compute steering vectors (3 strategies)
#   Phase 2: Layer sweep (find optimal layer)
#   Phase 3: Multiplier sweep (at best layer)
#   Phase 4: Compare pair strategies (at best config)
#   Phase 5: NTW directional ablation comparison
#   Phase 6: Print summary table
#
# Usage:
#   ./run_steering.sh                    # full pipeline
#   ./run_steering.sh --phase 2          # start from phase 2
#   ./run_steering.sh --nrp              # NRP cluster mode
#   ./run_steering.sh --skip_compute     # skip vector computation
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON_BIN:-venv/bin/python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    echo "Error: python3 or python is required." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL="unsloth/Llama-3.2-3B-Instruct"
TRAIN_CSV="results/ablation_random_3k/neighbors_faiss/train_split.csv"
NEIGHBORS_CSV="results/issues3k_debias/neighbors/neighbors_k3.csv"
TOP_K=3
CTX=8192
MAX_NEW_TOKENS=20
BATCH_SIZE=1
COMPUTE_BATCH=4
MAX_PAIRS=300
COMPUTE_CTX=4096

# Steering vector output dirs
SV_DIR="results/steering_vectors"
SV_ANSWER="${SV_DIR}/llama3b_3k_answer"
SV_FAISS="${SV_DIR}/llama3b_3k_faiss"
SV_MEANS="${SV_DIR}/llama3b_3k_means"

# Results dir
RESULTS="results/issues3k_steering"

# Multipliers to sweep
MULTIPLIERS="-0.5 -1.0 -1.5 -2.0 -3.0"

CACHE_DIR=""
START_PHASE=1
SKIP_COMPUTE=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)           MODEL="$2";           shift 2 ;;
    --train_csv)       TRAIN_CSV="$2";       shift 2 ;;
    --neighbors_csv)   NEIGHBORS_CSV="$2";   shift 2 ;;
    --top_k)           TOP_K="$2";           shift 2 ;;
    --max_seq_length)  CTX="$2";             shift 2 ;;
    --batch_size)      BATCH_SIZE="$2";      shift 2 ;;
    --phase)           START_PHASE="$2";     shift 2 ;;
    --skip_compute)    SKIP_COMPUTE=1;       shift ;;
    --nrp)
      CACHE_DIR="${CACHE_DIR:-$(pwd)/hf_cache}"
      echo ">>> NRP mode: cache -> $CACHE_DIR"
      shift ;;
    --cache_dir)       CACHE_DIR="$2";       shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--phase N] [--skip_compute] [--nrp] [--model M] [--batch_size N]"
      exit 0 ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

CACHE_FLAG=""
if [[ -n "$CACHE_DIR" ]]; then
  CACHE_FLAG="--cache_dir $CACHE_DIR"
fi

echo "============================================================"
echo "  Activation Steering Experiment"
echo "============================================================"
echo "  Model:          $MODEL"
echo "  Train CSV:      $TRAIN_CSV"
echo "  Neighbors CSV:  $NEIGHBORS_CSV"
echo "  Top K:          $TOP_K"
echo "  Context:        $CTX"
echo "  Start phase:    $START_PHASE"
echo "  Skip compute:   $SKIP_COMPUTE"
echo "============================================================"

# ---------------------------------------------------------------------------
# Helper: extract best layer from layer sweep CSV
# ---------------------------------------------------------------------------
get_best_layer() {
  local sweep_csv="$1"
  $PYTHON -c "
import pandas as pd
df = pd.read_csv('$sweep_csv')
best = df.loc[df['f1_macro'].idxmax()]
print(int(best['layer']))
"
}

# ============================================================================
# Phase 1: Compute steering vectors
# ============================================================================
if [[ $START_PHASE -le 1 && $SKIP_COMPUTE -eq 0 ]]; then
  echo ""
  echo "============================================================"
  echo "  PHASE 1: Compute Steering Vectors"
  echo "============================================================"

  # --- Strategy A: answer_conditioned ---
  if [[ -f "${SV_ANSWER}/steering_vectors.pt" ]]; then
    echo "[SKIP] answer_conditioned vectors already exist"
  else
    echo ""
    echo ">>> Computing answer_conditioned vectors..."
    $PYTHON compute_steering_vector.py \
      --model "$MODEL" \
      --train_csv "$TRAIN_CSV" \
      --output_dir "$SV_ANSWER" \
      --pair_strategy answer_conditioned \
      --max_pairs "$MAX_PAIRS" \
      --max_seq_length "$COMPUTE_CTX" \
      --batch_size "$COMPUTE_BATCH" \
      $CACHE_FLAG
  fi

  # --- Strategy B: faiss_matched ---
  if [[ -f "${SV_FAISS}/steering_vectors.pt" ]]; then
    echo "[SKIP] faiss_matched vectors already exist"
  else
    echo ""
    echo ">>> Computing faiss_matched vectors..."
    $PYTHON compute_steering_vector.py \
      --model "$MODEL" \
      --train_csv "$TRAIN_CSV" \
      --output_dir "$SV_FAISS" \
      --pair_strategy faiss_matched \
      --max_pairs "$MAX_PAIRS" \
      --max_seq_length "$COMPUTE_CTX" \
      --batch_size "$COMPUTE_BATCH" \
      $CACHE_FLAG
  fi

  # --- Strategy C: class_means ---
  if [[ -f "${SV_MEANS}/steering_vectors.pt" ]]; then
    echo "[SKIP] class_means vectors already exist"
  else
    echo ""
    echo ">>> Computing class_means vectors..."
    $PYTHON compute_steering_vector.py \
      --model "$MODEL" \
      --train_csv "$TRAIN_CSV" \
      --output_dir "$SV_MEANS" \
      --pair_strategy class_means \
      --max_seq_length "$COMPUTE_CTX" \
      --batch_size "$COMPUTE_BATCH" \
      $CACHE_FLAG
  fi

  echo ""
  echo "  Phase 1 complete. Steering vectors saved to ${SV_DIR}/"
fi

# ============================================================================
# Phase 2: Layer sweep (answer_conditioned, multiplier=-1.0)
# ============================================================================
SWEEP_DIR="${RESULTS}/layer_sweep_answer"
SWEEP_CSV="${SWEEP_DIR}/layer_sweep_results.csv"

if [[ $START_PHASE -le 2 ]]; then
  echo ""
  echo "============================================================"
  echo "  PHASE 2: Layer Sweep (answer_conditioned, m=-1.0)"
  echo "============================================================"

  if [[ -f "$SWEEP_CSV" ]]; then
    echo "[SKIP] Layer sweep results already exist: $SWEEP_CSV"
  else
    $PYTHON activation_steering.py \
      --model "$MODEL" \
      --neighbors_csv "$NEIGHBORS_CSV" \
      --steering_vectors "${SV_ANSWER}/steering_vectors.pt" \
      --output_dir "$SWEEP_DIR" \
      --method caa \
      --layer sweep \
      --multiplier -1.0 \
      --top_k "$TOP_K" \
      --max_seq_length "$CTX" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --inference_batch_size "$BATCH_SIZE" \
      --eval_dir "${SWEEP_DIR}/evaluations" \
      $CACHE_FLAG
  fi

  echo ""
  echo "  Phase 2 complete."
fi

# ============================================================================
# Phase 3: Multiplier sweep at best layer
# ============================================================================
if [[ $START_PHASE -le 3 ]]; then
  echo ""
  echo "============================================================"
  echo "  PHASE 3: Multiplier Sweep at Best Layer"
  echo "============================================================"

  if [[ ! -f "$SWEEP_CSV" ]]; then
    echo "[ERROR] Layer sweep results not found: $SWEEP_CSV"
    echo "        Run phase 2 first."
    exit 1
  fi

  BEST_LAYER=$(get_best_layer "$SWEEP_CSV")
  echo "  Best layer from sweep: $BEST_LAYER"

  MULT_DIR="${RESULTS}/multiplier_sweep_answer"
  for m in $MULTIPLIERS; do
    PRED_FILE="${MULT_DIR}/predictions/preds_caa_layer${BEST_LAYER}_m${m}.csv"
    if [[ -f "$PRED_FILE" ]]; then
      echo "[SKIP] m=${m} already exists"
      continue
    fi

    echo ""
    echo ">>> CAA layer=${BEST_LAYER}, multiplier=${m}"
    $PYTHON activation_steering.py \
      --model "$MODEL" \
      --neighbors_csv "$NEIGHBORS_CSV" \
      --steering_vectors "${SV_ANSWER}/steering_vectors.pt" \
      --output_dir "$MULT_DIR" \
      --method caa \
      --layer "$BEST_LAYER" \
      --multiplier "$m" \
      --top_k "$TOP_K" \
      --max_seq_length "$CTX" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --inference_batch_size "$BATCH_SIZE" \
      --eval_dir "${MULT_DIR}/evaluations" \
      $CACHE_FLAG
  done

  echo ""
  echo "  Phase 3 complete."
fi

# ============================================================================
# Phase 4: Compare pair strategies at best (layer, multiplier)
# ============================================================================
if [[ $START_PHASE -le 4 ]]; then
  echo ""
  echo "============================================================"
  echo "  PHASE 4: Compare Pair Strategies at Best Config"
  echo "============================================================"

  if [[ ! -f "$SWEEP_CSV" ]]; then
    echo "[ERROR] Layer sweep results not found. Run phase 2 first."
    exit 1
  fi

  BEST_LAYER=$(get_best_layer "$SWEEP_CSV")

  # Find best multiplier from phase 3
  MULT_DIR="${RESULTS}/multiplier_sweep_answer"
  BEST_MULT=$($PYTHON -c "
import pandas as pd, os, sys
from sklearn.metrics import precision_recall_fscore_support

LABELS = ['bug', 'feature', 'question']
best_f1 = -1
best_m = -1.0
for m in [${MULTIPLIERS// /, }]:
    path = '${MULT_DIR}/predictions/preds_caa_layer${BEST_LAYER}_m' + str(m) + '.csv'
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    valid = df[df['predicted_label'].isin(LABELS)]
    if len(valid) == 0:
        continue
    _,_,f1,_ = precision_recall_fscore_support(valid['ground_truth'], valid['predicted_label'],
                                                labels=LABELS, average='macro', zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_m = m
print(best_m)
" 2>/dev/null || echo "-1.0")

  echo "  Best layer: $BEST_LAYER, Best multiplier: $BEST_MULT"

  for strategy in faiss_matched class_means; do
    sv_path=""
    if [[ "$strategy" == "faiss_matched" ]]; then
      sv_path="${SV_FAISS}/steering_vectors.pt"
    elif [[ "$strategy" == "class_means" ]]; then
      sv_path="${SV_MEANS}/steering_vectors.pt"
    fi

    if [[ ! -f "$sv_path" ]]; then
      echo "[SKIP] ${strategy} vectors not found: $sv_path"
      continue
    fi

    # Each strategy gets its own output dir to avoid filename collisions
    STRAT_OUT="${RESULTS}/strategy_${strategy}"
    PRED_FILE="${STRAT_OUT}/predictions/preds_caa_layer${BEST_LAYER}_m${BEST_MULT}.csv"
    if [[ -f "$PRED_FILE" ]]; then
      echo "[SKIP] ${strategy} already exists"
      continue
    fi

    echo ""
    echo ">>> Strategy: ${strategy}, layer=${BEST_LAYER}, m=${BEST_MULT}"
    $PYTHON activation_steering.py \
      --model "$MODEL" \
      --neighbors_csv "$NEIGHBORS_CSV" \
      --steering_vectors "$sv_path" \
      --output_dir "$STRAT_OUT" \
      --method caa \
      --layer "$BEST_LAYER" \
      --multiplier "$BEST_MULT" \
      --top_k "$TOP_K" \
      --max_seq_length "$CTX" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --inference_batch_size "$BATCH_SIZE" \
      --eval_dir "${STRAT_OUT}/evaluations" \
      $CACHE_FLAG
  done

  echo ""
  echo "  Phase 4 complete."
fi

# ============================================================================
# Phase 5: NTW Directional Ablation
# ============================================================================
if [[ $START_PHASE -le 5 ]]; then
  echo ""
  echo "============================================================"
  echo "  PHASE 5: NTW Directional Ablation"
  echo "============================================================"

  if [[ ! -f "$SWEEP_CSV" ]]; then
    echo "[ERROR] Layer sweep results not found. Run phase 2 first."
    exit 1
  fi

  BEST_LAYER=$(get_best_layer "$SWEEP_CSV")
  NTW_DIR="${RESULTS}/ntw_ablation"

  # --- Single best layer ablation (class_means vectors) ---
  PRED_FILE="${NTW_DIR}/predictions/preds_ablation_layer${BEST_LAYER}.csv"
  if [[ -f "$PRED_FILE" ]]; then
    echo "[SKIP] Single-layer ablation already exists"
  else
    echo ""
    echo ">>> NTW ablation: single layer=${BEST_LAYER} (class_means vectors)"
    $PYTHON activation_steering.py \
      --model "$MODEL" \
      --neighbors_csv "$NEIGHBORS_CSV" \
      --steering_vectors "${SV_MEANS}/steering_vectors.pt" \
      --output_dir "$NTW_DIR" \
      --method ablation \
      --layer "$BEST_LAYER" \
      --top_k "$TOP_K" \
      --max_seq_length "$CTX" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --inference_batch_size "$BATCH_SIZE" \
      --eval_dir "${NTW_DIR}/evaluations" \
      $CACHE_FLAG
  fi

  # --- Full ablation (all layers, class_means vectors) ---
  PRED_FILE="${NTW_DIR}/predictions/preds_ablation_layerall.csv"
  if [[ -f "$PRED_FILE" ]]; then
    echo "[SKIP] Full ablation already exists"
  else
    echo ""
    echo ">>> NTW ablation: all layers (class_means vectors)"
    $PYTHON activation_steering.py \
      --model "$MODEL" \
      --neighbors_csv "$NEIGHBORS_CSV" \
      --steering_vectors "${SV_MEANS}/steering_vectors.pt" \
      --output_dir "$NTW_DIR" \
      --method ablation \
      --layer all \
      --top_k "$TOP_K" \
      --max_seq_length "$CTX" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --inference_batch_size "$BATCH_SIZE" \
      --eval_dir "${NTW_DIR}/evaluations" \
      $CACHE_FLAG
  fi

  echo ""
  echo "  Phase 5 complete."
fi

# ============================================================================
# Phase 6: Summary Table
# ============================================================================
echo ""
echo "============================================================"
echo "  PHASE 6: Results Summary"
echo "============================================================"

$PYTHON - <<'PYEOF'
import pandas as pd
import os
import glob
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

LABELS = ['bug', 'feature', 'question']
RESULTS_ROOT = "results/issues3k_steering"

def eval_preds(path):
    """Evaluate a predictions CSV. Returns dict with metrics."""
    df = pd.read_csv(path)
    total = len(df)
    valid = df[df['predicted_label'].isin(LABELS)]
    n_valid = len(valid)
    if n_valid == 0:
        return None
    p, r, f1, _ = precision_recall_fscore_support(
        valid['ground_truth'], valid['predicted_label'],
        labels=LABELS, average=None, zero_division=0)
    f1_macro = f1.mean()
    inv_rate = (total - n_valid) / total
    return {
        'f1_macro': f1_macro,
        'f1_bug': f1[0], 'f1_feat': f1[1], 'f1_ques': f1[2],
        'r_bug': r[0], 'r_feat': r[1], 'r_ques': r[2],
        'p_bug': p[0], 'p_feat': p[1], 'p_ques': p[2],
        'inv%': inv_rate * 100,
    }

rows = []

# --- Baselines ---
# RAGTAG baseline (from existing results)
for bp in [
    'results/issues3k_ctx8192/unsloth_Llama_3_2_3B_Instruct/ragtag/predictions/preds_k3.csv',
    'results/issues3k_debias/unsloth_Llama_3_2_3B_Instruct/ragtag/predictions/preds_k3.csv',
]:
    if os.path.exists(bp):
        m = eval_preds(bp)
        if m:
            rows.append(('RAGTAG baseline k=3', m))
        break

# Debiased retrieval m3
for bp in [
    'results/issues3k_debias_m3/unsloth_Llama_3_2_3B_Instruct/ragtag/predictions/preds_k3.csv',
]:
    if os.path.exists(bp):
        m = eval_preds(bp)
        if m:
            rows.append(('Debias retrieval m3', m))

# --- Layer sweep: best result ---
sweep_csv = os.path.join(RESULTS_ROOT, 'layer_sweep_answer', 'layer_sweep_results.csv')
if os.path.exists(sweep_csv):
    sweep_df = pd.read_csv(sweep_csv)
    best = sweep_df.loc[sweep_df['f1_macro'].idxmax()]
    rows.append((f"CAA answer L{int(best['layer'])} m-1.0", {
        'f1_macro': best['f1_macro'],
        'f1_bug': best['f1_bug'], 'f1_feat': best['f1_feature'], 'f1_ques': best['f1_question'],
        'r_bug': best['r_bug'], 'r_feat': best['r_feature'], 'r_ques': best['r_question'],
        'p_bug': best['p_bug'], 'p_feat': best['p_feature'], 'p_ques': best['p_question'],
        'inv%': best['invalid_rate'],
    }))

# --- Multiplier sweep ---
mult_dir = os.path.join(RESULTS_ROOT, 'multiplier_sweep_answer', 'predictions')
if os.path.isdir(mult_dir):
    for f in sorted(glob.glob(os.path.join(mult_dir, 'preds_caa_*.csv'))):
        name = os.path.basename(f).replace('preds_caa_', '').replace('.csv', '')
        m = eval_preds(f)
        if m:
            rows.append((f'CAA answer {name}', m))

# --- Strategy comparison ---
for strategy in ['faiss_matched', 'class_means']:
    strat_dir = os.path.join(RESULTS_ROOT, f'strategy_{strategy}', 'predictions')
    if os.path.isdir(strat_dir):
        for f in sorted(glob.glob(os.path.join(strat_dir, 'preds_caa_*.csv'))):
            name = os.path.basename(f).replace('preds_caa_', '').replace('.csv', '')
            m = eval_preds(f)
            if m:
                rows.append((f'CAA {strategy} {name}', m))

# --- NTW ablation ---
ntw_dir = os.path.join(RESULTS_ROOT, 'ntw_ablation', 'predictions')
if os.path.isdir(ntw_dir):
    for f in sorted(glob.glob(os.path.join(ntw_dir, 'preds_ablation_*.csv'))):
        name = os.path.basename(f).replace('preds_ablation_', '').replace('.csv', '')
        m = eval_preds(f)
        if m:
            rows.append((f'NTW ablation {name}', m))

# --- Print table ---
if rows:
    hdr = f"{'Method':<30} {'F1_mac':>6} {'F1_bug':>6} {'F1_ft':>5} {'F1_qu':>5} | {'R_bug':>5} {'R_qu':>5} | {'P_bug':>5} {'P_qu':>5} | {'Inv%':>4}"
    print(hdr)
    print('-' * len(hdr))
    for name, m in rows:
        print(f"{name:<30} {m['f1_macro']:6.4f} {m['f1_bug']:6.3f} {m['f1_feat']:5.3f} {m['f1_ques']:5.3f}"
              f" | {m['r_bug']:5.3f} {m['r_ques']:5.3f}"
              f" | {m['p_bug']:5.3f} {m['p_ques']:5.3f}"
              f" | {m['inv%']:4.1f}")
else:
    print("  No results found yet. Run the experiment first.")
PYEOF

echo ""
echo "============================================================"
echo "  Done!"
echo "============================================================"

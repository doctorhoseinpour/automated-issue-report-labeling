#!/usr/bin/env bash
# ============================================================================
# run_openai_ragtag.sh — canary-first OpenAI RAGTAG orchestrator
# ============================================================================
# Runs RAGTAG inference on the 11-project benchmark using an OpenAI model
# instead of Qwen, to remove the model-diversity threat to validity.
#
# Project-specific (PS) RAGTAG only, K = {12, 15}. A 30-issue canary runs first
# and the pipeline STOPS for cost review unless --full is passed.
#
# Usage:
#   ./run_openai_ragtag.sh --model gpt-3.5-turbo            # canary only
#   ./run_openai_ragtag.sh --model gpt-3.5-turbo --full     # canary + full PS
#   ./run_openai_ragtag.sh --model gpt-4o --full
#
# Requires:  OPENAI_API_KEY exported;  venv with openai + tiktoken.
# Never modifies existing files; never overwrites anything under results/.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

MODEL="gpt-3.5-turbo"
KS="12,15"
CANARY_KS="12"
CANARY_N=30
CTX=8192
RUN_FULL=0
PY="venv/bin/python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)     MODEL="$2"; shift 2 ;;
    --full)      RUN_FULL=1; shift ;;
    --ks)        KS="$2"; shift 2 ;;
    --canary_n)  CANARY_N="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# model tag: openai_<model with - and . replaced by _>
MODEL_TAG="openai_$(echo "$MODEL" | tr '.-' '__')"
RESULTS="results/issues11k/project_specific"
CANARY_ROOT="canary_openai/${MODEL_TAG}"

PROJECTS=(
  ansible_ansible bitcoin_bitcoin dart-lang_sdk dotnet_roslyn
  facebook_react flutter_flutter kubernetes_kubernetes
  microsoft_TypeScript microsoft_vscode opencv_opencv tensorflow_tensorflow
)

echo "============================================================"
echo "  OpenAI RAGTAG  |  model=$MODEL  tag=$MODEL_TAG  K=$KS"
echo "  mode: $([[ $RUN_FULL -eq 1 ]] && echo 'CANARY + FULL PS' || echo 'CANARY ONLY')"
echo "============================================================"

# --- Preflight ---------------------------------------------------------------
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-..." >&2
  exit 1
fi
if ! $PY -c "import openai, tiktoken" 2>/dev/null; then
  echo "ERROR: openai/tiktoken not installed in venv." >&2
  echo "  Run: $PY -m pip install -r requirements-openai.txt" >&2
  exit 1
fi

# --- Phase A: Canary ---------------------------------------------------------
CANARY_PROJECT="${PROJECTS[0]}"
CANARY_PRED_DIR="${CANARY_ROOT}/${CANARY_PROJECT}/ragtag/predictions"
echo
echo ">>> Phase A — Canary: $CANARY_N issues, k=$CANARY_KS, project=$CANARY_PROJECT"
$PY openai_labeler.py \
  --model "$MODEL" \
  --canary "$CANARY_N" \
  --neighbors_dir "${RESULTS}/${CANARY_PROJECT}/neighbors" \
  --top_ks "$CANARY_KS" \
  --output_dir "$CANARY_PRED_DIR" \
  --eval_dir "${CANARY_ROOT}/${CANARY_PROJECT}/ragtag/evaluations" \
  --log_dir "${CANARY_ROOT}/${CANARY_PROJECT}/ragtag/logs" \
  --model_name_for_eval "$MODEL" \
  --max_seq_length "$CTX"

echo
echo ">>> Canary cost summary:"
cat "${CANARY_PRED_DIR}/cost_metrics.csv" 2>/dev/null || echo "  (no cost_metrics.csv)"

if [[ $RUN_FULL -ne 1 ]]; then
  echo
  echo "============================================================"
  echo "  Canary complete. Review the cost above, then re-run with"
  echo "  --full to launch the full project-specific sweep."
  echo "============================================================"
  exit 0
fi

# --- Phase B: Full project-specific sweep ------------------------------------
echo
echo ">>> Phase B — Full PS sweep: ${#PROJECTS[@]} projects x K={$KS}"
for proj in "${PROJECTS[@]}"; do
  PRED_DIR="${RESULTS}/${proj}/${MODEL_TAG}/ragtag/predictions"
  EVAL_DIR="${RESULTS}/${proj}/${MODEL_TAG}/ragtag/evaluations"
  LOG_DIR="${RESULTS}/${proj}/${MODEL_TAG}/ragtag/logs"
  echo
  echo "--- Project: $proj ---"
  $PY openai_labeler.py \
    --model "$MODEL" \
    --neighbors_dir "${RESULTS}/${proj}/neighbors" \
    --top_ks "$KS" \
    --output_dir "$PRED_DIR" \
    --eval_dir "$EVAL_DIR" \
    --log_dir "$LOG_DIR" \
    --model_name_for_eval "$MODEL" \
    --max_seq_length "$CTX"
done

# --- Final aggregate cost ----------------------------------------------------
echo
echo ">>> Aggregate cost across all projects:"
$PY - "$RESULTS" "$MODEL_TAG" <<'PYEOF'
import sys, glob, os
import pandas as pd
results, tag = sys.argv[1], sys.argv[2]
files = glob.glob(os.path.join(results, "*", tag, "ragtag", "predictions", "cost_metrics.csv"))
if not files:
    print("  (no cost_metrics.csv found)")
else:
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"  cells: {len(df)}  total issues: {df['total_issues'].sum()}")
    print(f"  total cost: ${df['total_cost_usd'].sum():.2f}")
PYEOF

echo
echo "============================================================"
echo "  Done. Predictions + evaluations under:"
echo "  ${RESULTS}/<project>/${MODEL_TAG}/ragtag/"
echo "============================================================"

#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_data_efficiency.sh — Data efficiency crossover experiment
# ============================================================================
# Measures RAGTAG and FT performance as a function of training pool size on
# the 30k dataset. Subsample sizes: 1.5k, 3k, 9k, 15k (the 27k endpoint
# already exists in results/issues30k/ and is reused for the summary).
#
# Usage:
#   LOCAL  (RTX 4090):  bash run_data_efficiency.sh --mode local
#   REMOTE (A100):      bash run_data_efficiency.sh --mode remote --nrp
#
# --mode local:  Phase 0 (subsample+FAISS), Phase 1 (all RAGTAG), Phase 2 (Llama FT)
# --mode remote: Phase 0 (subsample only),  Phase 2 (Qwen FT)
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

# --- Defaults ---
PYTHON="${PYTHON_BIN:-venv/bin/python}"
MODE="local"
SKIP_INDEXING=0
SKIP_RAGTAG=0
SKIP_FT=0
NRP=0
MODEL_CACHE_DIR=""
MAX_NEW_TOKENS=50
BATCH_SIZE=1
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# --- Parse CLI flags ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)          MODE="$2";           shift 2 ;;
        --skip_indexing)  SKIP_INDEXING=1;     shift ;;
        --skip_ragtag)    SKIP_RAGTAG=1;      shift ;;
        --skip_ft)        SKIP_FT=1;          shift ;;
        --nrp)            NRP=1;              shift ;;
        *)  echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" != "local" && "$MODE" != "remote" ]]; then
    echo "ERROR: --mode must be 'local' or 'remote'"
    exit 1
fi

if [[ "$NRP" -eq 1 ]]; then
    MODEL_CACHE_DIR="hf_cache"
    mkdir -p "$MODEL_CACHE_DIR"
    export HF_HOME="$MODEL_CACHE_DIR"
    export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR"
fi

# ============================================================================
# Configuration
# ============================================================================

SUBSAMPLE_SIZES=(1500 3000 9000 15000)
EFF_DIR="results/issues30k_efficiency"
FULL_NB="results/issues30k/neighbors"
DATASET="issues30k.csv"
TEST_SIZE=3000

# Model configs: MODEL_ID | TAG | BEST_K | CTX
# Best RAGTAG configs from 3k development study
declare -a ALL_MODELS=(
    "unsloth/Llama-3.2-3B-Instruct"
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
)
declare -a ALL_TAGS=(
    "unsloth_Llama_3_2_3B_Instruct"
    "unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit"
    "unsloth_Qwen2_5_14B_Instruct_bnb_4bit"
    "unsloth_Qwen2_5_32B_Instruct_bnb_4bit"
)
declare -a ALL_BEST_K=(3 9 9 3)
declare -a ALL_CTX=(8192 8192 8192 8192)
declare -a ALL_SHORT=(Llama-3B Llama-8B Qwen-14B Qwen-32B)

# FT models by mode
LOCAL_FT_INDICES=(0 1)    # Llama-3B, Llama-8B
REMOTE_FT_INDICES=(2 3)   # Qwen-14B, Qwen-32B

# Union of all best_k values needed for neighbors
NEEDED_KS="3,9"

echo "============================================================"
echo "  Data Efficiency Crossover Experiment"
echo "  Mode:             ${MODE}"
echo "  Subsample sizes:  ${SUBSAMPLE_SIZES[*]}"
echo "  Results dir:      ${EFF_DIR}"
echo "  NRP mode:         ${NRP}"
echo "============================================================"

# ============================================================================
# Phase 0: Generate subsampled splits + FAISS indices
# ============================================================================

if [[ "$SKIP_INDEXING" -eq 0 ]]; then
    echo -e "\n[Phase 0] Generating subsampled training sets"

    # Ensure base 30k splits exist (critical for remote server)
    TRAIN_CSV="${FULL_NB}/train_split.csv"
    TEST_CSV="${FULL_NB}/test_split.csv"

    if [[ ! -f "$TRAIN_CSV" || ! -f "$TEST_CSV" ]]; then
        echo "  Base 30k splits not found — regenerating from ${DATASET}..."
        CACHE_ARG=""
        if [[ -n "$MODEL_CACHE_DIR" ]]; then
            CACHE_ARG="--model_cache_dir $MODEL_CACHE_DIR"
        fi
        $PYTHON build_and_query_index.py \
            --dataset "$DATASET" \
            --output_dir "$FULL_NB" \
            --top_ks "$NEEDED_KS" \
            --test_size "$TEST_SIZE" \
            --embedding_model "$EMBEDDING_MODEL" \
            $CACHE_ARG
    fi

    SUBSAMPLE_ARGS=(
        --train_csv "$TRAIN_CSV"
        --test_csv "$TEST_CSV"
        --sizes "$(IFS=,; echo "${SUBSAMPLE_SIZES[*]}")"
        --output_dir "$EFF_DIR"
        --seed 42
    )

    if [[ "$MODE" == "remote" ]]; then
        # Remote only needs subsampled train CSVs for FT (no FAISS needed)
        SUBSAMPLE_ARGS+=(--skip_indexing)
    else
        SUBSAMPLE_ARGS+=(
            --top_ks "$NEEDED_KS"
            --embedding_model "$EMBEDDING_MODEL"
        )
    fi

    if [[ -n "$MODEL_CACHE_DIR" ]]; then
        SUBSAMPLE_ARGS+=(--model_cache_dir "$MODEL_CACHE_DIR")
    fi

    $PYTHON subsample_and_index.py "${SUBSAMPLE_ARGS[@]}"
else
    echo -e "\n[Phase 0] SKIPPED (--skip_indexing)"
fi

# ============================================================================
# Phase 1: RAGTAG inference (LOCAL only)
# ============================================================================

if [[ "$MODE" == "local" && "$SKIP_RAGTAG" -eq 0 ]]; then
    echo -e "\n[Phase 1] RAGTAG inference (all 4 models)"

    CACHE_ARGS=()
    if [[ -n "$MODEL_CACHE_DIR" ]]; then
        CACHE_ARGS=(--cache_dir "$MODEL_CACHE_DIR")
    fi

    # Loop by model (outer) to group invocations per model
    for mi in "${!ALL_MODELS[@]}"; do
        MODEL="${ALL_MODELS[$mi]}"
        TAG="${ALL_TAGS[$mi]}"
        BEST_K="${ALL_BEST_K[$mi]}"
        CTX="${ALL_CTX[$mi]}"
        SHORT="${ALL_SHORT[$mi]}"

        echo -e "\n--- ${SHORT} (k=${BEST_K}, ctx=${CTX}) ---"

        for SIZE in "${SUBSAMPLE_SIZES[@]}"; do
            NB_DIR="${EFF_DIR}/n${SIZE}"
            PRED_DIR="${EFF_DIR}/n${SIZE}/${TAG}/ragtag/predictions"
            PRED_FILE="${PRED_DIR}/preds_k${BEST_K}.csv"

            if [[ -f "$PRED_FILE" ]]; then
                echo "  SKIP n=${SIZE}: ${PRED_FILE} exists"
                continue
            fi

            echo -e "\n  Running RAGTAG: ${SHORT} on n=${SIZE}"
            mkdir -p "$PRED_DIR"

            $PYTHON llm_labeler.py \
                --model "$MODEL" \
                --neighbors_dir "$NB_DIR" \
                --top_ks "$BEST_K" \
                --output_dir "$PRED_DIR" \
                --max_seq_length "$CTX" \
                --max_new_tokens "$MAX_NEW_TOKENS" \
                --inference_batch_size "$BATCH_SIZE" \
                --load_in_4bit \
                "${CACHE_ARGS[@]}"
        done
    done
else
    if [[ "$MODE" == "remote" ]]; then
        echo -e "\n[Phase 1] SKIPPED (remote mode — RAGTAG runs locally)"
    else
        echo -e "\n[Phase 1] SKIPPED (--skip_ragtag)"
    fi
fi

# ============================================================================
# Phase 2: Fine-tuning
# ============================================================================

if [[ "$SKIP_FT" -eq 0 ]]; then
    echo -e "\n[Phase 2] Fine-tuning"

    # Select model indices based on mode
    if [[ "$MODE" == "local" ]]; then
        FT_INDICES=("${LOCAL_FT_INDICES[@]}")
        echo "  Mode: local — training Llama-3B and Llama-8B"
    else
        FT_INDICES=("${REMOTE_FT_INDICES[@]}")
        echo "  Mode: remote — training Qwen-14B and Qwen-32B"
    fi

    CACHE_ARGS=()
    if [[ -n "$MODEL_CACHE_DIR" ]]; then
        CACHE_ARGS=(--cache_dir "$MODEL_CACHE_DIR")
    fi

    for mi in "${FT_INDICES[@]}"; do
        MODEL="${ALL_MODELS[$mi]}"
        TAG="${ALL_TAGS[$mi]}"
        SHORT="${ALL_SHORT[$mi]}"

        echo -e "\n--- ${SHORT} fine-tuning ---"

        for SIZE in "${SUBSAMPLE_SIZES[@]}"; do
            SUB_DIR="${EFF_DIR}/n${SIZE}"
            FT_DIR="${EFF_DIR}/n${SIZE}/${TAG}/finetune_fixed"
            FT_PREDS="${FT_DIR}/preds_finetune_fixed.csv"

            if [[ -f "$FT_PREDS" ]]; then
                echo "  SKIP n=${SIZE}: ${FT_PREDS} exists"
                continue
            fi

            echo -e "\n  Training FT: ${SHORT} on n=${SIZE}"
            mkdir -p "$FT_DIR"

            $PYTHON fixed_fine-tune.py \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --train_csv "${SUB_DIR}/train_split.csv" \
                --test_csv "${SUB_DIR}/test_split.csv" \
                --max_seq_length 2048 \
                --max_new_tokens "$MAX_NEW_TOKENS" \
                --inference_batch_size "$BATCH_SIZE" \
                --output_dir "$FT_DIR" \
                "${CACHE_ARGS[@]}"
        done
    done
else
    echo -e "\n[Phase 2] SKIPPED (--skip_ft)"
fi

# ============================================================================
# Phase 3: Evaluation
# ============================================================================

echo -e "\n[Phase 3] Evaluating all predictions"

for SIZE in "${SUBSAMPLE_SIZES[@]}"; do
    for mi in "${!ALL_MODELS[@]}"; do
        TAG="${ALL_TAGS[$mi]}"
        SHORT="${ALL_SHORT[$mi]}"

        # RAGTAG eval
        PRED_DIR="${EFF_DIR}/n${SIZE}/${TAG}/ragtag/predictions"
        EVAL_DIR="${EFF_DIR}/n${SIZE}/${TAG}/ragtag/evaluations"
        for PRED_FILE in "$PRED_DIR"/preds_*.csv; do
            [ -f "$PRED_FILE" ] || continue
            mkdir -p "$EVAL_DIR"
            BASENAME=$(basename "$PRED_FILE" .csv)
            K_TAG="${BASENAME#preds_}"
            EVAL_FILE="${EVAL_DIR}/eval_${K_TAG}.csv"
            if [[ -f "$EVAL_FILE" ]]; then
                continue
            fi
            echo "  Evaluating: ${SHORT} RAGTAG n=${SIZE} ${K_TAG}"
            $PYTHON evaluate.py --preds_csv "$PRED_FILE" --output_csv "$EVAL_FILE"
        done

        # FT eval
        FT_PREDS="${EFF_DIR}/n${SIZE}/${TAG}/finetune_fixed/preds_finetune_fixed.csv"
        FT_EVAL="${EFF_DIR}/n${SIZE}/${TAG}/finetune_fixed/eval_finetune_fixed.csv"
        if [[ -f "$FT_PREDS" && ! -f "$FT_EVAL" ]]; then
            echo "  Evaluating: ${SHORT} FT n=${SIZE}"
            $PYTHON evaluate.py --preds_csv "$FT_PREDS" --output_csv "$FT_EVAL"
        fi
    done
done

# ============================================================================
# Summary Report
# ============================================================================

echo -e "\n============================================================"
echo "  Data Efficiency Crossover — Summary"
echo "============================================================"

$PYTHON -c "
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os

LABELS = ['bug', 'feature', 'question']

def eval_preds(path):
    \"\"\"Compute macro-F1 and accuracy from a predictions CSV.\"\"\"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    valid = df[df['predicted_label'].isin(LABELS)]
    if len(valid) == 0:
        return None
    _, _, f1m, _ = precision_recall_fscore_support(
        valid['ground_truth'], valid['predicted_label'],
        labels=LABELS, average='macro', zero_division=0)
    acc = accuracy_score(valid['ground_truth'], valid['predicted_label'])
    inv = (~df['predicted_label'].isin(LABELS)).mean()
    return f1m, acc, inv

sizes = [1500, 3000, 9000, 15000]
models = [
    ('Llama-3B',  'unsloth_Llama_3_2_3B_Instruct',                 3),
    ('Llama-8B',  'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit',   9),
    ('Qwen-14B',  'unsloth_Qwen2_5_14B_Instruct_bnb_4bit',         9),
    ('Qwen-32B',  'unsloth_Qwen2_5_32B_Instruct_bnb_4bit',         3),
]

# Header
size_cols = ''.join(f'  n={s:<6}' for s in sizes) + '  n=27k  '
print(f\"{'Model':>10} {'Method':>7} {size_cols}\")
print('-' * (22 + 10 * (len(sizes) + 1)))

for short, tag, best_k in models:
    for method in ['RAGTAG', 'FT']:
        vals = []
        for s in sizes:
            if method == 'RAGTAG':
                path = f'results/issues30k_efficiency/n{s}/{tag}/ragtag/predictions/preds_k{best_k}.csv'
            else:
                path = f'results/issues30k_efficiency/n{s}/{tag}/finetune_fixed/preds_finetune_fixed.csv'
            r = eval_preds(path)
            vals.append(f'{r[0]:.4f}' if r else '  --  ')

        # 27k endpoint from existing results
        if method == 'RAGTAG':
            path_27k = f'results/issues30k/{tag}/ragtag/predictions/preds_k{best_k}.csv'
        else:
            path_27k = f'results/issues30k/{tag}/finetune_fixed/preds_finetune_fixed.csv'
        r27 = eval_preds(path_27k)
        vals.append(f'{r27[0]:.4f}' if r27 else '  --  ')

        row = ''.join(f'  {v:<8}' for v in vals)
        print(f'{short:>10} {method:>7} {row}')
    print()
"

echo -e "\nDone!"

#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_11k_experiments.sh — 11-Project Benchmark Experiment Pipeline
# ============================================================================
# Compares Zero-shot, VTAG, RAGTAG, and Fine-tuning on the 11-project
# benchmark (6,600 issues, stratified temporal split) across two settings:
#   - Project-Agnostic:  train/retrieve from all 3,300 train issues
#   - Project-Specific:  per project, 300 train → 300 test
#
# Usage:
#   LOCAL  (RTX 4090):  bash run_11k_experiments.sh --mode local
#   REMOTE (A100):      bash run_11k_experiments.sh --mode remote --nrp
#
# Step-by-step (skip already-done phases):
#   bash run_11k_experiments.sh --mode local --skip_indexing --skip_vtag
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

# --- Defaults ---
PYTHON="${PYTHON_BIN:-venv/bin/python}"
MODE="local"
SETTING="both"
SKIP_INDEXING=0
SKIP_ZERO_SHOT=0
SKIP_RAGTAG=0
SKIP_FT=0
SKIP_VTAG=0
SKIP_EVAL=0
NRP=0
MODEL_CACHE_DIR=""
MAX_NEW_TOKENS=50
BATCH_SIZE=1
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# --- Parse CLI flags ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)           MODE="$2";            shift 2 ;;
        --setting)        SETTING="$2";         shift 2 ;;
        --skip_indexing)   SKIP_INDEXING=1;      shift ;;
        --skip_zero_shot)  SKIP_ZERO_SHOT=1;    shift ;;
        --skip_ragtag)     SKIP_RAGTAG=1;       shift ;;
        --skip_ft)         SKIP_FT=1;           shift ;;
        --skip_vtag)       SKIP_VTAG=1;         shift ;;
        --skip_eval)       SKIP_EVAL=1;         shift ;;
        --nrp)             NRP=1;               shift ;;
        *)  echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" != "local" && "$MODE" != "remote" ]]; then
    echo "ERROR: --mode must be 'local' or 'remote'"
    exit 1
fi

if [[ "$SETTING" != "agnostic" && "$SETTING" != "specific" && "$SETTING" != "both" ]]; then
    echo "ERROR: --setting must be 'agnostic', 'specific', or 'both'"
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

TRAIN_CSV="issues11k_train.csv"
TEST_CSV="issues11k_test.csv"
DATASET="issues11k.csv"
RESULTS="results/issues11k"
CTX=8192

# k values justified by VTAG plateau analysis on this dataset
RAGTAG_KS="1,3,6,9"

declare -a ALL_MODELS=(
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
)
declare -a ALL_TAGS=(
    "unsloth_Qwen2_5_3B_Instruct_bnb_4bit"
    "unsloth_Qwen2_5_7B_Instruct_bnb_4bit"
    "unsloth_Qwen2_5_14B_Instruct_bnb_4bit"
    "unsloth_Qwen2_5_32B_Instruct_bnb_4bit"
)
declare -a ALL_SHORT=("Qwen-3B" "Qwen-7B" "Qwen-14B" "Qwen-32B")

# FT models by mode
LOCAL_FT_INDICES=(0 1)    # Qwen-3B, Qwen-7B
REMOTE_FT_INDICES=(2 3)   # Qwen-14B, Qwen-32B

declare -a PROJECTS=(
    "ansible/ansible"
    "bitcoin/bitcoin"
    "dart-lang/sdk"
    "dotnet/roslyn"
    "facebook/react"
    "flutter/flutter"
    "kubernetes/kubernetes"
    "microsoft/TypeScript"
    "microsoft/vscode"
    "opencv/opencv"
    "tensorflow/tensorflow"
)
declare -a PROJECT_TAGS=(
    "ansible_ansible"
    "bitcoin_bitcoin"
    "dart-lang_sdk"
    "dotnet_roslyn"
    "facebook_react"
    "flutter_flutter"
    "kubernetes_kubernetes"
    "microsoft_TypeScript"
    "microsoft_vscode"
    "opencv_opencv"
    "tensorflow_tensorflow"
)

# Helper: should we run agnostic / specific?
run_agnostic() { [[ "$SETTING" == "agnostic" || "$SETTING" == "both" ]]; }
run_specific() { [[ "$SETTING" == "specific" || "$SETTING" == "both" ]]; }

echo "============================================================"
echo "  11-Project Benchmark Experiment Pipeline"
echo "  Mode:      ${MODE}"
echo "  Setting:   ${SETTING}"
echo "  RAGTAG ks: ${RAGTAG_KS}"
echo "  CTX:       ${CTX}"
echo "  NRP mode:  ${NRP}"
echo "============================================================"

# ============================================================================
# Phase 0a: Ensure train/test split CSVs exist (no GPU needed)
# ============================================================================
# Split CSVs are needed by FT even when FAISS indexing is skipped.
# This step generates them from the source CSVs if they don't exist.

echo -e "\n[Phase 0a] Ensuring train/test split CSVs exist"

$PYTHON -c "
import pandas as pd, os

TRAIN_CSV = '$TRAIN_CSV'
TEST_CSV  = '$TEST_CSV'
RESULTS   = '$RESULTS'
SETTING   = '$SETTING'

PROJECTS = [
    ('ansible/ansible', 'ansible_ansible'),
    ('bitcoin/bitcoin', 'bitcoin_bitcoin'),
    ('dart-lang/sdk', 'dart-lang_sdk'),
    ('dotnet/roslyn', 'dotnet_roslyn'),
    ('facebook/react', 'facebook_react'),
    ('flutter/flutter', 'flutter_flutter'),
    ('kubernetes/kubernetes', 'kubernetes_kubernetes'),
    ('microsoft/TypeScript', 'microsoft_TypeScript'),
    ('microsoft/vscode', 'microsoft_vscode'),
    ('opencv/opencv', 'opencv_opencv'),
    ('tensorflow/tensorflow', 'tensorflow_tensorflow'),
]

def normalize(df):
    df['body'] = df['body'].fillna('')
    df['title'] = df['title'].fillna('')
    if 'label' in df.columns and 'labels' not in df.columns:
        df.rename(columns={'label': 'labels'}, inplace=True)
    df['labels'] = df['labels'].astype(str).str.lower().str.strip()
    return df

save_cols = ['repo', 'created_at', 'labels', 'title', 'body']

train_df = normalize(pd.read_csv(TRAIN_CSV))
test_df  = normalize(pd.read_csv(TEST_CSV))
save_cols = [c for c in save_cols if c in test_df.columns]

# Agnostic splits
if SETTING in ('agnostic', 'both'):
    ag_dir = os.path.join(RESULTS, 'agnostic/neighbors')
    train_path = os.path.join(ag_dir, 'train_split.csv')
    test_path  = os.path.join(ag_dir, 'test_split.csv')
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        os.makedirs(ag_dir, exist_ok=True)
        train_df[save_cols].to_csv(train_path, index=False)
        test_df[save_cols].to_csv(test_path, index=False)
        print(f'  Wrote agnostic splits: {len(train_df)} train, {len(test_df)} test')
    else:
        print(f'  Agnostic splits already exist')

# Project-specific splits
if SETTING in ('specific', 'both'):
    for repo_name, proj_tag in PROJECTS:
        ps_dir = os.path.join(RESULTS, f'project_specific/{proj_tag}/neighbors')
        train_path = os.path.join(ps_dir, 'train_split.csv')
        test_path  = os.path.join(ps_dir, 'test_split.csv')
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            os.makedirs(ps_dir, exist_ok=True)
            ptr = train_df[train_df['repo'] == repo_name].reset_index(drop=True)
            pte = test_df[test_df['repo'] == repo_name].reset_index(drop=True)
            ptr[save_cols].to_csv(train_path, index=False)
            pte[save_cols].to_csv(test_path, index=False)
            print(f'  Wrote {repo_name} splits: {len(ptr)} train, {len(pte)} test')
        else:
            print(f'  {repo_name} splits already exist')
"

# ============================================================================
# Phase 0b: Build FAISS indexes (requires GPU)
# ============================================================================

if [[ "$SKIP_INDEXING" -eq 0 ]]; then
    echo -e "\n[Phase 0b] Building FAISS indexes"

    CACHE_ARG=""
    if [[ -n "$MODEL_CACHE_DIR" ]]; then
        CACHE_ARG="--model_cache_dir $MODEL_CACHE_DIR"
    fi

    # Agnostic index
    if run_agnostic; then
        echo -e "\n  --- Agnostic index (3300 train) ---"
        $PYTHON build_11k_index.py \
            --train_csv "$TRAIN_CSV" --test_csv "$TEST_CSV" \
            --top_ks "3,9,30" \
            --output_dir "$RESULTS/agnostic/neighbors" \
            --embedding_model "$EMBEDDING_MODEL" \
            $CACHE_ARG
    fi

    # Per-project indexes
    if run_specific; then
        for i in "${!PROJECTS[@]}"; do
            echo -e "\n  --- Project-specific: ${PROJECTS[$i]} (300 train) ---"
            $PYTHON build_11k_index.py \
                --train_csv "$TRAIN_CSV" --test_csv "$TEST_CSV" \
                --repo_filter "${PROJECTS[$i]}" \
                --top_ks "3,9,30" \
                --output_dir "$RESULTS/project_specific/${PROJECT_TAGS[$i]}/neighbors" \
                --embedding_model "$EMBEDDING_MODEL" \
                $CACHE_ARG
        done
    fi
else
    echo -e "\n[Phase 0b] SKIPPED (--skip_indexing)"
fi

# ============================================================================
# Phase 1: Zero-shot (one run per model on agnostic, covers both settings)
# ============================================================================

if [[ "$SKIP_ZERO_SHOT" -eq 0 ]]; then
    echo -e "\n[Phase 1] Zero-shot inference"

    CACHE_ARGS=()
    if [[ -n "$MODEL_CACHE_DIR" ]]; then
        CACHE_ARGS=(--cache_dir "$MODEL_CACHE_DIR")
    fi

    for i in "${!ALL_MODELS[@]}"; do
        MODEL="${ALL_MODELS[$i]}"
        TAG="${ALL_TAGS[$i]}"
        SHORT="${ALL_SHORT[$i]}"
        PRED_DIR="$RESULTS/agnostic/${TAG}/ragtag/predictions"

        if [[ -f "$PRED_DIR/preds_zero_shot.csv" ]]; then
            echo "  SKIP ${SHORT}: preds_zero_shot.csv exists"
            continue
        fi

        echo -e "\n  --- ${SHORT} zero-shot (3300 issues) ---"
        mkdir -p "$PRED_DIR"

        $PYTHON llm_labeler.py \
            --model "$MODEL" \
            --neighbors_dir "$RESULTS/agnostic/neighbors" \
            --top_ks "0" \
            --output_dir "$PRED_DIR" \
            --max_seq_length "$CTX" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --inference_batch_size "$BATCH_SIZE" \
            "${CACHE_ARGS[@]}"
    done
else
    echo -e "\n[Phase 1] SKIPPED (--skip_zero_shot)"
fi

# ============================================================================
# Phase 2: RAGTAG (k=1,3,6,9)
# ============================================================================

if [[ "$SKIP_RAGTAG" -eq 0 ]]; then
    echo -e "\n[Phase 2] RAGTAG inference (k=${RAGTAG_KS})"

    CACHE_ARGS=()
    if [[ -n "$MODEL_CACHE_DIR" ]]; then
        CACHE_ARGS=(--cache_dir "$MODEL_CACHE_DIR")
    fi

    for i in "${!ALL_MODELS[@]}"; do
        MODEL="${ALL_MODELS[$i]}"
        TAG="${ALL_TAGS[$i]}"
        SHORT="${ALL_SHORT[$i]}"

        # 2a: Agnostic
        if run_agnostic; then
            PRED_DIR="$RESULTS/agnostic/${TAG}/ragtag/predictions"
            echo -e "\n  --- ${SHORT} RAGTAG agnostic (3300 issues, k=${RAGTAG_KS}) ---"
            mkdir -p "$PRED_DIR"

            $PYTHON llm_labeler.py \
                --model "$MODEL" \
                --neighbors_dir "$RESULTS/agnostic/neighbors" \
                --top_ks "$RAGTAG_KS" \
                --output_dir "$PRED_DIR" \
                --max_seq_length "$CTX" \
                --max_new_tokens "$MAX_NEW_TOKENS" \
                --inference_batch_size "$BATCH_SIZE" \
                "${CACHE_ARGS[@]}"
        fi

        # 2b: Project-specific
        if run_specific; then
            for j in "${!PROJECTS[@]}"; do
                PRED_DIR="$RESULTS/project_specific/${PROJECT_TAGS[$j]}/${TAG}/ragtag/predictions"
                echo -e "\n  --- ${SHORT} RAGTAG ${PROJECTS[$j]} (300 issues, k=${RAGTAG_KS}) ---"
                mkdir -p "$PRED_DIR"

                $PYTHON llm_labeler.py \
                    --model "$MODEL" \
                    --neighbors_dir "$RESULTS/project_specific/${PROJECT_TAGS[$j]}/neighbors" \
                    --top_ks "$RAGTAG_KS" \
                    --output_dir "$PRED_DIR" \
                    --max_seq_length "$CTX" \
                    --max_new_tokens "$MAX_NEW_TOKENS" \
                    --inference_batch_size "$BATCH_SIZE" \
                    "${CACHE_ARGS[@]}"
            done
        fi
    done
else
    echo -e "\n[Phase 2] SKIPPED (--skip_ragtag)"
fi

# ============================================================================
# Phase 3: Fine-tuning
# ============================================================================

if [[ "$SKIP_FT" -eq 0 ]]; then
    echo -e "\n[Phase 3] Fine-tuning"

    if [[ "$MODE" == "local" ]]; then
        FT_INDICES=("${LOCAL_FT_INDICES[@]}")
        echo "  Mode: local — training Qwen-3B and Qwen-7B"
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

        # 3a: Agnostic
        if run_agnostic; then
            FT_DIR="$RESULTS/agnostic/${TAG}/finetune_fixed"
            FT_PREDS="${FT_DIR}/preds_finetune_fixed.csv"

            if [[ -f "$FT_PREDS" ]]; then
                echo "  SKIP ${SHORT} agnostic FT: ${FT_PREDS} exists"
            else
                echo -e "\n  --- ${SHORT} FT agnostic (3300 train) ---"
                mkdir -p "$FT_DIR"

                $PYTHON fixed_fine-tune.py \
                    --model "$MODEL" \
                    --dataset "$DATASET" \
                    --train_csv "$RESULTS/agnostic/neighbors/train_split.csv" \
                    --test_csv "$RESULTS/agnostic/neighbors/test_split.csv" \
                    --max_seq_length 2048 \
                    --max_new_tokens "$MAX_NEW_TOKENS" \
                    --inference_batch_size "$BATCH_SIZE" \
                    --output_dir "$FT_DIR" \
                    "${CACHE_ARGS[@]}"
            fi
        fi

        # 3b: Project-specific
        if run_specific; then
            for j in "${!PROJECTS[@]}"; do
                FT_DIR="$RESULTS/project_specific/${PROJECT_TAGS[$j]}/${TAG}/finetune_fixed"
                FT_PREDS="${FT_DIR}/preds_finetune_fixed.csv"

                if [[ -f "$FT_PREDS" ]]; then
                    echo "  SKIP ${SHORT} ${PROJECTS[$j]} FT: exists"
                    continue
                fi

                echo -e "\n  --- ${SHORT} FT ${PROJECTS[$j]} (300 train) ---"
                mkdir -p "$FT_DIR"

                $PYTHON fixed_fine-tune.py \
                    --model "$MODEL" \
                    --dataset "$DATASET" \
                    --train_csv "$RESULTS/project_specific/${PROJECT_TAGS[$j]}/neighbors/train_split.csv" \
                    --test_csv "$RESULTS/project_specific/${PROJECT_TAGS[$j]}/neighbors/test_split.csv" \
                    --max_seq_length 2048 \
                    --max_new_tokens "$MAX_NEW_TOKENS" \
                    --inference_batch_size "$BATCH_SIZE" \
                    --output_dir "$FT_DIR" \
                    "${CACHE_ARGS[@]}"
            done
        fi
    done
else
    echo -e "\n[Phase 3] SKIPPED (--skip_ft)"
fi

# ============================================================================
# Phase 4: VTAG (no GPU needed)
# ============================================================================

if [[ "$SKIP_VTAG" -eq 0 ]]; then
    echo -e "\n[Phase 4] VTAG voting baseline"

    VTAG_KS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30"

    # 4a: Agnostic
    if run_agnostic; then
        NB_CSV="$RESULTS/agnostic/neighbors/neighbors_k30.csv"
        VTAG_PRED="$RESULTS/agnostic/vtag/predictions"
        VTAG_EVAL="$RESULTS/agnostic/vtag/evaluations"

        if [[ -f "$VTAG_EVAL/eval_k30.csv" ]]; then
            echo "  SKIP agnostic VTAG: already complete"
        else
            echo "  --- VTAG agnostic ---"
            $PYTHON vtag.py \
                --neighbors_csv "$NB_CSV" \
                --output_dir "$VTAG_PRED" \
                --eval_dir "$VTAG_EVAL" \
                --ks "$VTAG_KS"
        fi
    fi

    # 4b: Project-specific
    if run_specific; then
        for j in "${!PROJECTS[@]}"; do
            NB_CSV="$RESULTS/project_specific/${PROJECT_TAGS[$j]}/neighbors/neighbors_k30.csv"
            VTAG_PRED="$RESULTS/project_specific/${PROJECT_TAGS[$j]}/vtag/predictions"
            VTAG_EVAL="$RESULTS/project_specific/${PROJECT_TAGS[$j]}/vtag/evaluations"

            if [[ -f "$VTAG_EVAL/eval_k30.csv" ]]; then
                echo "  SKIP ${PROJECTS[$j]} VTAG: already complete"
                continue
            fi

            echo "  --- VTAG ${PROJECTS[$j]} ---"
            $PYTHON vtag.py \
                --neighbors_csv "$NB_CSV" \
                --output_dir "$VTAG_PRED" \
                --eval_dir "$VTAG_EVAL" \
                --ks "$VTAG_KS"
        done
    fi
else
    echo -e "\n[Phase 4] SKIPPED (--skip_vtag)"
fi

# ============================================================================
# Phase 5: Evaluation
# ============================================================================

if [[ "$SKIP_EVAL" -eq 0 ]]; then
    echo -e "\n[Phase 5] Evaluation"

    # 5a: Evaluate all RAGTAG + zero-shot predictions
    echo -e "\n  --- Evaluating RAGTAG / zero-shot predictions ---"
    for mi in "${!ALL_MODELS[@]}"; do
        TAG="${ALL_TAGS[$mi]}"
        SHORT="${ALL_SHORT[$mi]}"

        # Agnostic
        if run_agnostic; then
            PRED_DIR="$RESULTS/agnostic/${TAG}/ragtag/predictions"
            EVAL_DIR="$RESULTS/agnostic/${TAG}/ragtag/evaluations"
            for PRED_FILE in "$PRED_DIR"/preds_*.csv; do
                [ -f "$PRED_FILE" ] || continue
                mkdir -p "$EVAL_DIR"
                BASENAME=$(basename "$PRED_FILE" .csv)
                K_TAG="${BASENAME#preds_}"
                EVAL_FILE="${EVAL_DIR}/eval_${K_TAG}.csv"
                if [[ -f "$EVAL_FILE" ]]; then
                    continue
                fi
                echo "    Eval: ${SHORT} agnostic ${K_TAG}"
                $PYTHON evaluate.py --preds_csv "$PRED_FILE" --output_csv "$EVAL_FILE"
            done
        fi

        # Project-specific
        if run_specific; then
            for j in "${!PROJECTS[@]}"; do
                PRED_DIR="$RESULTS/project_specific/${PROJECT_TAGS[$j]}/${TAG}/ragtag/predictions"
                EVAL_DIR="$RESULTS/project_specific/${PROJECT_TAGS[$j]}/${TAG}/ragtag/evaluations"
                for PRED_FILE in "$PRED_DIR"/preds_*.csv; do
                    [ -f "$PRED_FILE" ] || continue
                    mkdir -p "$EVAL_DIR"
                    BASENAME=$(basename "$PRED_FILE" .csv)
                    K_TAG="${BASENAME#preds_}"
                    EVAL_FILE="${EVAL_DIR}/eval_${K_TAG}.csv"
                    if [[ -f "$EVAL_FILE" ]]; then
                        continue
                    fi
                    echo "    Eval: ${SHORT} ${PROJECTS[$j]} ${K_TAG}"
                    $PYTHON evaluate.py --preds_csv "$PRED_FILE" --output_csv "$EVAL_FILE"
                done
            done
        fi
    done

    # 5b: Evaluate FT predictions
    echo -e "\n  --- Evaluating FT predictions ---"
    for mi in "${!ALL_MODELS[@]}"; do
        TAG="${ALL_TAGS[$mi]}"
        SHORT="${ALL_SHORT[$mi]}"

        if run_agnostic; then
            FT_PREDS="$RESULTS/agnostic/${TAG}/finetune_fixed/preds_finetune_fixed.csv"
            FT_EVAL="$RESULTS/agnostic/${TAG}/finetune_fixed/eval_finetune_fixed.csv"
            if [[ -f "$FT_PREDS" && ! -f "$FT_EVAL" ]]; then
                echo "    Eval: ${SHORT} agnostic FT"
                $PYTHON evaluate.py --preds_csv "$FT_PREDS" --output_csv "$FT_EVAL"
            fi
        fi

        if run_specific; then
            for j in "${!PROJECTS[@]}"; do
                FT_PREDS="$RESULTS/project_specific/${PROJECT_TAGS[$j]}/${TAG}/finetune_fixed/preds_finetune_fixed.csv"
                FT_EVAL="$RESULTS/project_specific/${PROJECT_TAGS[$j]}/${TAG}/finetune_fixed/eval_finetune_fixed.csv"
                if [[ -f "$FT_PREDS" && ! -f "$FT_EVAL" ]]; then
                    echo "    Eval: ${SHORT} ${PROJECTS[$j]} FT"
                    $PYTHON evaluate.py --preds_csv "$FT_PREDS" --output_csv "$FT_EVAL"
                fi
            done
        fi
    done

    # 5c: Per-project evaluation from agnostic predictions
    if run_agnostic; then
        echo -e "\n  --- Per-project breakdown from agnostic predictions ---"

        $PYTHON -c "
import pandas as pd
import subprocess
import os
import sys

RESULTS = '$RESULTS'
PROJECTS = [
    ('ansible/ansible', 'ansible_ansible'),
    ('bitcoin/bitcoin', 'bitcoin_bitcoin'),
    ('dart-lang/sdk', 'dart-lang_sdk'),
    ('dotnet/roslyn', 'dotnet_roslyn'),
    ('facebook/react', 'facebook_react'),
    ('flutter/flutter', 'flutter_flutter'),
    ('kubernetes/kubernetes', 'kubernetes_kubernetes'),
    ('microsoft/TypeScript', 'microsoft_TypeScript'),
    ('microsoft/vscode', 'microsoft_vscode'),
    ('opencv/opencv', 'opencv_opencv'),
    ('tensorflow/tensorflow', 'tensorflow_tensorflow'),
]
TAGS = [
    'unsloth_Llama_3_2_3B_Instruct',
    'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit',
    'unsloth_Qwen2_5_14B_Instruct_bnb_4bit',
    'unsloth_Qwen2_5_32B_Instruct_bnb_4bit',
]
SHORTS = ['Llama-3B', 'Llama-8B', 'Qwen-14B', 'Qwen-32B']

# Load test_split.csv to get repo column
test_split = os.path.join(RESULTS, 'agnostic/neighbors/test_split.csv')
if not os.path.exists(test_split):
    print('  WARNING: test_split.csv not found, skipping per-project eval')
    sys.exit(0)

test_df = pd.read_csv(test_split)
test_df['test_idx'] = test_df.index

for tag, short in zip(TAGS, SHORTS):
    # RAGTAG + zero-shot per-project
    pred_dir = os.path.join(RESULTS, f'agnostic/{tag}/ragtag/predictions')
    pp_dir = os.path.join(RESULTS, f'agnostic/{tag}/ragtag/per_project')

    for pred_file in sorted(os.listdir(pred_dir)) if os.path.isdir(pred_dir) else []:
        if not pred_file.startswith('preds_') or not pred_file.endswith('.csv'):
            continue
        k_tag = pred_file.replace('preds_', '').replace('.csv', '')

        preds = pd.read_csv(os.path.join(pred_dir, pred_file))
        if 'test_idx' not in preds.columns:
            preds['test_idx'] = range(len(preds))

        merged = preds.merge(test_df[['test_idx', 'repo']], on='test_idx', how='left')

        for repo_name, proj_tag in PROJECTS:
            eval_file = os.path.join(pp_dir, f'eval_{k_tag}_{proj_tag}.csv')
            if os.path.exists(eval_file):
                continue

            proj_preds = merged[merged['repo'] == repo_name]
            if len(proj_preds) == 0:
                continue

            os.makedirs(pp_dir, exist_ok=True)
            tmp_csv = os.path.join(pp_dir, f'_tmp_{k_tag}_{proj_tag}.csv')
            proj_preds.to_csv(tmp_csv, index=False)

            subprocess.run([
                '$PYTHON', 'evaluate.py',
                '--preds_csv', tmp_csv,
                '--output_csv', eval_file,
            ], check=True)
            os.remove(tmp_csv)
            print(f'    {short} {repo_name} {k_tag}')

    # FT per-project
    ft_pred = os.path.join(RESULTS, f'agnostic/{tag}/finetune_fixed/preds_finetune_fixed.csv')
    ft_pp_dir = os.path.join(RESULTS, f'agnostic/{tag}/finetune_fixed/per_project')
    if os.path.exists(ft_pred):
        preds = pd.read_csv(ft_pred)
        if 'test_idx' not in preds.columns:
            preds['test_idx'] = range(len(preds))
        merged = preds.merge(test_df[['test_idx', 'repo']], on='test_idx', how='left')

        for repo_name, proj_tag in PROJECTS:
            eval_file = os.path.join(ft_pp_dir, f'eval_finetune_fixed_{proj_tag}.csv')
            if os.path.exists(eval_file):
                continue

            proj_preds = merged[merged['repo'] == repo_name]
            if len(proj_preds) == 0:
                continue

            os.makedirs(ft_pp_dir, exist_ok=True)
            tmp_csv = os.path.join(ft_pp_dir, f'_tmp_ft_{proj_tag}.csv')
            proj_preds.to_csv(tmp_csv, index=False)

            subprocess.run([
                '$PYTHON', 'evaluate.py',
                '--preds_csv', tmp_csv,
                '--output_csv', eval_file,
            ], check=True)
            os.remove(tmp_csv)
            print(f'    {short} {repo_name} FT')
"
    fi
else
    echo -e "\n[Phase 5] SKIPPED (--skip_eval)"
fi

# ============================================================================
# Phase 6: Summary Report
# ============================================================================

echo -e "\n============================================================"
echo "  11-Project Benchmark — Summary Report"
echo "============================================================"

$PYTHON -c "
import pandas as pd
import os

RESULTS = '$RESULTS'
LABELS = ['bug', 'feature', 'question']
TAGS = [
    ('Llama-3B',  'unsloth_Llama_3_2_3B_Instruct'),
    ('Llama-8B',  'unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit'),
    ('Qwen-14B',  'unsloth_Qwen2_5_14B_Instruct_bnb_4bit'),
    ('Qwen-32B',  'unsloth_Qwen2_5_32B_Instruct_bnb_4bit'),
]
PROJECTS = [
    ('ansible/ansible', 'ansible_ansible'),
    ('bitcoin/bitcoin', 'bitcoin_bitcoin'),
    ('dart-lang/sdk', 'dart-lang_sdk'),
    ('dotnet/roslyn', 'dotnet_roslyn'),
    ('facebook/react', 'facebook_react'),
    ('flutter/flutter', 'flutter_flutter'),
    ('kubernetes/kubernetes', 'kubernetes_kubernetes'),
    ('microsoft/TypeScript', 'microsoft_TypeScript'),
    ('microsoft/vscode', 'microsoft_vscode'),
    ('opencv/opencv', 'opencv_opencv'),
    ('tensorflow/tensorflow', 'tensorflow_tensorflow'),
]

def read_macro_f1(eval_path):
    if not os.path.exists(eval_path):
        return None
    df = pd.read_csv(eval_path)
    return df['f1_macro'].values[0]

def read_preds_f1(preds_path):
    \"\"\"Compute macro-F1 directly from predictions CSV.\"\"\"
    if not os.path.exists(preds_path):
        return None
    from sklearn.metrics import f1_score
    df = pd.read_csv(preds_path)
    valid = df[df['predicted_label'].isin(LABELS)]
    if len(valid) == 0:
        return None
    return f1_score(valid['ground_truth'], valid['predicted_label'],
                    labels=LABELS, average='macro', zero_division=0)

# ---- Agnostic Overall ----
print()
print('=== AGNOSTIC SETTING (overall macro F1) ===')
header = f'{\"Model\":>10}  {\"Zero-shot\":>9}  {\"VTAG\":>7}  {\"k=1\":>7}  {\"k=3\":>7}  {\"k=6\":>7}  {\"k=9\":>7}  {\"FT\":>7}'
print(header)
print('-' * len(header))

# VTAG best
vtag_eval_dir = os.path.join(RESULTS, 'agnostic/vtag/evaluations')
vtag_best = 0
for k in range(1, 31):
    f1 = read_macro_f1(os.path.join(vtag_eval_dir, f'eval_k{k}.csv'))
    if f1 and f1 > vtag_best:
        vtag_best = f1

for short, tag in TAGS:
    zs = read_macro_f1(os.path.join(RESULTS, f'agnostic/{tag}/ragtag/evaluations/eval_zero_shot.csv'))
    k1 = read_macro_f1(os.path.join(RESULTS, f'agnostic/{tag}/ragtag/evaluations/eval_k1.csv'))
    k3 = read_macro_f1(os.path.join(RESULTS, f'agnostic/{tag}/ragtag/evaluations/eval_k3.csv'))
    k6 = read_macro_f1(os.path.join(RESULTS, f'agnostic/{tag}/ragtag/evaluations/eval_k6.csv'))
    k9 = read_macro_f1(os.path.join(RESULTS, f'agnostic/{tag}/ragtag/evaluations/eval_k9.csv'))
    ft = read_macro_f1(os.path.join(RESULTS, f'agnostic/{tag}/finetune_fixed/eval_finetune_fixed.csv'))

    def fmt(v): return f'{v:.4f}' if v else '   --  '
    print(f'{short:>10}  {fmt(zs):>9}  {vtag_best:7.4f}  {fmt(k1):>7}  {fmt(k3):>7}  {fmt(k6):>7}  {fmt(k9):>7}  {fmt(ft):>7}')

# ---- Per-project (agnostic, best RAGTAG k) ----
print()
print('=== AGNOSTIC PER-PROJECT (best RAGTAG k per model) ===')
for short, tag in TAGS:
    # Find best k for this model
    best_k, best_f1 = None, 0
    for k in [1, 3, 6, 9]:
        f1 = read_macro_f1(os.path.join(RESULTS, f'agnostic/{tag}/ragtag/evaluations/eval_k{k}.csv'))
        if f1 and f1 > best_f1:
            best_f1 = f1
            best_k = k

    if best_k is None:
        continue

    print(f'\n  {short} (best k={best_k}, overall F1={best_f1:.4f})')
    pp_header = f'  {\"Project\":<25} {\"VTAG\":>7}  {\"RAGTAG\":>7}  {\"FT\":>7}'
    print(pp_header)
    print('  ' + '-' * (len(pp_header) - 2))

    for repo_name, proj_tag in PROJECTS:
        # VTAG per-project (from project-specific VTAG, best k)
        vtag_pp_dir = os.path.join(RESULTS, f'project_specific/{proj_tag}/vtag/evaluations')
        vtag_pp = 0
        for vk in range(1, 31):
            f = read_macro_f1(os.path.join(vtag_pp_dir, f'eval_k{vk}.csv'))
            if f and f > vtag_pp:
                vtag_pp = f

        # RAGTAG per-project from agnostic
        ragtag_pp = read_macro_f1(os.path.join(RESULTS, f'agnostic/{tag}/ragtag/per_project/eval_k{best_k}_{proj_tag}.csv'))

        # FT per-project from agnostic
        ft_pp = read_macro_f1(os.path.join(RESULTS, f'agnostic/{tag}/finetune_fixed/per_project/eval_finetune_fixed_{proj_tag}.csv'))

        def fmt(v): return f'{v:.4f}' if v else '   --  '
        print(f'  {repo_name:<25} {fmt(vtag_pp):>7}  {fmt(ragtag_pp):>7}  {fmt(ft_pp):>7}')

# ---- Project-specific vs Agnostic comparison ----
print()
print('=== PROJECT-SPECIFIC vs AGNOSTIC (RAGTAG, per model) ===')
for short, tag in TAGS:
    best_k, best_f1 = None, 0
    for k in [1, 3, 6, 9]:
        f1 = read_macro_f1(os.path.join(RESULTS, f'agnostic/{tag}/ragtag/evaluations/eval_k{k}.csv'))
        if f1 and f1 > best_f1:
            best_f1 = f1
            best_k = k

    if best_k is None:
        continue

    print(f'\n  {short} (k={best_k})')
    cmp_header = f'  {\"Project\":<25} {\"Agnostic\":>9}  {\"Specific\":>9}  {\"Delta\":>7}'
    print(cmp_header)
    print('  ' + '-' * (len(cmp_header) - 2))

    for repo_name, proj_tag in PROJECTS:
        ag = read_macro_f1(os.path.join(RESULTS, f'agnostic/{tag}/ragtag/per_project/eval_k{best_k}_{proj_tag}.csv'))
        sp = read_macro_f1(os.path.join(RESULTS, f'project_specific/{proj_tag}/{tag}/ragtag/evaluations/eval_k{best_k}.csv'))

        def fmt(v): return f'{v:.4f}' if v else '   --  '
        if ag and sp:
            delta = f'{sp - ag:+.4f}'
        else:
            delta = '   --  '
        print(f'  {repo_name:<25} {fmt(ag):>9}  {fmt(sp):>9}  {delta:>7}')

print()
"

echo -e "\nDone!"

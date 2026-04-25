#!/usr/bin/env bash
# Wave 0 indexing: agnostic + 11 project-specific. Run inside the container.
# Working dir is /workspace/llm-labler. Writes to mounted results-pvc.
set -euo pipefail

TRAIN_CSV="issues11k_train.csv"
TEST_CSV="issues11k_test.csv"
TOP_KS="3,9,30"
EMBED="sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR="/workspace/hf_cache"
RESULTS="results/issues11k"

PROJECTS_FILTERS=(
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
PROJECT_TAGS=(
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

echo "[indexing] === agnostic ==="
python build_11k_index.py \
  --train_csv "$TRAIN_CSV" --test_csv "$TEST_CSV" \
  --top_ks "$TOP_KS" \
  --output_dir "$RESULTS/agnostic/neighbors" \
  --embedding_model "$EMBED" \
  --model_cache_dir "$CACHE_DIR"

for i in "${!PROJECTS_FILTERS[@]}"; do
  filter="${PROJECTS_FILTERS[$i]}"
  tag="${PROJECT_TAGS[$i]}"
  echo "[indexing] === project-specific: $filter -> $tag ==="
  python build_11k_index.py \
    --train_csv "$TRAIN_CSV" --test_csv "$TEST_CSV" \
    --repo_filter "$filter" \
    --top_ks "$TOP_KS" \
    --output_dir "$RESULTS/project_specific/${tag}/neighbors" \
    --embedding_model "$EMBED" \
    --model_cache_dir "$CACHE_DIR"
done

echo "[indexing] all done."

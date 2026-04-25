#!/usr/bin/env bash
# Smoke test indexing: ONE project. Args: <repo_filter> <project_tag> <output_root>
# Example: run_indexing_one.sh ansible/ansible ansible_ansible results/issues11k/_smoketest
set -euo pipefail

FILTER="${1:?repo_filter required}"
TAG="${2:?project_tag required}"
OUTROOT="${3:?output_root required}"

EMBED="sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR="/workspace/hf_cache"

echo "[smoketest-indexing] $FILTER -> $TAG -> $OUTROOT"
python build_11k_index.py \
  --train_csv issues11k_train.csv --test_csv issues11k_test.csv \
  --repo_filter "$FILTER" \
  --top_ks "1,3,9" \
  --output_dir "${OUTROOT}/${TAG}/neighbors" \
  --embedding_model "$EMBED" \
  --model_cache_dir "$CACHE_DIR"
echo "[smoketest-indexing] done."

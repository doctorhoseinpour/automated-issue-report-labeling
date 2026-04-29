#!/usr/bin/env bash
# run_qwen_3b_7b_full.sh — End-to-end pipeline for Qwen2.5-3B and Qwen2.5-7B.
# Runs the main 11k experiments (zero-shot, RAGTAG k=1,3,6,9, FT) then debias.
# All output goes straight to the terminal.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Qwen2.5-3B / Qwen2.5-7B full pipeline"
echo "  Started: $(date)"
echo "============================================================"

# Phase 1+2+3 (zero-shot, RAGTAG, FT). Indexing and VTAG already done; skip them.
bash run_11k_experiments.sh \
    --mode local \
    --setting both \
    --skip_indexing \
    --skip_vtag

echo ""
echo "============================================================"
echo "  Main pipeline done — starting debias"
echo "  $(date)"
echo "============================================================"

bash run_11k_debias_qwen.sh

echo ""
echo "============================================================"
echo "  All Qwen2.5-3B/7B runs complete — $(date)"
echo "============================================================"

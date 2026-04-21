#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_local_experiments.sh — Run all local experiments on RTX 4090
# ============================================================================
# Usage:
#   nohup bash run_local_experiments.sh > local_log.txt 2>&1 &
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Local experiments — $(date)"
echo "============================================================"

# --- Experiment 1: RAGTAG (all models) + Llama FT at 1.5k, 3k, 9k, 15k ---
echo -e "\n[1/2] Data efficiency — RAGTAG + Llama fine-tuning"
bash run_data_efficiency.sh --mode local

# --- Experiment 2: Qwen debias on 3k ---
echo -e "\n[2/2] Debiased retrieval — Qwen on 3k"
bash run_debias_qwen.sh --skip_30k

echo -e "\n============================================================"
echo "  All local experiments complete — $(date)"
echo "============================================================"

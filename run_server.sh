#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# run_server.sh — Run all remote experiments on OSC A100
# ============================================================================
# Usage:
#   nohup bash run_server.sh > server_log.txt 2>&1 &
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Server experiments — $(date)"
echo "============================================================"

# --- Experiment 1: Qwen FT at 1.5k, 3k, 9k, 15k ---
echo -e "\n[1/2] Data efficiency — Qwen fine-tuning"
bash run_data_efficiency.sh --mode remote

# --- Experiment 2: Qwen debias on 30k ---
echo -e "\n[2/2] Debiased retrieval — Qwen on 30k"
bash run_debias_qwen.sh --skip_3k

echo -e "\n============================================================"
echo "  All server experiments complete — $(date)"
echo "============================================================"

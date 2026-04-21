#!/usr/bin/env bash
#SBATCH --job-name=ragtag-experiments
#SBATCH --account=PCS0289
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --output=server_log_%j.txt
#SBATCH --error=server_log_%j.txt
set -euo pipefail

# ============================================================================
# run_server.sh — Slurm job for remote experiments on OSC H100
# ============================================================================
# Usage:
#   sbatch run_server.sh
#
# Monitor:
#   tail -f server_log_<jobid>.txt
#   squeue -u $USER
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

module load python/3.12
source venv/bin/activate

echo "============================================================"
echo "  Server experiments — $(date)"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================================"

# --- Experiment 1: Qwen FT at 1.5k, 3k, 9k, 15k ---
echo -e "\n[1/2] Data efficiency — Qwen fine-tuning"
bash run_data_efficiency.sh --mode remote --nrp

# --- Experiment 2: Qwen debias on 30k ---
echo -e "\n[2/2] Debiased retrieval — Qwen on 30k"
bash run_debias_qwen.sh --skip_3k --nrp

echo -e "\n============================================================"
echo "  All server experiments complete — $(date)"
echo "============================================================"

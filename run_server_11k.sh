#!/usr/bin/env bash
#SBATCH --job-name=11k-qwen-ft
#SBATCH --account=PCS0289
#SBATCH --partition=batch
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=server_11k_log_%j.txt
#SBATCH --error=server_11k_log_%j.txt
set -euo pipefail

# ============================================================================
# run_server_11k.sh — Slurm job for Qwen FT on 11-project benchmark (OSC H100)
# ============================================================================
# Runs Qwen-14B and Qwen-32B fine-tuning on:
#   - Agnostic setting (3300 train)
#   - Project-specific setting (11 × 300 train)
#
# Estimated runtime: ~10h (16h wall time with safety margin)
#
# Usage:
#   sbatch run_server_11k.sh
#
# Monitor:
#   tail -f server_11k_log_<jobid>.txt
#   squeue -u $USER
# ============================================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

module load python/3.12
source venv/bin/activate

echo "============================================================"
echo "  11k Qwen FT — $(date)"
echo "  Job ID: ${SLURM_JOB_ID:-local}"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================================"

# Qwen FT only — skip everything else (indexing, zero-shot, RAGTAG, VTAG done locally)
bash run_11k_experiments.sh \
    --mode remote \
    --nrp \
    --skip_indexing \
    --skip_zero_shot \
    --skip_ragtag \
    --skip_vtag

echo -e "\n============================================================"
echo "  11k Qwen FT complete — $(date)"
echo "============================================================"

#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=ragnext-extract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/fs/ess/PCS0289/rag_next/logs/%x-%j.out
#
# GPU job (Cardinal, 1x H100-94GB quarter node): run llm_features.py with the venv built by
# setup_env.sbatch. Arguments are passed straight to llm_features.py, e.g.
#   sbatch extract.sbatch --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit --k 0 \
#       --roles inner,dev,test --max_seq_length 8192 \
#       --out /fs/ess/PCS0289/rag_next/repo/results/issues11k/exploration/rag_next/features/q32_k0
# Optional env XCHECK_14B=1: first extract 14B features for the ansible dev issues (90) so they
# can be compared with the lab 4090's q14_k0 (cross-hardware consistency check).
set -euo pipefail
ROOT=/fs/ess/PCS0289/rag_next
module load python/3.12
export PYTHONNOUSERSITE=1
export HF_HOME=$ROOT/hf_cache
export HF_HUB_OFFLINE=1                      # weights were prefetched by setup_env.sbatch
export XDG_CACHE_HOME=$TMPDIR/xdg TRITON_CACHE_DIR=$TMPDIR/triton
export RESULTS_DIR=$ROOT/repo/results/issues11k
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"
echo "[osc] job=$SLURM_JOB_ID node=$(hostname) args=$*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
cd "$ROOT/repo"                              # unsloth writes unsloth_compiled_cache/ into cwd
PY="$ROOT/venv/bin/python"
F=$RESULTS_DIR/exploration/rag_next/features
if [ "${XCHECK_14B:-0}" = "1" ]; then
  $PY scripts/experiments/rag_next/llm_features.py --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit --k 0 \
      --roles dev --projects ansible_ansible --max_seq_length 8192 --out "$F/xcheck_q14_ansible_dev"
fi
$PY scripts/experiments/rag_next/llm_features.py "$@"
echo "[osc] DONE"

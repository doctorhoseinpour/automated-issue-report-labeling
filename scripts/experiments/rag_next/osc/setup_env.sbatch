#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=ragnext-setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/fs/ess/PCS0289/rag_next/logs/%x-%j.out
#
# CPU-only (pennies): build the Python venv (pinned to the lab venv) and prefetch Qwen2.5
# bnb-4bit weights into the project HF cache, so GPU jobs never spend GPU-hours installing
# or downloading. Runs as a batch job because login nodes kill >20-CPU-min processes.
# Idempotent: skips the venv if it already imports unsloth, skips cached models.
#   sbatch infra: cd /fs/ess/PCS0289/rag_next && sbatch repo/scripts/experiments/rag_next/osc/setup_env.sbatch
set -euo pipefail
ROOT=/fs/ess/PCS0289/rag_next
module load python/3.12
export PYTHONNOUSERSITE=1                    # ~/.local on OSC holds a stray torch (my-la gotcha)
export PIP_CACHE_DIR=$ROOT/pipcache
export HF_HOME=$ROOT/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"
if [ ! -x "$ROOT/venv/bin/python" ]; then
  python3 -m venv "$ROOT/venv"
fi
if ! "$ROOT/venv/bin/python" -c "import unsloth" >/dev/null 2>&1; then
  "$ROOT/venv/bin/pip" install --upgrade pip wheel
  "$ROOT/venv/bin/pip" install -r "$ROOT/repo/scripts/experiments/rag_next/osc/requirements-osc.txt"
fi
"$ROOT/venv/bin/pip" freeze > "$ROOT/logs/pip-freeze-$SLURM_JOB_ID.txt"
"$ROOT/venv/bin/python" -c "import torch, transformers, bitsandbytes; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'bnb', bitsandbytes.__version__)"
for m in ${MODELS:-unsloth/Qwen2.5-32B-Instruct-bnb-4bit unsloth/Qwen2.5-14B-Instruct-bnb-4bit}; do
  echo "[setup] prefetch $m"
  "$ROOT/venv/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$m'))"
done
du -sh "$ROOT/venv" "$HF_HOME"
echo "[setup] DONE"

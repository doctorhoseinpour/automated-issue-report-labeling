#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=nm-fetch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/ess/PCS0289/rag_next/logs/newllms/%x-%j.out
#
# CPU job: download one model's weights into the shared HF cache (uses the existing rag_next
# venv's huggingface_hub, so it can run while venv-nm is being built).
#   sbatch prefetch_nm.sbatch Qwen/Qwen3.5-9B
set -euo pipefail
ROOT=/fs/ess/PCS0289/rag_next
module load python/3.12
export PYTHONNOUSERSITE=1 HF_HOME=$ROOT/hf_cache
M="${1:?HF repo id}"
"$ROOT/venv/bin/python" - "$M" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
t = time.time()
p = snapshot_download(sys.argv[1], ignore_patterns=["consolidated.safetensors", "*.gguf", "*.pth", "original/*"])
print("cached", sys.argv[1], "->", p, f"({time.time() - t:.0f}s)")
PY
du -sh "$ROOT/hf_cache"
echo "[prefetch_nm] DONE $M"

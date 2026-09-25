#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --job-name=nm-setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PCS0289/alirezzzhp1378/nm/logs/%x-%j.out
#
# CPU job: build the newllms venv with a uv-managed CPython 3.12 in $HOME/nm, so the same venv runs
# on Cardinal and Ascend nodes (a venv on /apps/python only exists per cluster). It lives in $HOME
# because the shared /fs/ess/PCS0289 fileset is at its 200k-inode limit (weights stay on /fs/ess).
# Idempotent.  sbatch ~/nm/repo/scripts/experiments/newllms/osc/setup_nm.sbatch
set -euo pipefail
ROOT=/users/PCS0289/alirezzzhp1378/nm
module load python/3.12
export PYTHONNOUSERSITE=1
export UV_CACHE_DIR=${TMPDIR:-/tmp}/uvcache UV_PYTHON_INSTALL_DIR=$ROOT/uv-python UV_LINK_MODE=copy
if [ ! -x "$ROOT/uvboot/bin/uv" ]; then
  python3 -m venv "$ROOT/uvboot"
  "$ROOT/uvboot/bin/pip" install -q uv
fi
UV=$ROOT/uvboot/bin/uv
$UV python install 3.12
if [ ! -x "$ROOT/venv-nm/bin/python" ]; then
  $UV venv --python 3.12 --python-preference only-managed "$ROOT/venv-nm"
fi
VIRTUAL_ENV=$ROOT/venv-nm $UV pip install -r "$ROOT/repo/scripts/experiments/newllms/osc/requirements-nm.txt"
cat "$ROOT/venv-nm/pyvenv.cfg"
VIRTUAL_ENV=$ROOT/venv-nm $UV pip freeze > "$ROOT/logs/nm-freeze-$SLURM_JOB_ID.txt"
"$ROOT/venv-nm/bin/python" - <<'PY'
import torch, transformers, bitsandbytes, accelerate
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "bnb", bitsandbytes.__version__, "accelerate", accelerate.__version__)
for c in ["Qwen3_5ForConditionalGeneration", "Gemma4UnifiedForConditionalGeneration", "Mistral3ForConditionalGeneration"]:
    print(c, getattr(transformers, c))
import fla
print("fla", getattr(fla, "__version__", "?"))
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
print("fla chunk_gated_delta_rule OK")
PY
du -sh "$ROOT/venv-nm" "$ROOT/uv-python"
echo "[setup_nm] DONE"

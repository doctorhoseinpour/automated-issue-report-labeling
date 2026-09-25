#!/usr/bin/env bash
#SBATCH --account=PCS0289
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gpus-per-node=1
#SBATCH --output=/users/PCS0289/alirezzzhp1378/nm/logs/%x-%j.out
#
# Generic newllms GPU job (Cardinal or Ascend; cluster, partition, time and job name are given
# at submission). First argument: the script in scripts/experiments/newllms/, the rest are its
# arguments. The shard index comes from NM_SHARD / NM_NSHARDS (exported by launch_nm.sh).
#   sbatch -M ascend --partition=debug-nextgen --time=00:30:00 --job-name=nm-smoke-qw35_9b \
#       gpu_nm.sbatch smoke.py --tag qw35_9b
set -euo pipefail
ROOT=/users/PCS0289/alirezzzhp1378/nm                       # venv, code, outputs (the /fs/ess fileset is at its inode limit)
export HF_HOME=/fs/ess/PCS0289/rag_next/hf_cache   # weights (read-only)
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=$ROOT/triton-cache XDG_CACHE_HOME=${TMPDIR:-/tmp}/xdg TOKENIZERS_PARALLELISM=false
export RESULTS_DIR=$ROOT/repo/results/issues11k
# Jobs submitted from a Cardinal login node inherit its Intel-compiler module env (CC=.../icc),
# which does not exist on Ascend; Triton (fla kernels) compiles its launcher with $CC.
export CC=/usr/bin/gcc CXX=/usr/bin/g++
unset FC F77 F90 LD_LIBRARY_PATH
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
SCRIPT="${1:?script name}"; shift
echo "[nm] job=$SLURM_JOB_ID cluster=${SLURM_CLUSTER_NAME:-?} node=$(hostname) shard=${NM_SHARD:-0}/${NM_NSHARDS:-1} $SCRIPT $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
cd "$ROOT/repo/scripts/experiments/newllms"
"$ROOT/venv-nm/bin/python" "$SCRIPT" "$@"
echo "[nm] EXIT OK"

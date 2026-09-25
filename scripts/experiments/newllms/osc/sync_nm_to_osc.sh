#!/usr/bin/env bash
# Local PC -> OSC: newllms code, the modules it imports (llm_labeler.py, rag_next/common.py), and
# the read-only inputs relayed from the lab machine into a local staging dir ($1):
#   $1/pool.csv, $1/val495.csv, $1/nb_PS_raw_dev.npz, $1/nb_PS_raw_test.npz
#   bash scripts/experiments/newllms/osc/sync_nm_to_osc.sh /path/to/staging   (staging optional)
set -euo pipefail
DEST=alirezzzhp1378@cardinal.osc.edu
ROOT=/users/PCS0289/alirezzzhp1378/nm   # the /fs/ess fileset is at its inode limit; weights stay in /fs/ess/PCS0289/rag_next/hf_cache
R=$ROOT/repo/results/issues11k/exploration
cd "$(git rev-parse --show-toplevel)"
ssh "$DEST" "mkdir -p $ROOT/logs $ROOT/triton-cache $ROOT/repo/scripts/experiments/newllms $ROOT/repo/scripts/experiments/rag_next \
  $R/rag_next/splits $R/rag_next/features $R/newllms/splits $R/newllms/raw"
rsync -a llm_labeler.py "$DEST:$ROOT/repo/"
rsync -a scripts/experiments/rag_next/common.py "$DEST:$ROOT/repo/scripts/experiments/rag_next/"
rsync -a --exclude __pycache__ scripts/experiments/newllms/ "$DEST:$ROOT/repo/scripts/experiments/newllms/"
if [ -n "${1:-}" ]; then
  S="$1"
  rsync -a --checksum "$S/pool.csv" "$DEST:$R/rag_next/splits/pool.csv"
  rsync -a --checksum "$S/nb_PS_raw_dev.npz" "$S/nb_PS_raw_test.npz" "$DEST:$R/rag_next/features/"
  rsync -a --checksum "$S/val495.csv" "$DEST:$R/newllms/splits/val495.csv"
fi
echo "[sync_nm_to_osc] DONE"

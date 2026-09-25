#!/usr/bin/env bash
# Local PC -> OSC: code needed by llm_features.py (+ llm_labeler.py) and the split file.
# pool.csv must first be pulled from the lab machine into the local scratch copy given as $1.
#   bash scripts/experiments/rag_next/osc/sync_to_osc.sh /path/to/local/pool.csv
set -euo pipefail
POOL="${1:?path to a local copy of splits/pool.csv}"
DEST=alirezzzhp1378@cardinal.osc.edu
ROOT=/fs/ess/PCS0289/rag_next
cd "$(git rev-parse --show-toplevel)"
ssh "$DEST" "mkdir -p $ROOT/logs $ROOT/hf_cache $ROOT/pipcache $ROOT/repo/scripts/experiments/rag_next \
  $ROOT/repo/results/issues11k/exploration/rag_next/splits $ROOT/repo/results/issues11k/exploration/rag_next/features"
rsync -a llm_labeler.py "$DEST:$ROOT/repo/"
rsync -a scripts/experiments/rag_next/ "$DEST:$ROOT/repo/scripts/experiments/rag_next/"
rsync -a "$POOL" "$DEST:$ROOT/repo/results/issues11k/exploration/rag_next/splits/pool.csv"
echo "[sync_to_osc] DONE"

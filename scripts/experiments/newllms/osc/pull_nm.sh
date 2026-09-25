#!/usr/bin/env bash
# OSC -> local staging -> lab machine relay of newllms raw outputs (bgsulab and OSC cannot reach
# each other). Incremental (rsync); partial *.tmp parts are skipped.
#   bash scripts/experiments/newllms/osc/pull_nm.sh [staging_dir]
set -euo pipefail
SRC=alirezzzhp1378@cardinal.osc.edu:/users/PCS0289/alirezzzhp1378/nm/repo/results/issues11k/exploration/newllms/raw
ST="${1:-${NM_STAGE:?staging dir}}"
DST=results/issues11k/exploration/newllms/raw
mkdir -p "$ST"
rsync -a --exclude '*.tmp' "$SRC/" "$ST/"
ssh bgsulab "mkdir -p ~/llm-labler/$DST"
rsync -a --exclude '*.tmp' "$ST/" "bgsulab:~/llm-labler/$DST/"
echo "[pull_nm] $(find "$ST" -name 'done_*.json' | wc -l) done markers, $(du -sh "$ST" | cut -f1) relayed"

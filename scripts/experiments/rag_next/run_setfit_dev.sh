#!/bin/bash
# Dev-phase SetFit reference: SetFit-PS with the given body, trained on each project's
# inner slice and scored on its dev slice. Unmodified run_setfit.py; idempotent
# (run_setfit.py skips when preds exist). Run from the repo root on the lab machine.
#   bash scripts/experiments/rag_next/run_setfit_dev.sh Collab-uniba/github-issues-mpnet-st-e10
set -u
cd ~/llm-labler
body=${1:-Collab-uniba/github-issues-mpnet-st-e10}
EXP=results/issues11k/exploration/rag_next
tag=${body//\//_}
for d in $EXP/splits/ps/*/; do
  p=$(basename "$d")
  venv-setfit/bin/python run_setfit.py --body_model "$body" \
    --train_csv "$d/inner.csv" --test_csv "$d/dev.csv" \
    --output_dir "$EXP/setfit_dev/$tag/$p" > "$EXP/logs/setfit_dev_${tag}_$p.log" 2>&1
  echo "$(date +%T) setfit_dev $tag $p exit=$?" >> $EXP/logs/progress.txt
done

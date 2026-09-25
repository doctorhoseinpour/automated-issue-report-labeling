#!/usr/bin/env bash
# The 18 pre-registered TEST evaluations of docs/NEWLLMS_STUDY.md section 2 (lab machine, run once).
# Per model: R, K, ZS, RAG@K* (primary) and ZSc, RAGc (descriptive), each with paired bootstrap CIs
# against the same model's RAG@K* and ZS plus the fixed baselines in eval_test.py.
#   cd ~/llm-labler/scripts/experiments/newllms && bash run_test_evals.sh
set -euo pipefail
PY=../../../venv/bin/python
P=../../../results/issues11k/exploration/newllms/test_preds
for t in qw35_9b gm4_12b mi3_8b; do
  for f in "$P/rag_$t.csv" "$P/zs_$t.csv" "$P/readout_$t.csv" "$P/stateknn_$t.csv"; do
    [ -f "$f" ] || { echo "missing $f"; exit 1; }
  done
done
for t in qw35_9b gm4_12b mi3_8b; do
  VS=(--vs "RAG@K* $t=$P/rag_$t.csv" --vs "ZS $t=$P/zs_$t.csv")
  for key in readout stateknn rag zs ragc zsc; do
    $PY eval_test.py "$P/${key}_$t.csv" "${key}_$t" "${VS[@]}"
  done
done

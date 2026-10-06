# Exploration results (post-deadline studies)

Outputs of the two studies that followed the SANER 2027 submission, copied into git so they
travel with the repo. The write-ups with every number, protocol and caveat are the notebooks:

- `rag_next/`: [docs/RAG_NEXT_STUDY.md](../docs/RAG_NEXT_STUDY.md). The decision-state read-out on Qwen2.5 beats SetFit-PS.
- `newllms/`: [docs/NEWLLMS_STUDY.md](../docs/NEWLLMS_STUDY.md). The same frozen read-out on Qwen3.5-9B, Gemma-4-12B and Ministral-3-8B vs val-tuned RAG.

**Source:** `bgsulab:~/llm-labler/results/issues11k/exploration/{rag_next,newllms}/`, copied
2026-10-06. All 228 files were checked against the lab copies with SHA-256 before compression.
`SHA256SUMS` lists the **uncompressed** originals.

## What is here

| Path | Content |
|---|---|
| `*/test_eval_log.csv` | Every pre-registered test evaluation, in run order (12 for rag_next, 18 for newllms). Macro F1 is pooled over the 3,300 test issues |
| `*/test_eval/` | Per-run `_eval` (per-class P/R/F1), `_per_project`, `_cis` (paired bootstrap CIs) and the post-hoc robustness tables |
| `*/test_preds/*.csv.gz` | Test predictions in the `evaluate.py` schema (`test_idx, title, body, ground_truth, predicted_label, ...`), plus `*.config.json` describing each read-out/kNN head |
| `rag_next/splits/*.csv.gz` | The dev split carved from train (`pool`, `inner`, `dev`, and per-project `ps/<project>/`) |
| `rag_next/headroom/` | Headroom analysis over the existing paper predictions (`headroom.py`) |
| `rag_next/dev_logs/`, `*/logs/` | Dev-study and job logs |
| `rag_next/queue.txt` | The GPU-queue job list (exact feature-extraction commands) |
| `newllms/splits/val495.csv`, `newllms/val/` | Validation set and K-selection curves / read-out dev checks |
| `newllms/gen/` | Generation outputs for zero-shot, RAG@K* on test, and the validation K sweeps |

Test macro F1 (from the logs):

| | Read-out | State kNN | Other |
|---|---|---|---|
| Qwen2.5 3B / 7B / 14B / 32B | 0.8127 / 0.8223 / 0.8278 / **0.8445** | 0.7730 / 0.7793 / 0.7916 / 0.7993 | Fusion 0.8186; 14B with 10 labels per class per project 0.7866 / 0.7970 / 0.7921 |
| Qwen3.5-9B | 0.8292 | 0.8002 | RAG@9 0.7516, zero-shot 0.6841 |
| Gemma-4-12B | 0.8270 | 0.7856 | RAG@12 0.7741, zero-shot 0.6963 |
| Ministral-3-8B | 0.8266 | 0.7836 | RAG@15 0.7388, zero-shot 0.6376 |

`ragc_*`/`zsc_*` are the constrained-decoding variants of `rag_*`/`zs_*`. The reference bar is SetFit-PS at 0.8053.

## Not copied (too large; lab machine only)

| Path on bgsulab | Size | What |
|---|---|---|
| `rag_next/features/` | 2.5 GB | Answer-position hidden states per model (`.npz`); needed to refit heads |
| `rag_next/setfit_dev/` | 14 GB | SetFit models trained on the dev split |
| `newllms/features/` | 6.2 GB | Hidden states for the three new models |
| `newllms/raw/` | 5.6 GB | Raw OSC shard outputs (also on OSC under `~/nm/repo/results/...`) |

## Using these files with the scripts

The scripts read from `results/issues11k/exploration/...`. To restore that layout on a machine
without the lab copy:

```bash
mkdir -p results/issues11k/exploration
cp -r exploration_results/rag_next exploration_results/newllms results/issues11k/exploration/
find results/issues11k/exploration -name '*.csv.gz' -exec gunzip {} +
# verify against the originals
(cd results/issues11k/exploration && sha256sum -c ../../../exploration_results/SHA256SUMS --quiet)
```

pandas also reads the `.csv.gz` files directly (`pd.read_csv("....csv.gz")`).

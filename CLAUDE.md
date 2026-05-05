# CLAUDE.md — RAGTAG+ vs Fine-Tuning Project

This file gives Claude persistent context about this research project. Read it at the start of every session.

---

## Project Overview

**Title:** RAGTAG+ vs Fine-Tuning: Automated GitHub Issue Classification

**Goal:** An academic research paper comparing four approaches to classifying GitHub issues into `bug`, `feature`, or `question`.

**Sole dataset:** 11-project benchmark — `issues11k.csv` (6,600 issues, 11 projects × 600, balanced bug/feature/question). The 3k and 30k datasets are deprecated and their analyses have been removed from the repo.

### Four Approaches

1. **RAGTAG (RAG-enhanced few-shot prompting)** — primary method
   - FAISS retrieval over the labeled training set, top-k neighbors injected as few-shot examples in chat format with `<label>X</label>` XML tags
   - Assistant prefill (`<label>`) so the model only generates the label + closing tag
   - Inference-only — no training
   - Two settings: **project-agnostic** (one FAISS index over all 3,300 train issues) and **project-specific** (separate index per project, 300 train each)
   - k values for the 11k benchmark: 0 (zero-shot), 1, 3, 6, 9 (justified by VTAG's plateau analysis on this dataset)

2. **VTAG (Voting-based RAG baseline, no LLM)** — retrieval floor
   - Script: [vtag.py](vtag.py)
   - For each test issue, retrieve top-k neighbors and vote on their labels. Zero GPU at inference.
   - Voting schemes: `similarity` (Dudani-weighted k-NN, paper default), `shepard` (sim²), `majority` (uniform).
   - Establishes a pure-retrieval floor any RAGTAG number must clear to justify the LLM.

3. **Fine-Tune (LoRA, Unsloth)**
   - LoRA fine-tuning via Unsloth on the same train split as RAGTAG
   - Consistent `PROMPT_TEMPLATE` between train and inference, `tokenizer.eos_token`, `num_train_epochs=1`, strict token truncation, `paged_adamw_8bit`, all predictions included in metrics
   - Represents the fine-tuning baseline the paper compares against

---

## Research Questions

1. **RQ1 — Comparison.** How do RAGTAG, VTAG, and zero-shot perform in project-specific vs project-agnostic settings, per-project and overall? Built-in finding: 88.3% of agnostic-retrieved neighbors come from the same project, so agnostic ≈ project-specific for RAGTAG — that equivalence is itself a result.
2. **RQ2 — Diagnosis.** Why do RAGTAG and zero-shot fall short of fine-tune-agnostic? The leading hypothesis is bug-bias in retrieval/prompting (feature precision is high but recall is low; features get mislabeled as bugs).
3. **RQ3 — Bridging.** Can we close the gap between RAGTAG and fine-tuning without training? Margin-based retrieval debiasing (margin=3) is the validated intervention across the Qwen2.5 family (3B, 7B, 14B, 32B): hurts at low k, helps at k≥6 on every model, with the biggest wins at k=9. At Qwen-14B+ scale, debiased RAGTAG matches or beats fine-tuning. A second method is under consideration (vote-prior injection, prompt-level disambiguation, or batch calibration).

---

## Datasets

| File | Size | Purpose |
|------|------|---------|
| `issues11k.csv` | 6,600 issues | Full pool (11 projects × 600 each) |
| `issues11k_train.csv` | ~3,300 issues | Train split (stratified, project-balanced) |
| `issues11k_test.csv` | ~3,300 issues | Test split (stratified, project-balanced) |

The 11 projects: `ansible/ansible`, `bitcoin/bitcoin`, `dart-lang/sdk`, `dotnet/roslyn`, `facebook/react`, `flutter/flutter`, `kubernetes/kubernetes`, `microsoft/TypeScript`, `microsoft/vscode`, `opencv/opencv`, `tensorflow/tensorflow`.

The orchestrator generates project-specific train/test split CSVs and FAISS indices on first run. Splits are deterministic from the source CSVs.

---

## Models Tested

All models loaded via **Unsloth** (optimized inference + training):

All models are Qwen2.5-Instruct, bnb-4bit (uniform quantization across the family for clean scale comparisons):

| Model | Size | Notes |
|-------|------|-------|
| `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` | 3B 4-bit | Runs on local 4090 |
| `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | 7B 4-bit | Runs on local 4090 |
| `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` | 14B 4-bit | Inference fits 4090; FT runs on NRP A6000 |
| `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` | 32B 4-bit | Inference fits 4090 (~23GB); FT runs on NRP A6000 |

Llama-3B and Llama-8B were used in earlier experiments. Their predictions/evals remain on disk under `results/issues11k/.../unsloth_Llama_*` for reference and the historical record, but are no longer part of the active model lineup.

---

## Infrastructure

- **Local machine:** RTX 4090 (24GB VRAM). Runs all RAGTAG / zero-shot / VTAG inference and Qwen-3B / Qwen-7B fine-tuning.
- **NRP (Nautilus Research Platform):** Shared Kubernetes cluster, namespace `bgsu-cs-heydarnoori`. Migration to Kubernetes Jobs is implemented — pipeline lives in `scripts/nrp/`. Strategy: a single mega-runner Job ([scripts/nrp/runners/run_remaining_cells.py](scripts/nrp/runners/run_remaining_cells.py)) holds one A6000 and processes all 58 production cells sequentially via subprocess, with idempotent skip on existing `preds_*.csv`. Image is SHA-pinned (currently `:55ba8f1`). Two CephFS RWX PVCs back the run: `hf-cache-pvc` (100 Gi, model weights cached once) and `results-pvc` (50 Gi, outputs + `_outbox/` for `sync.sh` pickup). Live status is in [docs/NRP_MIGRATION_STATUS.md](docs/NRP_MIGRATION_STATUS.md).
- **OSC Ascend:** Available as a backup for fine-tuning via `run_server_11k.sh` (Slurm, A100 partition).

---

## Pipeline Architecture

### `run_11k_experiments.sh` — primary orchestrator
End-to-end pipeline for the 11-project benchmark across both settings (agnostic + project-specific). Phases: train/test split CSVs → FAISS indexes → zero-shot → RAGTAG (k=1,3,6,9) → fine-tuning → VTAG → evaluation → summary report.

Key flags:
- `--mode local|remote` — local trains Qwen-3B/7B; remote trains Qwen-14B/32B
- `--setting agnostic|specific|both`
- `--skip_indexing`, `--skip_zero_shot`, `--skip_ragtag`, `--skip_ft`, `--skip_vtag`, `--skip_eval` — phase-level resume

Built-in resume: re-running with the same flags skips phases whose output files already exist.

### `run_11k_debias_qwen.sh` — debiased RAGTAG for Qwen on 11k
Runs debiased RAGTAG (margin=3, k=1,3,6,9, ctx=8192) for Qwen-14B and Qwen-32B across all 11 projects. Expects neighbor files from a prior `run_11k_experiments.sh` run. Outputs to `results/issues11k_debias_m3/...`.

### `run_server_11k.sh` — Slurm wrapper for OSC
SBATCH script that calls `run_11k_experiments.sh --mode remote --skip_indexing --skip_zero_shot --skip_ragtag --skip_vtag` on an A100 partition. 16h wall time.

### `build_11k_index.py` — FAISS indexing for the 11k benchmark
Builds project-agnostic and project-specific FAISS indices from `issues11k_train.csv`, queries with `issues11k_test.csv`, writes `neighbors_k{N}.csv` per setting. Imports `clean_text` and `build_faiss_index` from [build_and_query_index.py](build_and_query_index.py) (the latter is kept solely as a helper module — its CLI is no longer used).

### `llm_labeler.py` — RAGTAG inference
Loads the model once via Unsloth, runs all k values sequentially. Chat format with few-shot examples as user/assistant turns, assistant prefill (`<label>`), XML parsing + regex fallback, smart proportional truncation when prompt exceeds `max_seq_length`. Supports `--inference_batch_size` (left-padded), `--debias_retrieval` + `--debias_margin` for the RQ3 intervention, and `--cache_dir` for HF cache redirection.

### `fixed_fine-tune.py` — fine-tune baseline
LoRA via Unsloth. Accepts external `--train_csv` / `--test_csv` for shared splits with RAGTAG. Tracks absolute peak GPU memory across training and inference for fair comparison against RAGTAG.

### `evaluate.py` — metrics
Per-label and macro precision/recall/F1, accuracy, invalid prediction rate. Works on prediction CSVs from any approach.

### `vtag.py` — voting baseline
Reads a neighbors CSV with similarity scores, votes on labels, writes predictions in the same schema as RAGTAG so `evaluate.py` works as-is. Auto-invokes `evaluate.py` per k if `--eval_dir` is passed.

---

## Output Structure

```
results/issues11k/
  agnostic/
    neighbors/
      train_split.csv
      test_split.csv
      neighbors_k{3,9,30}.csv
    <model_tag>/
      ragtag/
        predictions/preds_{zero_shot,k1,k3,k6,k9}.csv
        evaluations/eval_{zero_shot,k1,k3,k6,k9}.csv
      finetune_fixed/
        preds_finetune_fixed.csv
        eval_finetune_fixed.csv
    vtag/
      predictions/, evaluations/
  project_specific/
    <project_tag>/         # 11 directories, e.g. ansible_ansible, facebook_react
      neighbors/
      <model_tag>/
        ragtag/, finetune_fixed/
      vtag/

results/issues11k_debias_m3/
  <model_tag>/
    ragtag/
      predictions/preds_k{1,3,6,9}.csv
      evaluations/eval_k{1,3,6,9}.csv
```

---

## GPU Memory Tracking

Fine-tuning scripts:
1. Load model
2. Reset peak memory tracker
3. Run training → record `gpu_peak_memory_training_mb`
4. Run inference (no reset) → record `gpu_peak_memory_mb` (absolute max across training + inference)

This ensures the reported peak reflects true maximum VRAM usage for fair comparison against RAGTAG (which has no training phase).

---

## Current Status — Remaining Experiments

The 11k benchmark is partially complete. Remaining work tracked for the NRP migration plan:

1. **Qwen-32B RAGTAG re-run** (agnostic + project-specific) — prior run had OOM / invalid outputs; needs clean re-execution
2. **Qwen-14B fine-tune on 11k** (agnostic + project-specific)
3. **Qwen-32B fine-tune on 11k** (agnostic + project-specific) — A100 not needed; runs on A6000 (peak ~32 GB at `max_seq_length=2048` per 30k cost_metrics, well under the 48 GB cap). Wave 5 of the mega-runner.
4. **Qwen-14B debias on 11k** (margin=3, k=1,3,6,9, project-specific) — script ready: `run_11k_debias_qwen.sh`
5. **Qwen-32B debias on 11k** (same config) — same script

A second RQ3 intervention method is under discussion; vote-prior injection is the leading candidate but not yet implemented.

---

## Conventions

- **`results/` is paper-archival.** Never delete, move, or rename anything inside it during refactors. Even content from deprecated 3k/30k experiments stays as evidence trail.
- **Paper writing lives in `paper/`** (LIPIcs template, `sections/*.tex`); the canonical session handoff is [paper/SESSION_HANDOFF.md](paper/SESSION_HANDOFF.md). Older findings docs in `docs/` (`11K_BENCHMARK_FINDINGS.md`, `PAPER_NARRATIVE.md`) are reference only and partially superseded — their FT numbers are 1-epoch and their links to `docs/analysis/` are broken because that directory was deleted along with `docs/legacy/` to remove stale-analysis bloat.
- **Splits are deterministic** — regenerated from the source CSV on first run; safe to delete and re-create.

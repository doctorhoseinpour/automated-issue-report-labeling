# CLAUDE.md — RAGTAG vs Fine-Tuning Project

This file gives Claude persistent context about this research project. Read it at the start of every session, then read [paper/SESSION_HANDOFF.md](paper/SESSION_HANDOFF.md) for the live state.

---

## Project Overview

**Title:** RAGTAG vs Fine-Tuning: Automated GitHub Issue Classification (ESEM 2026 submission, LIPIcs template, anonymous mode).

**Goal:** An academic research paper comparing four approaches to classifying GitHub issues into `bug`, `feature`, or `question`.

**Sole dataset:** 11-project benchmark — `issues11k.csv` (6,600 issues, 11 projects × 600, balanced bug/feature/question). The 3k and 30k datasets are deprecated and removed from the active repo.

### Four Approaches

1. **VOTAG (Voting-based RAG baseline, no LLM)** — retrieval floor
   - Script: [vtag.py](vtag.py)
   - For each test issue, retrieve top-k neighbors and vote on their labels. Zero GPU at inference.
   - Voting schemes: `similarity` (Dudani-weighted k-NN, paper default), `shepard` (sim²), `majority` (uniform).
   - Establishes a pure-retrieval floor any RAGTAG number must clear to justify the LLM.
   - K-curve scanned over {1..30} establishes the plateau used to justify the {0,1,3,6,9} grid.

2. **RAGTAG (RAG-enhanced few-shot prompting)** — primary method
   - FAISS retrieval over the labeled training set, top-k neighbors injected as few-shot examples in chat format with `<label>X</label>` XML tags.
   - Assistant prefill (`<label>`) so the model only generates the label + closing tag.
   - Inference-only — no training.
   - Two settings: **project-agnostic** (one FAISS index over all 3,300 train issues) and **project-specific** (separate index per project, 300 train each).
   - Active k values: {0 (zero-shot), 1, 3, 6, 9, 12, 15} at ctx=8192. The {12, 15} extension is complete for all four Qwen sizes (3B/7B/14B local, 32B NRP).

3. **Debiased RAGTAG** — RQ3 intervention
   - Margin-based retrieval debiasing (`--debias_retrieval --debias_margin 3`) on top of RAGTAG.
   - PS-only by design.
   - Outputs in canonical paths under `ragtag_debias_m3/` (no `_v2` suffix anywhere — all paths post-canonicalization).

4. **Fine-Tune (LoRA, Unsloth)**
   - LoRA fine-tuning via Unsloth on the same train split as RAGTAG.
   - Current campaign: `num_train_epochs=3` (was 1; bumped to 3 after 1-epoch results were under-trained on 3.3k issues).
   - `paged_adamw_8bit`, consistent `PROMPT_TEMPLATE` between train and inference.
   - Represents the fine-tuning baseline the paper compares against.

---

## Datasets

| File | Size | Purpose |
|------|------|---------|
| `issues11k.csv` | 6,600 issues | Full pool (11 projects × 600 each) |
| `issues11k_train.csv` | ~3,300 issues | Train split (stratified, project-balanced) |
| `issues11k_test.csv` | ~3,300 issues | Test split (stratified, project-balanced) |

The 11 projects: `ansible/ansible`, `bitcoin/bitcoin`, `dart-lang/sdk`, `dotnet/roslyn`, `facebook/react`, `flutter/flutter`, `kubernetes/kubernetes`, `microsoft/TypeScript`, `microsoft/vscode`, `opencv/opencv`, `tensorflow/tensorflow`.

Splits and FAISS indices are deterministic — regenerated from the source CSV on first run; safe to delete and re-create.

---

## Models Tested

All Qwen2.5-Instruct, bnb-4bit (uniform quantization across the family for clean scale comparisons), loaded via **Unsloth**:

| Model | Size | Notes |
|-------|------|-------|
| `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` | 3B 4-bit | Local 4090 |
| `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | 7B 4-bit | Local 4090 |
| `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` | 14B 4-bit | Inference local; FT on NRP A6000/L40 |
| `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` | 32B 4-bit | Inference local (~23GB); FT on NRP A6000/L40 |

Llama-3B and Llama-8B from earlier experiments live on disk under `unsloth_Llama_*` for the historical record but are not in the active lineup.

An exploratory DeBERTa-v3-large encoder fine-tune (PA only) was run as a candidate encoder baseline but mode-collapsed (predicts only `bug`, macro $F_1$ = 0.167) and was dropped from the paper. Outputs preserved at `results/issues11k/agnostic/microsoft_deberta-v3-large/` for the audit trail.

---

## Infrastructure

- **Local machine:** RTX 4090 (24GB VRAM). Runs all RAGTAG / zero-shot / VOTAG inference and Qwen-3B / Qwen-7B fine-tuning.
- **NRP (Nautilus Research Platform):** Shared Kubernetes cluster, namespace `bgsu-cs-heydarnoori`. Pipeline lives in `scripts/nrp/`. Strategy: a single mega-runner Job ([scripts/nrp/runners/run_remaining_cells.py](scripts/nrp/runners/run_remaining_cells.py)) holds one GPU and processes all cells sequentially via subprocess, with idempotent skip on existing `preds_*.csv`. Image is SHA-pinned in [scripts/nrp/plan.yaml](scripts/nrp/plan.yaml). Two CephFS RWX PVCs back the run: `hf-cache-pvc` (model weights) and `results-pvc` (outputs + `_outbox/` for `sync.sh` pickup). Live status in [paper/SESSION_HANDOFF.md](paper/SESSION_HANDOFF.md).
- **OSC Ascend:** Available as a fine-tuning backup via `run_server_11k.sh` (Slurm, A100 partition).

---

## Pipeline Architecture

### `run_11k_experiments.sh` — primary orchestrator
End-to-end pipeline for the 11-project benchmark across both settings (agnostic + project-specific). Phases: train/test split CSVs → FAISS indexes → zero-shot → RAGTAG → fine-tuning → VOTAG → evaluation → summary report. Phase-level resume via `--skip_*` flags.

### `run_11k_debias_qwen.sh` — debiased RAGTAG for Qwen on 11k
Runs debiased RAGTAG (margin=3) for Qwen across all 11 projects. Expects neighbor files from a prior `run_11k_experiments.sh` run.

### `run_k12_k15_local_8k.sh` — k-grid extension
Local 8K extension to k∈{12,15} for Qwen-3B/7B/14B. Qwen-32B counterpart was run on NRP via [scripts/nrp/plan.yaml](scripts/nrp/plan.yaml) waves 6 and 7. Both campaigns are complete; idempotent skip on `preds_k15.csv` if re-run.

### `build_11k_index.py` — FAISS indexing for the 11k benchmark
Builds project-agnostic and project-specific FAISS indices from `issues11k_train.csv`, queries with `issues11k_test.csv`, writes `neighbors_k{N}.csv` per setting.

### `llm_labeler.py` — RAGTAG inference
Loads the model once via Unsloth, runs all k values sequentially (`--top_ks "12,15"` is supported). Chat format with few-shot examples as user/assistant turns, assistant prefill (`<label>`), XML parsing + regex fallback, smart proportional truncation when prompt exceeds `max_seq_length`. Supports `--inference_batch_size`, `--debias_retrieval` + `--debias_margin`, `--cache_dir`.

### `fixed_fine-tune.py` — LoRA fine-tune
LoRA via Unsloth. Accepts external `--train_csv` / `--test_csv`. Tracks absolute peak GPU memory across training and inference for fair comparison against RAGTAG.

### `run_transformer_ft.py` — encoder fine-tune (exploratory)
DeBERTa-v3-large fine-tune via HF Trainer; emits cost_metrics matching the LLM-FT schema for clean comparison.

### `evaluate.py` — metrics
Per-label and macro precision/recall/F1, accuracy, invalid prediction rate.

### `vtag.py` — VOTAG voting baseline
Reads a neighbors CSV with similarity scores, votes on labels, writes predictions in the same schema as RAGTAG so `evaluate.py` works as-is.

---

## Canonical Output Structure

All paths are now canonical (no `_v2` anywhere). The 32B retrieval directories were renamed during the May 6 surgery after the 32B OOM rerun completed cleanly.

```
results/issues11k/
  agnostic/
    neighbors/
      train_split.csv, test_split.csv, neighbors_k{3,9,30}.csv
    <model_tag>/
      ragtag/
        predictions/preds_{zero_shot,k1,k3,k6,k9,k12,k15}.csv
        evaluations/eval_*.csv
      finetune_fixed/
        preds_finetune_fixed.csv, eval_finetune_fixed.csv, cost_metrics.csv
    vtag/
      predictions/, evaluations/
  project_specific/
    <project_tag>/         # 11 directories
      neighbors/
      <model_tag>/
        ragtag/
        ragtag_debias_m3/   # Debiased RAGTAG, margin=3 (PS-only by design)
        finetune_fixed/
      vtag/
```

`results/` is paper-archival. Never delete, move, or rename anything inside it during refactors. OOM-affected and superseded data is preserved under `archive/oom_runs_20260422/`.

---

## GPU Memory Tracking

Fine-tuning scripts: load model → reset peak tracker → train (record `gpu_peak_memory_training_mb`) → infer without reset (record `gpu_peak_memory_mb` as absolute max). This is the apples-to-apples peak vs RAGTAG, which has no training phase.

---

## Current Status

Live status, in-flight campaigns, and outstanding TODOs are tracked in [paper/SESSION_HANDOFF.md](paper/SESSION_HANDOFF.md) and [paper/TODO.md](paper/TODO.md). Read those at the start of every session.

---

## Conventions

- **`results/` is paper-archival.** See above. OOM/superseded runs go to `archive/`, not deletion.
- **Canonical paths only.** No `_v2` or other suffixes in active paths. Any new directory follows the structure above.
- **Splits are deterministic** — regenerated from source CSVs on first run.
- **Image SHA discipline.** Each `scripts/nrp/plan.yaml` change requires a commit + image rebuild + push + manifest update before submission. Stale image is the failure mode that has bitten this project most.
- **Pooled aggregation for all reported metrics.** PA and PS macro $F_1$ are both computed by concat-then-evaluate over the 3,300-issue test set (never per-project mean). See [`paper/sections/04_setup.tex`](paper/sections/04_setup.tex) §"Evaluation Metrics" for the rationale. New paper figures/tables go in `scripts/paper/` with pooling baked in; do not retrofit `scripts/analysis/*.py` (legacy, uses per-project mean for PS).

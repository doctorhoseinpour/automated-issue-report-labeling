# CLAUDE.md — RAGTAG+ vs Fine-Tuning Project

This file gives Claude persistent context about this research project. Read it at the start of every session.

---

## Project Overview

**Title:** RAGTAG+ vs Fine-Tuning: Automated GitHub Issue Classification

**Goal:** An academic research paper comparing four approaches to classifying GitHub issues into `bug`, `feature`, or `question`.

### Four Approaches

1. **RAGTAG (RAG-enhanced few-shot prompting)** — primary method
   - Uses FAISS to retrieve similar labeled issues from the training set
   - Includes them as few-shot examples in the LLM prompt (chat format, `<label>X</label>` XML tags)
   - Uses assistant prefill (`<label>`) so the model only generates the label + closing tag
   - No training required — inference only
   - Variables studied: k=0,1,3,5,9,15 retrieved examples; context window=2048/4096/8192/16384

2. **VTAG (Voting-based RAG baseline, no LLM)** — retrieval floor
   - Script: [vtag.py](vtag.py). Full results + analysis: [docs/VTAG_FINDINGS.md](docs/VTAG_FINDINGS.md). Raw prediction/eval CSVs live under `results/vtag/` and `results/vtag_embed/` (gitignored).
   - For each test issue, retrieve top-k neighbors and vote on their labels. Zero GPU at inference, ~3 ms per k for 1,497 issues.
   - Voting schemes: `similarity` (Dudani-weighted k-NN, paper default), `shepard` (sim²), `majority` (uniform).
   - **Canonical config for the paper: `sentence-transformers/all-MiniLM-L6-v2` + `similarity` voting. Best macro-F1 = 0.6451 @ k=16.** This is the "retrieval floor" any RAGTAG number must beat to justify the LLM.
   - Ablations already run but **excluded from main paper** (noted as future work): voting scheme (shepard 0.6465, majority 0.6444 — spread 0.002) and embedder swap (bge-base 0.6692, bge-large 0.6675 — both beat MiniLM by ~0.022). Raw data retained in [results/vtag/](results/vtag/) and [results/vtag_embed/](results/vtag_embed/).

3. **Flawed Fine-Tune Baseline**
   - Faithful reproduction of a state-of-the-art paper's pipeline with **all original flaws preserved**
   - Flaws: train/inference prompt mismatch, chain-of-thought prefix in training but not inference, hardcoded `EOS_TOKEN = "<|endoftext|>"`, `max_steps=60` (not full epoch), no input truncation, invalid predictions skipped from metrics, `top_p=0`, `adamw_8bit`
   - Purpose: demonstrate that published fine-tuning results are inflated due to implementation bugs

4. **Fixed Fine-Tune**
   - Corrects all flaws: consistent `PROMPT_TEMPLATE`, no CoT prefix, uses `tokenizer.eos_token`, `num_train_epochs=1`, strict token truncation, `paged_adamw_8bit`, all predictions included in metrics
   - Represents what fine-tuning actually achieves when implemented correctly

---

## Research Narrative (Paper Claims)

1. The fine-tuning baseline from literature was implemented with significant flaws that inflate reported results. We reproduced it faithfully and then fixed it.
2. RAGTAG achieves competitive or superior performance to correctly-implemented fine-tuning, while requiring **no training** and **lower peak GPU memory**. We study RAGTAG's best configuration (optimal k and context window) across models and datasets.
3. VTAG establishes a **pure-retrieval floor** (macro-F1 = 0.6451) that RAGTAG must clear. Without VTAG, the obvious reviewer question — *"is your LLM doing anything beyond k-NN?"* — has no concrete answer. With it, the paper quantifies the LLM's marginal value as (RAGTAG − VTAG).

## VTAG → RAGTAG prompt-design insights (not yet implemented)

VTAG's per-class error pattern and k-curve suggest three concrete RAGTAG prompt tweaks worth testing. User is deciding whether to pursue; noted here so they aren't re-derived later.

1. **Reduce k from 15 to 7–9.** VTAG plateaus at k≈7 (F1=0.6427) and peaks at k=16 (0.6451) — +0.002 over 9 extra neighbors of mostly-noise. For RAGTAG, the LLM has to *read* that noise. Combined with the prompt-token analysis (k=15 only fits 44.7% of ctx=8k prompts; k=7 fits ~90%+), this is free tokens and less truncation.
2. **Inject the retrieval vote as an explicit prior.** After few-shot examples, add *"Among these examples, the label distribution is {bug: X, feature: Y, question: Z}."* VTAG proves this distribution alone carries 0.645 F1. Giving the LLM the k-NN vote explicitly lets it anchor when it agrees and reason about *why* when it disagrees. Biggest paper-worthy candidate — novel, cheap, directly ablatable.
3. **Bug-vs-feature disambiguation in the system prompt.** VTAG shows feature precision ~0.76 but recall ~0.55: features are systematically mislabeled as bugs. An explicit rule targeting this confusion pair should help. Example: *"If the issue requests new or improved functionality, label feature even when current behavior is called insufficient."*

---

## Datasets

| File | Size | Split |
|------|------|-------|
| `issues3k.csv` | ~3,000 issues | `--test_size 0.5` → ~1,497 test / ~1,498 train (balanced bug/feature/question) |
| `issues30k.csv` | ~30,000 issues | Used after best RAGTAG config is found on 3k, to test generalizability |

**Shared splits:** The FAISS indexing step saves `train_split.csv` and `test_split.csv` in the neighbors directory. Both fine-tuning scripts accept `--train_csv` and `--test_csv` to use these exact splits. All three approaches are evaluated on identical data.

---

## Models Tested

All models loaded via **Unsloth** (optimized inference + training):

| Model | Size | Notes |
|-------|------|-------|
| `unsloth/Llama-3.2-3B-Instruct` | 3B | Runs on local 4090 |
| `unsloth/Meta-Llama-3.1-8B-Instruct` | 8B 4-bit | Runs on local 4090 |
| `unsloth/Qwen2.5-14B-Instruct` | 14B 4-bit | Requires NRP GPU for fine-tuning |
| `unsloth/Qwen2.5-32B-Instruct` | 32B 4-bit | Requires NRP A100 |

---

## Infrastructure

### NRP (Nautilus Research Platform)
- Shared Kubernetes research cluster
- Access via JupyterHub pods
- GPUs available: **NVIDIA A6000 (48GB)** and **A100 (80GB)**
- Home directories may be on slow network storage → use `--nrp` flag to redirect HF model cache to `hf_cache/` inside the project folder
- Sessions can be killed → launch long experiments with `nohup ... &` to survive disconnects

### Local Machine
- **RTX 4090 (24GB VRAM)**
- Can run 3B and 8B models (RAGTAG + fine-tuning)
- 14B and 32B fine-tuning requires NRP

### `--nrp` flag
Sets `HF_HOME` / `TRANSFORMERS_CACHE` to `./hf_cache/` so model downloads stay on fast local storage inside the project directory.

---

## Pipeline Architecture

### Primary Entry Point: `run_experiment.sh`
Unified orchestrator for each dataset × model combination:
1. Runs RAGTAG (produces shared train/test splits)
2. Runs flawed fine-tune (uses shared splits)
3. Runs fixed fine-tune (uses shared splits)

Flags: `--skip_ragtag`, `--skip_flawed_ft`, `--skip_fixed_ft`
Resume logic: if prediction files already exist for a step, that step is skipped.

### Context Window Study: `context_experiment.sh`
Batch runner that calls `run_experiment.sh` four times with `max_seq_length=2048/4096/8192/16384`.
- Only the first run (2048) includes fine-tuning (fine-tune performance doesn't vary with RAGTAG's context window)
- Results go to separate directories: `results/issues3k_ctx2048/`, `results/issues3k_ctx4096/`, etc.

### `build_and_query_index.py`
- Loads dataset, deduplicates, stratified-splits into train/test
- Builds FAISS index from **training data only**
- Queries index to retrieve k nearest neighbors for each test issue
- Saves `train_split.csv` and `test_split.csv` for downstream fine-tuning
- Neighbor CSVs include a `neighbor_similarity` column (cosine sim from `IndexFlatIP` over L2-normalized vectors). Required by VTAG's weighted voting.

### `vtag.py`
- Non-LLM voting baseline. Reads a neighbors CSV with similarity scores, votes on labels, writes predictions in the same schema as RAGTAG for evaluate.py compatibility.
- Usage: run `build_and_query_index.py` once with a large-enough `--top_ks` (e.g. 30), then call `vtag.py --neighbors_csv ... --voting {similarity,shepard,majority} --ks 1,2,...,30`.
- Deterministic tie-break (label of highest-similarity tied neighbor). No seeds.
- Auto-invokes `evaluate.py` per k if `--eval_dir` is passed.

### `random_neighbors.py`
- Ablation tool: generates neighbor CSVs with **random** training examples instead of FAISS-retrieved ones.
- Same CSV schema as `build_and_query_index.py` output — drop-in replacement for `llm_labeler.py`.
- Accepts `--seeds` for multiple runs (default: 1,2,3) to account for variance.
- Output: `{output_dir}/seed{N}/neighbors_k{K}.csv` per seed.
- Purpose: proves RAGTAG's gain comes from retrieval quality, not just having few-shot examples.

### `analyze_prompt_tokens.py`
- Tokenizer-only analysis of RAGTAG prompt-token distributions (no GPU, no model weights).
- Reports percentiles + % of prompts fitting each context window (2k/4k/8k/16k).
- Uses real FAISS neighbors if `--neighbors_dir` is given, otherwise random train samples.
- Key finding: at k=15 only 44.7% of prompts fit in ctx=8k; at k=9 it's 67.5%. Informs k-vs-context-window tradeoffs.

### `llm_labeler.py`
- Loads model **once** via Unsloth, then runs all k values sequentially
- Chat format with few-shot examples as user/assistant turns
- Assistant prefill (`<label>`) for instruct models
- XML parsing + regex fallback for output extraction
- Smart truncation: compresses neighbor bodies proportionally when prompt exceeds `max_seq_length`
- Supports `--inference_batch_size` for batched GPU inference with left-padding

### `baseline_finetune_flawed.py`
- Accepts `--train_csv` and `--test_csv` for external splits
- All original flaws intentionally preserved (see above)
- Supports `--inference_batch_size`, `--max_new_tokens`
- Tracks absolute peak GPU memory across training + inference phases

### `fixed_fine-tune.py`
- Same external split support
- All flaws corrected (see above)
- Same batched inference and GPU tracking as flawed script

### `evaluate.py`
- Computes per-label precision/recall/F1, macro/weighted averages, accuracy, invalid prediction rate
- Works on prediction CSVs from any approach

### `run_pipeline.sh`
- RAGTAG-only orchestrator (predecessor to `run_experiment.sh`)
- Still functional but `run_experiment.sh` is the primary entry point

---

## Key CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_seq_length` | 16384 | Context window for RAGTAG inference. Tested: 2048, 4096, 8192, 16384 |
| `--ft_max_seq_length` | 2048 | Context window for fine-tuning (matches original paper) |
| `--max_new_tokens` | 50 | Max generated tokens for all approaches |
| `--inference_batch_size` | 1 | Batch size for GPU inference |
| `--top_ks` | `1,3,9,15` | Retrieved neighbors for RAGTAG. k=0 = zero-shot |
| `--test_size` | 0.5 | Test set fraction per label (issues3k uses 0.5) |
| `--nrp` | off | Redirect HF cache to `./hf_cache/` |

---

## Output Structure

Each run produces:
```
results/run_<timestamp>/
  neighbors/
    train_split.csv          # shared train split
    test_split.csv           # shared test split
    neighbors_k1.csv         # retrieved neighbors per test issue
    neighbors_k3.csv
    ...
  predictions/
    <model_tag>/
      preds_zero_shot.csv    # ground truth + predicted label + raw output + token counts
      preds_k1.csv
      ...
      cost_metrics.csv       # timing, token usage, GPU memory
  evaluations/
    <model_tag>/
      eval_k0.csv            # per-label and aggregate P/R/F1
      eval_k1.csv
      ...
  all_results.csv            # aggregated evaluations (single run)
  all_cost_metrics.csv       # aggregated cost metrics (single run)
  timing.csv                 # per-stage wall-clock time
```

Context window experiments use separate root dirs: `results/issues3k_ctx2048/`, etc.

---

## GPU Memory Tracking

Fine-tuning scripts:
1. Load model
2. Reset peak memory tracker
3. Run training → record `gpu_peak_memory_training_mb`
4. Run inference (no reset) → record `gpu_peak_memory_mb` (absolute max across training + inference)

This ensures the reported peak reflects true maximum VRAM usage for fair comparison against RAGTAG (which has no training phase).

---

## Experiment Design

### Context Window Study
Run `context_experiment.sh` to test RAGTAG at 4 context sizes × all k values × all models.
Fine-tuning only runs once (at ctx=2048) since it's unaffected by RAGTAG's context window.

### K Study
k=0 (zero-shot), 1, 3, 5, 9, 15 neighbors. Find the best k per model.

### Scale Generalization
After finding best config on `issues3k.csv`, re-run on `issues30k.csv` to verify results generalize.

---

## Known Issues / Current Status

- **No cross-directory aggregation:** Each context window experiment has its own `all_results.csv`. A script to merge results across `results/issues3k_ctx*/` for paper analysis still needs to be written.
- **VTAG complete** on issues3k with MiniLM+similarity (canonical), plus voting-scheme and embedder ablations (deferred to future work). Findings: [docs/VTAG_FINDINGS.md](docs/VTAG_FINDINGS.md).
- **Analysis / findings docs live in `docs/`** (tracked in git). Raw run artifacts stay under `results/` (gitignored). Any future analysis writeups should go in `docs/` so they survive `results/` being cleaned.
- Next steps: run data efficiency experiment, run Qwen debias, analyze final results, write paper.

### Data Efficiency Crossover Experiment
- **Scripts:** `subsample_and_index.py` + `run_data_efficiency.sh`
- **Subsample sizes:** 1.5k, 3k, 9k, 15k from 30k training pool (27k endpoint already exists)
- **Results dir:** `results/issues30k_efficiency/n{1500,3000,9000,15000}/`
- **Execution:** `--mode local` for RAGTAG (all models) + Llama FT; `--mode remote --nrp` for Qwen FT
- **Best RAGTAG configs:** Llama-3B k=3/ctx=8192, Llama-8B k=9/ctx=8192, Qwen-14B k=9/ctx=8192, Qwen-32B k=3/ctx=8192

### Debiased Retrieval — Qwen Models
- **Script:** `run_debias_qwen.sh`
- **Models:** Qwen-14B and Qwen-32B, both k=9, margin=3
- **Execution:** `--skip_30k` for local 3k runs; `--skip_3k --nrp` for remote 30k runs
- **Output:** integrates into existing `results/issues{3k,30k}_debias_m3/` alongside Llama results

### Remote Server Setup
- Server has code + `issues30k.csv` (NOT `results/` or `issues3k.csv`)
- Use `requirements-server.txt` for portable install (omits hardware-specific CUDA pins)
- Splits are deterministic — regenerated from dataset on first run

# Encoder Baselines Results (SANER revision, ESEM Review #126C)

**Date:** 2026-07-31 · **Branch:** `encoder-baselines` · **Seed:** 42 (single run, literature-faithful)
**Runner scripts:** `run_setfit.py`, `run_transformer_ft.py`, driver `run_encoder_baselines.sh`
**Analysis:** `scripts/paper/{_encoders,tab_encoder_baselines,significance_encoder_baselines}.py`
**Environment:** `venv-setfit/` (setfit 1.1.2, transformers 4.57.6, sentence-transformers 3.4.1, torch 2.13.0+cu126 — pinned in `requirements-setfit.txt`); RoBERTa in the main venv (transformers 5.5.0). GPU: RTX 4090 24 GB.

## TL;DR

**Reviewer C was right.** SetFit — the baseline in the very paper we cite
(`colavito2025benchmarking`) — **beats every LLM configuration in our paper** on the
11-project benchmark, at a small fraction of the cost:

- **SetFit (domain-adapted body), PS: macro F1 0.810** — the best number anyone has
  posted on this benchmark in our study, +0.025 over our best LLM config
  (Fine-Tune-PA Qwen-14B, 0.785) and +0.029 over BRAGTAG-32B (0.781).
- Every SetFit-vs-LLM paired bootstrap 95% CI is strictly positive; every McNemar
  p < 0.001. This is not a tie — SetFit **significantly outperforms** all 12 LLM
  configurations (RAGTAG/BRAGTAG/FT × 4 sizes).
- **RoBERTa-base** (0.771 PA) is statistically **tied** with our largest configs
  (FT-32B: diff +0.000, CI [−0.014, +0.016]; BRAGTAG-32B: −0.010, CI [−0.025, +0.005];
  FT-14B: −0.014, CI [−0.029, +0.000]) and significantly beats everything smaller —
  at 4 GB GPU and 4 minutes of training.
- SetFit inference over all 3,300 test issues takes **~10 seconds** vs 4.15 h for
  BRAGTAG-32B (≈1,500× faster) and produces **zero invalid outputs by construction**.

## Configurations (provenance)

| Baseline | Checkpoint | Config | Source |
|---|---|---|---|
| SetFit (mpnet) | `sentence-transformers/all-mpnet-base-v2` | batch 16, 1 epoch, num_iterations 20, LogisticRegression head, seed 42 | Colavito IST 2025 §3.6; JSS 2026; NLBSE'24 baseline |
| SetFit (issues) | `Collab-uniba/github-issues-mpnet-st-e10` | same | Colavito IST 2025 (their released domain-adapted body) |
| RoBERTa-base | `roberta-base` | lr 2e-5, batch 16, weight decay 0.01, 15 epochs, warmup 0.1, max_seq 512, seed 42 | Colavito JSS 2026 recipe (most recent fully-specified) |

Both settings per baseline: **PA** (train on pooled 3,300) and **PS** (300/project × 11,
pooled concat-then-evaluate — the paper's standard aggregation). Same train/test splits,
same `Title:\nBody:` input text as all other methods. Real `setfit` library (not
reimplemented). Text template and splits identical to the LLM pipeline.

## Results (raw predictions, pooled over 3,300 test issues)

| Method | Scope | Macro F1 | Acc | F1 bug | F1 feat | F1 q | RAM (GB) | Train | Infer |
|---|---|---|---|---|---|---|---|---|---|
| **SetFit (issues)** | **PS** | **0.810** | 0.810 | 0.810 | 0.850 | **0.770** | 21.0 | 59.3 m | 0.2 m |
| SetFit (mpnet) | PS | 0.805 | 0.806 | 0.801 | 0.848 | 0.767 | 14.1 | 43.2 m | 0.2 m |
| SetFit (mpnet) | PA | 0.797 | 0.798 | 0.800 | 0.840 | 0.751 | 14.1 | 43.3 m | 0.2 m |
| SetFit (issues) | PA | 0.794 | 0.794 | 0.791 | 0.839 | 0.751 | 21.0 | 59.4 m | 0.2 m |
| RoBERTa-base | PA | 0.771 | 0.775 | 0.786 | 0.825 | 0.704 | 4.8 | 4.2 m | 0.1 m |
| RoBERTa-base | PS | 0.757 | 0.758 | 0.763 | 0.807 | 0.700 | 4.8 | 4.2 m | 0.1 m |

Paper's existing best configs for reference (raw, pooled): RAGTAG-PS 0.697/0.718/0.732/0.767
(3B/7B/14B/32B), BRAGTAG-PS 0.714/0.738/0.756/0.781, Fine-Tune-PA 0.708/0.762/0.785/0.771.
Best LLM cell overall: FT-14B 0.785 (16.7 GB, 1.71 h total).

Notes:
- **PS > PA for SetFit** (+0.008/+0.016) — mirrors our RAGTAG finding that per-project
  data is sufficient and cross-project pooling adds little; strengthens the paper's
  "PS data scope is enough" narrative, now across three method families.
- RoBERTa is the reverse (PA > PS): 300 examples/project is thin for full fine-tuning
  of a 125M encoder, consistent with SetFit's few-shot design advantage.
- **Question class:** SetFit-PS reaches 0.767–0.770 question F1 — better than every
  LLM config except none (best LLM question F1: FT-14B 0.736, BRAGTAG-32B 0.729).
  The literature's "question is hardest" pattern holds for every method family.
- No mode collapse anywhere: all three classes predicted in near-balanced proportions,
  per-class recalls healthy, probability outputs vary per issue. (Contrast with the
  archived DeBERTa-v3-large run, which produced a constant softmax.)
- Times exclude model load (project convention). PS times = sum over 11 per-project runs.
  RAM = peak GPU memory, absolute-peak convention (train then infer without reset).
  SetFit-issues' higher RAM comes from its longer body max_seq_length (512 vs 384).

## Significance (paired bootstrap 95% CI, 1,000 resamples; McNemar; TOST)

Full grid in `scripts/paper/significance_encoder_baselines.py` output. Summary
(diff = encoder − LLM method, macro F1, aligned 3,300-row pairing):

- **SetFit (issues) PS vs every LLM config:** diff +0.025 … +0.113, all CIs strictly
  positive, all McNemar p < 0.001. Closest race: vs FT-14B (+0.025 [+0.009, +0.039]).
- **SetFit (mpnet) PS vs every LLM config:** diff +0.020 … +0.108, all CIs strictly
  positive, all p < 0.01.
- **RoBERTa-base PA:** significantly better than all 3B/7B configs and RAGTAG at all
  sizes; statistically indistinguishable from FT-7B (+0.010 [−0.005, +0.024]),
  FT-14B (−0.014 [−0.029, +0.000]), BRAGTAG-32B (−0.010 [−0.025, +0.005]), and
  FT-32B (+0.000 [−0.014, +0.016], TOST δ=0.02 PASS).

## What this means for the paper

The honest conclusion is the one the plan anticipated (and the one Colavito et al.
themselves draw): **the generative LLM does not pay for itself on standard 3-class IRC
when a few hundred labeled examples per project exist.** The paper's contribution must
be reframed accordingly — this makes the paper *stronger and more honest*, not weaker:

1. **RAGTAG's niche is the no-training / cold-start / dynamic regime.** SetFit needs a
   training run per project (and re-training as labels accumulate or drift); RAGTAG
   needs only an index insert. RAGTAG works with an off-the-shelf model swap; SetFit's
   head is model-bound. The deployment-cost argument (retraining cadence, model churn)
   survives; the raw-performance argument does not.
2. **The comparison table gets three new rows and the cost axis becomes the story:**
   macro F1 per GPU-hour and per GB. SetFit-PS: 0.810 at ~1 h / 21 GB (or 0.805 at
   43 m / 14 GB); best LLM: 0.785 at 1.71 h / 16.7 GB (FT-14B) or 0.781 at 4.15 h /
   22.3 GB (BRAGTAG-32B).
3. **Label-topic misalignment analysis (plan P0.2) becomes more central:** every method
   family — kNN vote, encoder, generative LLM — struggles most on question, so the
   failure is in the data/embedding geometry, not the classifier family. That is a
   defensible, durable finding.
4. **RQ4 should become "how much does the LLM buy, and when is it worth it?"** with
   SetFit as the cost-effective reference point, rather than "can RAG match FT?".
5. Optional next experiment if we want RAGTAG to compete on its own turf: sample-
   efficiency curve (SetFit vs RAGTAG at 5→100 labels/project) — the literature says
   SetFit saturates at ~40/class, but RAGTAG at k≤9 needs far fewer *retrieved*
   examples; the cold-start region (<20 labels) may be RAGTAG's win. Not yet run.

## Reproduction

```bash
bash run_encoder_baselines.sh                      # all 36 cells, idempotent
venv/bin/python scripts/paper/tab_encoder_baselines.py
venv/bin/python scripts/paper/significance_encoder_baselines.py
```

# Paper Narrative & Structure

**Status (2026-05-04):** This doc has been trimmed. The pre-2026-04-29 sections (Llama-as-active-models, retired three-phase narrative, Apr 25 paper-structure draft) have been removed because they actively contradicted the canonical refresh below. For the most current state of the paper and experiments, read [paper/SESSION_HANDOFF.md](../paper/SESSION_HANDOFF.md) first; this doc remains as the discussion-thread reference.

> Active model lineup is **Qwen-only** (3B, 7B, 14B, 32B; uniform bnb-4bit). Llama-3B/8B legacy data is on disk but excluded from the paper. The narrative spine below assumes that lineup.

---

## Working Title

**"RAGTAG: When Does Retrieval-Augmented Few-Shot Classification Match Fine-Tuning for GitHub Issue Triage?"**

---

## Story arc (canonical)

> Fine-tuning LLMs has been the dominant approach in prior work on issue-report classification, but it is computationally expensive and data-hungry. We propose two retrieval-based alternatives — VOTAG (no LLM, voting-only) and RAGTAG (retrieval + LLM reasoning) — and a context-engineering intervention applied to RAGTAG that closes the FT gap without any training.

The paper is organised around three coupled research questions on a single 11-project benchmark, evaluated across four Qwen2.5-Instruct sizes (3B / 7B / 14B / 32B, uniform bnb-4bit).

- **RQ1.** Establish VOTAG as a fast, near-zero-cost retrieval-only baseline; identify its plateau; use it to justify the RAGTAG k grid.
- **RQ2.** Comprehensive comparison of VOTAG, RAGTAG, and Fine-Tune — strengths, weaknesses, Pareto frontier on GPU-time/cost, qualitative + quantitative failure analysis surfacing systematic bug-bias.
- **RQ3.** Introduce a context-engineering intervention; compare against fine-tuning; ablate against the no-LLM voting variant to isolate the LLM-reasoning component.

> **2026-05-04 update.** The originally-planned RQ3 method (margin-based retrieval debiasing, m=3) has been demoted from the headline. After the 3-epoch FT campaign tightened the FT baseline, margin-debiasing's gains shrank and the user judged the method to plateau. A replacement context-engineering method is being brainstormed (Gemini consultation outstanding); margin-debiasing may stay in the paper as an early-attempt baseline rather than the main RQ3 result. Treat the original RQ3 wording in §3.6 of the bullet structure below as **placeholder**.

## Core findings (one-line each, all evidence-backed against the 1-epoch FT data; FT rows pending update once 3-epoch campaign lands)

1. **VOTAG is a competitive non-LLM floor.** Pure similarity-weighted k-NN reaches macro F1 ≈ 0.604 (agnostic) and 0.584 (project-specific mean) at the plateau (k ≈ 13 ag / k ≈ 7 ps), in negligible compute (≈10 ms per cell).
2. **The RAGTAG k grid is anchored by VOTAG, not arbitrary.** RAGTAG's chosen k ∈ {1, 3, 6, 9} brackets VOTAG's climb (k=1..6) plus the entry to its plateau (k=9).
3. **Adding the LLM is significant at every scale.** RAGTAG−VOTAG gaps in macro F1: 3B +0.10, 7B +0.11, 14B +0.12, 32B +0.16, all p < 1e-24 (McNemar). The advantage grows with model size.
4. **Bug-bias is a question→bug misclassification, more LLM than retrieval.** Top-k retrieval is roughly balanced for non-bug ground truth (≈30 % bug fraction at k=9), but every LLM still over-predicts bug by 24–54 %; question recall in zero-shot is as low as 0.39 on Qwen-32B before retrieval helps.
5. **Debiased RAGTAG closes the FT gap from 14B up (vs 1-epoch FT).** Debias-ps at k=9 beats RAGTAG-ps consistently (Δ +0.02 to +0.03, p < 0.001 on 7B/14B/32B). Caveat: 1-epoch FT was under-trained; the 3-epoch re-run will tighten this comparison and the headline may change.
6. **VOTAG-debias makes a bug↔question trade; RAGTAG-debias rescues true bugs.** VOTAG-debias loses ~0.19 bug recall to gain ~0.16 question recall (1:1 trade, +0.006 macro F1). RAGTAG-debias on Qwen-7B+ loses essentially no bug recall while gaining 5–8 % question recall (+0.020 macro F1). The LLM rescue rate climbs from 48 % at 3B to 81–85 % at 7B+. This is the cleanest available evidence that the LLM does substantive reasoning over rebalanced examples.
7. **FT-project-specific collapses on small/mid models (1-epoch).** Qwen-7B FT-ps drops to F1 0.51, 0.23 below FT-ag. Per-project FT was unstable except at 32B. Under 3 epochs the gap narrows: PS-avg lifts to 0.665 (3B) and 0.677 (7B), but per-project FT is still volatile.
8. **Pareto: agnostic FT is on the frontier; project-specific FT is dominated.** RAGTAG/debias-ps occupy the Pareto frontier in the project-specific setting from end to end. Debias-ps uses 1.2–2.0× more total GPU-seconds than FT-ag because of long retrieval prompts, but requires no training step.

## Discussion threads (outline only — to be written into prose by the user)

### Thread A — Practitioner decision tree
- **No training infra, single shared classifier:** RAGTAG-ag at the largest size you can afford. At 32B, RAGTAG-ag k=9 hits 0.759, beating 1-epoch FT-ag 0.746 (CI overlaps zero — ~tie). Pending 3-epoch update.
- **Training infra available, want one model for all projects:** FT-ag is competitive and cheaper at inference (no retrieval prompts). 7B+ FT-ag is on the Pareto frontier; 3-epoch campaign expected to widen its lead.
- **Per-project tuning desired:** debias-ps at the largest size you can afford. FT-ps was unstable at 7B/14B under 1 epoch; debias-ps wins 9–11/11 projects vs FT-ps regardless of model size. Recheck after 3-epoch lands.
- **Latency-critical, low budget:** VOTAG is the no-cost floor. At ≈0.60 macro F1, it's already 0.06 above 3B zero-shot and only ~0.16 below the 32B ceiling.

### Thread B — Why bug-bias arises (mechanism hypotheses, supported by the data)
- **LLM-side prior, not retrieval-side imbalance.** Top-k retrieval is roughly class-balanced for non-bug ground truth. The 24–54 % bug over-prediction shows up even with balanced few-shots.
- **Most labeling-noise is question→bug, not feature→bug.** Of 264 consensus-failure issues (all 4 Qwen sizes mislabel), 79 % were question→bug, 21 % feature→bug.
- **User framing influences the label drift.** Many consensus-failure issues contain explicit "Bug:" framing in the title or body, even though the maintainer's final label is feature/question. The LLM correctly reads the user's framing but misses the maintainer's reclassification — a labeling-mismatch problem more than a model error.

### Thread C — Data-efficiency reframed
- The headline "300 retrieval examples per project beats 3,300 training examples" was true on macro F1 from 14B up under 1-epoch FT. Pending recheck.
- It costs more inference compute (because 11 × per-project inferences with long retrieval prompts > 1 × training-once + agnostic inference).
- The right framing is **operational simplicity** (no gradient updates, no training pipeline, swap models trivially) rather than absolute compute reduction.
- For low-resource projects, debias-ps is uniquely valuable: it needs only 300 labeled examples to outperform any FT-ps trained on the same 300 issues.

### Thread D — Why VOTAG-debias works less than RAGTAG-debias (the LLM-as-rescuer mechanism)
- The same retrieval-time intervention gives VOTAG a marginal +0.006 macro F1 (project-specific +0.007), but RAGTAG +0.017 to +0.032 depending on model size and k.
- The LLM rescue rate analysis (RQ3.5) directly shows the mechanism: where retrieval rebalancing flips an issue from bug to question, the LLM correctly says "bug" (rescuing the true label) 81–85 % of the time at 7B+, vs 48 % at 3B.
- This is the cleanest available evidence that the LLM does substantive reasoning over rebalanced examples — it isn't just consuming a class-balanced prior.

### Thread E — Limitations
- Single random seed throughout. Multi-seed validation is the most important follow-up; bootstrap CIs partly compensate but do not capture training-init variance.
- Qwen-only, bnb-4bit only — model-family generalization is not directly demonstrated. Llama-3B/8B legacy data on disk hints at portability but is excluded for cleanliness.
- Three-class label space; richer label sets (e.g., "documentation", "performance", "security") may have different dynamics.
- English-language repos only.
- No temporal split — train and test issues coexist in the same time window per project.

### Thread F — Future work
- Multi-seed validation across all cells.
- Voting-scheme ablation for VOTAG (similarity vs Shepard vs majority) — only similarity was run on 11k.
- A second / replacement RQ3 intervention method. As of 2026-05-04, context-engineering candidates are being brainstormed (see [paper/SESSION_HANDOFF.md](../paper/SESSION_HANDOFF.md) §4 for the constraints).
- Out-of-distribution evaluation: train-on-N-projects, test-on-held-out-projects.

---

## Evidence

The previous `docs/analysis/` directory (CSVs, figures, leaderboard tables) was deleted on 2026-05-04 because it was built from 1-epoch FT data and an older RQ3 narrative we are no longer committed to. New analysis will be regenerated from `results/issues11k/` once the 3-epoch FT campaign completes; the destination directory and naming scheme are TBD and may change. Use [paper/SESSION_HANDOFF.md](../paper/SESSION_HANDOFF.md) for current numbers.

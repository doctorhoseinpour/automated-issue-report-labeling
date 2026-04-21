# RAGTAG: When RAG Beats Fine-Tuning for GitHub Issue Labeling
## Professor Meeting — April 21, 2026

---

## Slide 1: Project Overview

**Problem:** Automated GitHub issue classification (bug / feature / question)

**Research question:** At what labeled-data budget does fine-tuning overtake training-free RAG classification, and why?

**Three-phase study:**
1. **Phase 1 (3k dataset):** Develop RAGTAG, compare approaches, discover the bug bias
2. **Phase 2 (30k dataset):** Validate at scale, data efficiency crossover
3. **Phase 3 (bias correction):** Debiased retrieval to push RAGTAG's ceiling higher

---

## Slide 2: Four Approaches Compared

| Approach | Training? | GPU Needed? | Key Idea |
|---|---|---|---|
| **VTAG** (voting baseline) | No | No | k-NN vote on retrieved neighbors — the retrieval floor |
| **RAGTAG** (our method) | No | Inference only | Retrieve similar issues → few-shot prompt → LLM classifies |
| **Fine-Tuning** | Yes | Train + inference | LoRA fine-tune on labeled data |
| **Zero-Shot** | No | Inference only | LLM classifies with no examples |

**Models tested:** Llama-3B, Llama-8B (4-bit), Qwen-14B (4-bit), Qwen-32B (4-bit)

---

## Slide 3: Phase 1 Results — RAGTAG Beats FT on 3k

| Model | RAGTAG F1 | FT F1 | Winner | RAGTAG VRAM |
|---|---|---|---|---|
| Llama-3B | 0.674 | 0.667 | RAGTAG (+0.007) | 70-80% of FT |
| Llama-8B | 0.712 | 0.687 | RAGTAG (+0.025) | 70-80% of FT |
| Qwen-14B | 0.742 | 0.739 | RAGTAG (+0.004) | 70-80% of FT |
| Qwen-32B | 0.778 | 0.735 | RAGTAG (+0.043) | 70-80% of FT |

- VTAG retrieval floor: **0.645** — all RAGTAG configs beat it
- LLM marginal value over VTAG: +4.5% (3B) to +20.5% (32B)
- Random neighbor ablation: dropping FAISS for random retrieval costs 0.05-0.08 F1

---

## Slide 4: Configuration Analysis (3k)

**k x Context Window interaction:**
- Best k varies by model: k=3 for 3B/32B, k=9 for 8B/14B
- ctx=8192 best raw F1; ctx=4096 is Pareto-optimal (13-22% less VRAM, within 0.01 F1)
- At ctx=2048 + k>=5: truncation wall — invalid rates hit 28-44%

**Cost-performance:**
- RAGTAG uses 70-80% of FT's peak VRAM
- Zero training cost — fits on consumer RTX 4090 for all models
- ctx=4096 faster than FT total (training + inference) for most models

---

## Slide 5: Phase 2 — The 30k Flip

On 30k (~27k train), fine-tuning pulls ahead for smaller models:

| Model | RAGTAG F1 | FT F1 | Gap | Winner |
|---|---|---|---|---|
| Llama-3B | 0.722 | 0.790 | -0.067 | FT |
| Llama-8B | 0.744 | 0.793 | -0.048 | FT |
| Qwen-14B | 0.779 | 0.767 | +0.012 | **RAGTAG** |
| Qwen-32B | 0.767 | 0.810 | -0.043 | FT |

**Why?** RAGTAG always shows k examples per prompt regardless of pool size. FT trains on ALL 27k examples. The fixed prompt budget is a structural ceiling.

**Question F1 is the bottleneck** — accounts for ~50% of the macro-F1 gap.

---

## Slide 6: The Bug Bias — Root Cause Discovery

**Zero-shot reveals the problem:**
- Bug recall: 0.88-0.95 across all models
- Question recall: 0.19-0.44 across all models
- The model's prior: "problem description = bug report"

**Retrieval is NOT the bottleneck:**
- 42% of question→bug errors have retrieval correctly favoring question
- The LLM overrides correct retrieval signal 35-45% of the time across all models
- Only 7.4% of errors have genuine retrieval failure (zero question neighbors)

**The bias is parametric** — it lives in the model's activation space, not its text interpretation.

---

## Slide 7: Failed Interventions — Six Approaches, Four Levels

| # | Intervention | Level | Result | Why It Failed |
|---|---|---|---|---|
| 1 | Vote prior injection | Prompt | Zero effect | LLM ignores statistical evidence in text |
| 2 | Enhanced system prompt | Prompt | No improvement | Small models can't follow meta-reasoning |
| 3 | Post-hoc ensemble (RAGTAG+VTAG) | Output | +0.010 max | Margins too slim for reliable rules |
| 4 | Model ensemble (3B+8B) | Output | +0.007 over best single | Two biased models can't unbias each other |
| 5 | Batch Calibration (BC) | Logit | +0.009 to +0.030 | Consistent but modest — re-scales logits but doesn't fix the underlying geometry |
| 6 | Contrastive Decoding (CD) | Logit | **Catastrophic** (F1 drops to 0.26–0.42) | Amplifies noise; subtracting amateur logits destroys good signal |

- BC+CD combined: CD dominates destructively, negating BC gains
- **Pattern:** Prompt-level, output-level, and logit-level interventions all fail to overcome the parametric bug bias. The bias is geometric — it lives in the activation space, not in text interpretation or output probabilities.

---

## Slide 8: What Worked — Debiased Retrieval

**Idea:** Bug neighbors *reinforce* the parametric prior instead of providing corrective signal. Remove them when evidence is ambiguous.

**Mechanism:** If `bug_count - question_count <= margin(3)`: remove all bug neighbors.

**3k Results (all 4 models complete):**

| Model | Baseline F1 | Debias F1 | Delta | Question Recall Gain |
|---|---|---|---|---|
| Llama-3B | 0.674 | 0.693 | +0.019 | +0.122 |
| Llama-8B | 0.712 | 0.758 | +0.046 | +0.209 |
| Qwen-14B | 0.742 | 0.768 | +0.026 | — |
| Qwen-32B | 0.778 | 0.793 | +0.015 | — |

Works on all 4 models — consistent +0.015 to +0.046 F1 gain.

**30k Results (Llama complete, Qwen running on server):**

| Model | Baseline F1 | Debias F1 | Delta |
|---|---|---|---|
| Llama-3B | 0.722 | 0.724 | +0.002 |
| Llama-8B | 0.743 | 0.757 | +0.014 |
| Qwen-14B | 0.779 | *running* | — |
| Qwen-32B | 0.767 | *running* | — |

Narrows FT gap but doesn't close it — confirms ceiling is structural.

---

## Slide 9: Activation Steering (Mechanistic Evidence)

**CAA (Contrastive Activation Addition)** on Llama-3B 3k:
- Best F1: 0.706 at layer 23, multiplier=-1.0 (+0.032 over baseline)
- Confirms bias lives in the residual stream geometry
- Optimal layer at 82% depth — classification decisions made late in network

Not in main results (only tested on one model), but supports the mechanistic story in Discussion.

---

## Slide 10: Data Efficiency Crossover — Llama Results (Complete)

**Setup:** Stratified subsamples (1.5k / 3k / 9k / 15k / 27k) from 30k training pool, fixed 3k test set.

**Figure description:** 2-panel plot (Llama-3B | Llama-8B), RAGTAG (blue) vs FT (red) curves.

| Train Size | Llama-3B RAGTAG | Llama-3B FT | Llama-8B RAGTAG | Llama-8B FT |
|---|---|---|---|---|
| 1.5k | 0.698 | 0.714 | 0.718 | 0.748 |
| 3k | 0.699 | **0.679** | 0.719 | 0.768 |
| 9k | 0.705 | 0.767 | 0.726 | 0.779 |
| 15k | 0.714 | 0.790 | 0.726 | 0.788 |
| 27k | 0.722 | 0.790 | 0.743 | 0.792 |

**Key findings:**
- **RAGTAG curve is nearly flat** — gentle slope from 0.698→0.722 (3B) and 0.718→0.743 (8B). Prompt budget is fixed regardless of pool size.
- **FT curve rises steeply** — 0.714→0.790 (3B) and 0.748→0.792 (8B). Thousands of gradient updates scale with data.
- **Llama-3B crossover at ~3k:** RAGTAG wins at n=3k (+0.021) where FT dips to 0.679. FT pulls ahead from 9k onward.
- **Llama-8B: FT leads at all sizes** — even at 1.5k, FT already beats RAGTAG by 0.030.
- FT plateaus around 15k–27k for both models.

---

## Slide 11: Data Efficiency Crossover — Qwen RAGTAG (FT pending from server)

**RAGTAG results complete for all 4 models. Qwen FT running on OSC H100.**

| Train Size | Qwen-14B RAGTAG | Qwen-14B FT | Qwen-32B RAGTAG | Qwen-32B FT |
|---|---|---|---|---|
| 1.5k | 0.778 | *running* | 0.782 | *running* |
| 3k | 0.776 | *running* | 0.780 | *running* |
| 9k | 0.768 | *running* | 0.782 | *running* |
| 15k | 0.775 | *running* | 0.779 | *running* |
| 27k | 0.779 | 0.767 | 0.785 | 0.810 |

**Preliminary observation:**
- Qwen RAGTAG is **almost perfectly flat** (0.768–0.782) — larger models extract maximum value from even small retrieval pools
- At 27k: Qwen-14B RAGTAG still beats FT (+0.012), but Qwen-32B FT wins (-0.025)
- **Hypothesis:** Larger model = higher crossover point (more data needed before FT overtakes)

---

## Slide 12: Paper Structure

**Title:** "RAGTAG: When Retrieval-Augmented Classification Beats Fine-Tuning for GitHub Issue Labeling"

| Section | Content | Data Status |
|---|---|---|
| 1. Introduction | Practical problem, two paradigms, contributions | Done |
| 2. Background | Related work, gap in literature | Done |
| 3. Approach | RAGTAG, FT, VTAG, debiased retrieval | Done |
| 4. Experimental Setup | Datasets, models, metrics | Done |
| 5. RAGTAG Analysis (3k) | k x ctx, vs FT/VTAG, random ablation | **Complete** |
| 6. Cost-Performance | Pareto frontier, time, VRAM | **Complete** |
| 7. Bug Bias | Discovery, diagnosis, mechanistic interpretation | **Complete** |
| 8. Data Efficiency (30k) | Config transfer, crossover plot | **Llama complete, Qwen FT running** |
| 9. Debiased Retrieval | Mechanism, results all 4 models | **3k complete, 30k Qwen running** |
| 10. Discussion | Decision framework, structural ceiling | Ready to write |
| 11-13. Threats/Future/Conclusion | | Ready to write |

---

## Slide 13: Three Key Takeaways

1. **Below a data threshold → use RAGTAG.** No training, 70-80% of FT's VRAM, competitive or superior F1. The threshold is model-dependent.

2. **Above the threshold → fine-tuning wins.** The advantage is structural: thousands of gradient updates vs a fixed k-example prompt budget. No prompt engineering closes this gap.

3. **Debiased retrieval pushes the threshold higher** by partially correcting the parametric bug bias that limits all training-free LLM classifiers.

---

## Slide 14: Timeline & Status

| Task | Status |
|---|---|
| Phase 1 experiments (3k, all 4 models) | Done |
| Phase 2 experiments (30k at 27k, all 4 models) | Done |
| All 6 failed interventions tested | Done |
| Activation steering (CAA) — mechanistic evidence | Done |
| Debiased retrieval — Llama (3k + 30k) | Done |
| Debiased retrieval — Qwen (3k) | **Done** |
| Data efficiency crossover — Llama (RAGTAG + FT) | **Done** |
| Data efficiency crossover — Qwen RAGTAG | **Done** |
| Data efficiency crossover — Qwen FT | **Running on OSC H100** |
| Debiased retrieval — Qwen (30k) | **Running on OSC H100** |
| Write paper | Next (all data expected within 24h) |

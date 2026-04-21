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

## Slide 7: Failed Interventions

| Intervention | Level | Result | Why It Failed |
|---|---|---|---|
| Vote prior injection | Prompt | Zero effect | LLM ignores statistical evidence |
| Enhanced system prompt | Prompt | No improvement | Small models can't follow meta-reasoning |
| Post-hoc ensemble | Output | +0.010 max | Margins too slim for strong rules |
| Model ensemble (3B+8B) | Output | +0.007 over best | Two biased models can't unbias each other |

**Pattern:** All text-level and output-level interventions fail. The bias is geometric — it lives in the activation space, not text interpretation.

---

## Slide 8: What Worked — Debiased Retrieval

**Idea:** Bug neighbors *reinforce* the parametric prior instead of providing corrective signal. Remove them when evidence is ambiguous.

**Mechanism:** If `bug_count - question_count <= margin(3)`: remove all bug neighbors.

**3k Results:**

| Model | Baseline F1 | Debias F1 | Delta | Question Recall Gain |
|---|---|---|---|---|
| Llama-3B | 0.674 | 0.697 | +0.023 | +0.122 |
| Llama-8B | 0.712 | 0.756 | +0.044 | +0.209 |

**30k Results:**

| Model | Baseline F1 | Debias F1 | Delta |
|---|---|---|---|
| Llama-3B | 0.722 | 0.727 | +0.005 |
| Llama-8B | 0.743 | 0.757 | +0.014 |

Narrows FT gap but doesn't close it — confirms ceiling is structural.
Qwen-14B and Qwen-32B debias: **running now**.

---

## Slide 9: Activation Steering (Mechanistic Evidence)

**CAA (Contrastive Activation Addition)** on Llama-3B 3k:
- Best F1: 0.706 at layer 23, multiplier=-1.0 (+0.032 over baseline)
- Confirms bias lives in the residual stream geometry
- Optimal layer at 82% depth — classification decisions made late in network

Not in main results (only tested on one model), but supports the mechanistic story in Discussion.

---

## Slide 10: Data Efficiency Crossover (THE HERO EXPERIMENT)

**Status: Running now** on local RTX 4090 + OSC H100

Subsample 30k training pool at 1.5k / 3k / 9k / 15k / 27k:
- RAGTAG curve: gentle slope (prompt budget is fixed)
- FT curve: steep rise with data
- Crossover point depends on model scale

**Expected figure:** 4 panels (one per model), two curves each

| Already have | Still running |
|---|---|
| 27k endpoints (all models) | 1.5k, 3k, 9k, 15k RAGTAG (all models) |
| 3k development results | 1.5k, 3k, 9k, 15k FT (all models) |
| | Qwen debias on 3k and 30k |

---

## Slide 11: Paper Structure

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
| 8. Data Efficiency (30k) | Config transfer, crossover plot | **Running** |
| 9. Debiased Retrieval | Mechanism, results all 4 models | **Partially complete** |
| 10. Discussion | Decision framework, structural ceiling | Ready to write |
| 11-13. Threats/Future/Conclusion | | Ready to write |

---

## Slide 12: Three Key Takeaways

1. **Below a data threshold → use RAGTAG.** No training, 70-80% of FT's VRAM, competitive or superior F1. The threshold is model-dependent.

2. **Above the threshold → fine-tuning wins.** The advantage is structural: thousands of gradient updates vs a fixed k-example prompt budget. No prompt engineering closes this gap.

3. **Debiased retrieval pushes the threshold higher** by partially correcting the parametric bug bias that limits all training-free LLM classifiers.

---

## Slide 13: Timeline

| Task | Status |
|---|---|
| Phase 1 experiments (3k) | Done |
| Phase 2 experiments (30k at 27k) | Done |
| All failed interventions tested | Done |
| Debiased retrieval (Llama) | Done |
| Data efficiency crossover | **Running (local + server)** |
| Debiased retrieval (Qwen) | **Running (local + server)** |
| Write paper | Next |

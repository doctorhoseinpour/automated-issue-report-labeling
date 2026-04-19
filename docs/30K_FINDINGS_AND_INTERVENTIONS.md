# 30k Findings, Failed Interventions, and Next Steps

**Date:** 2026-04-19
**Status:** Active — documenting all 30k results and intervention attempts to date

---

## 1. The 30k Result: Fine-Tuning Pulls Ahead

On 3k, RAGTAG beat fine-tuning on all 4 models. On 30k, the relationship flips for smaller models.

### Headline Table (30k, best config per model)

| Model | RAGTAG F1 | RAGTAG k | FT F1 | Gap (R-FT) | Winner |
|---|---|---|---|---|---|
| Llama-3B | 0.7222 | k=3 | 0.7897 | -0.067 | FT |
| Llama-8B | 0.7442 | k=9 | 0.7925 | -0.048 | FT |
| Qwen-14B | 0.7790 | k=9 | 0.7668 | +0.012 | **RAGTAG** |
| Qwen-32B | 0.7669 | k=3 | pending | — | — |

**Qwen-14B is the exception:** RAGTAG beats fine-tuning by +0.012. FT overcorrects — bug recall drops to 0.669 (268 bugs misclassified as question). Fine-tuning on 14B learned the question boundary *too aggressively*.

### Per-Class Breakdown: Where the Gap Lives

| Model | Approach | F1_bug | F1_feature | F1_question |
|---|---|---|---|---|
| Llama-3B | RAGTAG k=3 | 0.739 | 0.784 | 0.645 |
| Llama-3B | FT | 0.792 | 0.824 | 0.753 |
| Llama-3B | **Gap** | **-0.053** | **-0.041** | **-0.108** |
| Llama-8B | RAGTAG k=9 | 0.766 | 0.811 | 0.652 |
| Llama-8B | FT | 0.775 | 0.835 | 0.768 |
| Llama-8B | **Gap** | **-0.009** | **-0.024** | **-0.116** |

**Question F1 is the dominant bottleneck**, accounting for ~50% of the total macro-F1 gap. Bug and feature gaps are smaller.

---

## 2. Why RAGTAG Loses at Scale (Root Cause)

### The scaling asymmetry

| Factor | RAGTAG | Fine-Tuning |
|---|---|---|
| Training data utilization | Fixed: shows k=3-9 examples per prompt | Scales: trains on all ~27k examples |
| Decision boundary learning | Must infer from handful of examples | Learns from thousands of gradient updates |
| Question-vs-bug intent | Relies on LLM's general reasoning | Learns the specific intent boundary from data |

RAGTAG's prompt budget is fixed regardless of training pool size. Whether the pool is 1.5k or 27k, the model still only sees a handful of examples per inference. The retrieval pool quality improves (better neighbors), but the LLM's exposure to the decision boundary doesn't scale.

### Evidence: 3k vs 30k improvement over zero-shot

| Model | RAGTAG Δ over ZS (3k) | RAGTAG Δ over ZS (30k) |
|---|---|---|
| Llama-3B | +0.097 | +0.053 |
| Llama-8B | +0.091 | +0.074 |

The zero-shot floor is higher on 30k (better retrieval pool lifts even k=0 via the system prompt), but RAGTAG's *marginal* gain shrinks. The LLM extracts less value from examples when the zero-shot baseline is already stronger.

---

## 3. The Core Error: LLM Bug Bias

### Cross-model confirmation (30k)

| Model | question→bug errors | Retrieval correct (q>=bug) | % LLM ignored |
|---|---|---|---|
| Llama-3B (k=3) | 273 | 95 | 34.8% |
| Llama-8B (k=9) | 326 | 137 | 42.0% |
| Qwen-14B (k=9) | 272 | 113 | 41.5% |
| Qwen-32B (k=3) | 224 | 101 | 45.1% |

**35-45% of question→bug errors have retrieval correctly favoring question, but the LLM overrides.**

The model's prior: "problem description = bug." Issues describing errors, crashes, or failures get labeled bug regardless of whether the user is reporting a defect or asking for help.

### The similarity paradox

Higher embedding similarity correlates with MORE question errors, not fewer. The embedding space captures topic similarity (both bugs and questions discuss the same software problems), not intent similarity. Retrieval brings back topically similar issues but can't distinguish "reporting a defect" from "seeking help."

### Universal errors

145 questions are misclassified by ALL 4 RAGTAG models. Many are genuinely ambiguous — titles like "FFmpeg session crashes" or "poetry lock fails" could be either bug reports or help-seeking questions depending on context that's often absent from the issue body.

---

## 4. K-Value Study on 30k (Llama-3B and Llama-8B)

Tested k=1,3,5,9,15 with ctx=8192 on 30k to verify 3k→30k k transfer.

| Model | k=1 | k=3 | k=5 | k=9 | k=15 | Best k |
|---|---|---|---|---|---|---|
| Llama-3B | **0.7265** | 0.7247 | 0.7256 | 0.6922 | 0.7052 | k=1 (≈k=3) |
| Llama-8B | 0.7280 | 0.7386 | 0.7351 | **0.7442** | 0.7342 | k=9 |

**Findings:**
- k doesn't matter much on 30k — spread is ~0.03 (3B) and ~0.016 (8B)
- Best k from 3k transfers correctly (Llama-3B k=3 ≈ best, Llama-8B k=9 = best)
- Bug recall climbs with k (0.807→0.892 for 3B), question recall drops (0.623→0.496)
- Invalid rate increases with k: 0% at k=1, 2.9% at k=15
- No k value closes the FT gap

---

## 5. Failed Interventions

### 5a. Vote Prior Injection

**Idea:** After few-shot examples, inject "Among these examples, the label distribution is {bug: X, feature: Y, question: Z}." VTAG proves this distribution alone carries 0.645 F1 — giving it to the LLM explicitly should help.

**Result:** Zero effect. F1 unchanged across all k values for both Llama-3B and Llama-8B.

**Why it failed:** The LLM already "sees" the neighbor labels — it just doesn't weight them appropriately. Stating the distribution explicitly doesn't change the underlying geometric dominance of the bug prior in the activation space. The model ignores the statistical evidence just like it ignores the individual labels.

### 5b. Enhanced System Prompt (Label Definitions + Bias Warning)

**Idea:** Add explicit label definitions distinguishing intent:
- "bug = defect in existing functionality"
- "question = seeking help or clarification"
- Plus a gentle bias warning: "Note: issues describing problems may be questions if the user is seeking help rather than reporting a defect."

**Result:** Llama-3B: slightly worse. Llama-8B: essentially unchanged. No improvement on question recall.

**Why it failed:** Small models (3B-8B) lack the reasoning capacity to follow nuanced meta-instructions. The model can't apply the rule "distinguish intent from symptoms" because it processes the issue text through the same activation pathways that create the bug bias in the first place. Telling the model about its bias doesn't change its internal representations.

### 5c. Post-Hoc Ensemble (Margin-Based Override Rules)

**Idea:** When VTAG and RAGTAG disagree, use margin thresholds to override RAGTAG predictions.

**Result:**
- Simple rules: best was +0.010 for Llama-8B (question_override), marginal for others
- High-confidence thresholds (margin>1): +0.013 for Llama-8B, closes 26.2% of FT gap
- Model-dependent — rules that help one model hurt another

**Why it's limited:** The disagreement margins are too slim for strong rules. The ensemble can only correct cases where VTAG's signal is strong AND RAGTAG is wrong — a narrow intersection.

### 5d. Llama-3B + Llama-8B Joint Prediction

**Idea:** Check if the two models have complementary errors that could be exploited.

**Result:**
- 80% agreement rate (2,401/3,000)
- When they disagree (599 cases): 3B better on question (+24pp), 8B better on bug (+44pp)
- Smart ensemble (per-pair winner): F1 = 0.7503 (+0.028 over 3B, +0.007 over 8B)
- Oracle ceiling: 0.8178 (if we could always pick the right model)
- Still trails FT by 0.042

**Why it's limited:** The per-pair winner margins are too close to 50/50 for most confusion pairs. The biggest rule ("3B says question, 8B says bug → trust 3B") is only 43% vs 41%. Not enough signal for reliable override.

---

## 6. What We Learned from Failed Interventions

| Intervention | Level | Why it failed | Lesson |
|---|---|---|---|
| Vote prior | Prompt (explicit stats) | LLM ignores statistical evidence | Text-level signals can't override parametric prior |
| Enhanced prompt | Prompt (instructions) | Small models can't follow meta-reasoning | Prompt engineering hits a wall with 3-8B models |
| Post-hoc ensemble | Output (rule-based) | Margins too slim for strong rules | Need intervention before or during generation, not after |
| Model ensemble | Output (model voting) | Complementary but too balanced | Two biased models can't unbias each other reliably |

**The pattern:** All text-level and output-level interventions fail because the bug bias is geometric — it lives in the model's activation space, not in its text interpretation. The model doesn't "read" the labels and choose to ignore them; the "bug" vector in the residual stream overwhelms the "question" vector regardless of what the prompt says.

---

## 7. Gemini Deep Research: Literature Survey (2021-2026)

Commissioned an exhaustive survey of training-free interventions for closing the ICL-FT gap. Key findings organized by viability:

### Tier 1: Logit-Level (we have logit access via Unsloth)

| Method | Mechanism | Overhead | Viability |
|---|---|---|---|
| **Batch Calibration (BC)** | Marginalize batch predictions to estimate template bias, subtract from each prediction | Negligible | High — easiest to implement |
| **Task Calibration (TC)** | Penalize partial-context reliance via mutual information (3 forward passes) | 3x compute | High but expensive |
| **Contrastive Decoding (CD)** | Subtract context-free logits from full-context logits | 2x compute | High — directly targets our problem |

### Tier 2: Activation-Level (Unsloth exposes hidden states)

| Method | Mechanism | Overhead | Viability |
|---|---|---|---|
| **ASA (Activation Steering Adapter)** | Mid-layer steering vectors suppress bug prior | ~0 (20KB asset) | Theoretically best; implementation-heavy |
| **NL-ITI** | MLP probes on attention heads, multi-token intervention | Probe training | Very high for our specific error |
| **RepE** | PCA on contrastive pairs, add/subtract direction in residual stream | Pre-computation | Strong, slightly simpler than ASA |

### Tier 3: Prompt-Level (already tried, limited for small models)

| Method | Mechanism | Viability |
|---|---|---|
| **Contrastive CoT** | Show "why this is NOT a bug" alongside "why this IS a question" | Low — our models can't follow meta-reasoning |
| **Self-Verification** | Two-pass: classify then verify | Low — model defends its own hallucination |

### Key insight from literature

The "80% Rule": prompt engineering reliably achieves 80-90% of a model's peak. The final delta requires weight updates or activation-level intervention. The literature suggests 2-3% gap is the realistic floor for training-free methods — full closure is "theoretically unattainable without altering weights."

---

## 8. Current Plan: Phased Intervention Strategy

### Phase 1: Simple heuristic (design pending)
- Design and test a custom heuristic based on our specific error patterns
- Test on Llama-3B and Llama-8B (fast iteration)
- If it works: adopt. If not: move to Phase 2.

### Phase 2: Logit-level interventions
- **Batch Calibration** — subtract average label distribution bias from each prediction's logits
- **Contrastive Decoding** — subtract zero-shot logits from RAG-augmented logits, suppressing prior-driven predictions
- Test both on Llama-3B and Llama-8B

### Phase 3: Activation steering (if Phases 1-2 insufficient)
- **RepE or NL-ITI** — extract bug-vs-question direction via contrastive pairs, intervene on residual stream during inference
- Most complex but theoretically strongest

### Success criteria
- Close the FT gap by >=50% (from ~0.06 to <=0.03 macro-F1)
- Improve question F1 by >=0.05 without destroying bug recall
- Method must be training-free (no weight updates to the LLM)

---

## 9. Reference: Complete 30k Results Table

| Model | Approach | F1_macro | F1_bug | F1_feat | F1_ques | Bug R | Ques R | Inv% |
|---|---|---|---|---|---|---|---|---|
| Llama-3B | RAGTAG k=3 | 0.7222 | 0.739 | 0.784 | 0.645 | 0.801 | 0.593 | 0.0% |
| Llama-3B | FT | 0.7897 | 0.792 | 0.824 | 0.753 | 0.870 | 0.737 | 0.1% |
| Llama-8B | RAGTAG k=9 | 0.7442 | 0.766 | 0.812 | 0.654 | 0.918 | 0.547 | 2.2% |
| Llama-8B | FT | 0.7925 | 0.775 | 0.835 | 0.768 | 0.734 | 0.822 | 0.0% |
| Qwen-14B | RAGTAG k=9 | 0.7790 | 0.783 | 0.840 | 0.714 | 0.855 | 0.628 | — |
| Qwen-14B | FT | 0.7668 | 0.733 | 0.823 | 0.745 | 0.669 | 0.845 | — |
| Qwen-32B | RAGTAG k=3 | 0.7669 | 0.762 | 0.822 | 0.716 | 0.808 | 0.655 | 4.7% |
| Qwen-32B | FT | pending | — | — | — | — | — | — |

*Note: Llama-8B RAGTAG updated to k=9 from k-study (0.7442 vs original 0.7350 from initial run). Qwen-14B numbers from NRP run.*

---

## 10. Cross-Dataset Per-Label Analysis: Zero-Shot vs RAGTAG vs Fine-Tune

Full comparison of each model's inherent ability (zero-shot), RAGTAG (best k, ctx=8192), and fine-tune across both datasets. This reveals where the bias lives and how each approach handles it.

**Plots:** [zs_ragtag_ft_f1_comparison.png](zs_ragtag_ft_f1_comparison.png), [zs_ragtag_ft_recall.png](zs_ragtag_ft_recall.png), [zs_ragtag_ft_deltas.png](zs_ragtag_ft_deltas.png)

### 10a. Full Results Table

| Model | Data | Approach | F1_macro | F1_bug | F1_feat | F1_ques | R_bug | R_feat | R_ques | P_bug | P_feat | P_ques |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3B | 3k | Zero-Shot | 0.574 | 0.656 | 0.760 | 0.306 | 0.904 | 0.749 | 0.194 | 0.515 | 0.771 | 0.724 |
| Llama-3B | 3k | RAGTAG | 0.674 | 0.688 | 0.773 | 0.562 | 0.778 | 0.801 | 0.468 | 0.617 | 0.747 | 0.703 |
| Llama-3B | 3k | Fine-Tune | 0.667 | 0.697 | 0.735 | 0.570 | 0.858 | 0.636 | 0.512 | 0.586 | 0.869 | 0.642 |
| Llama-3B | 30k | Zero-Shot | 0.693 | 0.730 | 0.782 | 0.566 | 0.923 | 0.744 | 0.444 | 0.604 | 0.825 | 0.779 |
| Llama-3B | 30k | RAGTAG | 0.725 | 0.740 | 0.787 | 0.648 | 0.801 | 0.783 | 0.597 | 0.687 | 0.791 | 0.707 |
| Llama-3B | 30k | Fine-Tune | 0.790 | 0.792 | 0.824 | 0.753 | 0.870 | 0.761 | 0.737 | 0.727 | 0.898 | 0.770 |
| Llama-8B | 3k | Zero-Shot | 0.618 | 0.672 | 0.794 | 0.387 | 0.938 | 0.753 | 0.254 | 0.523 | 0.839 | 0.814 |
| Llama-8B | 3k | RAGTAG | 0.712 | 0.727 | 0.822 | 0.586 | 0.864 | 0.787 | 0.464 | 0.627 | 0.860 | 0.794 |
| Llama-8B | 3k | Fine-Tune | 0.687 | 0.556 | 0.810 | 0.694 | 0.419 | 0.857 | 0.824 | 0.826 | 0.768 | 0.600 |
| Llama-8B | 30k | Zero-Shot | 0.706 | 0.743 | 0.809 | 0.567 | 0.945 | 0.780 | 0.433 | 0.612 | 0.840 | 0.822 |
| Llama-8B | 30k | RAGTAG | 0.744 | 0.766 | 0.812 | 0.654 | 0.918 | 0.788 | 0.547 | 0.657 | 0.839 | 0.813 |
| Llama-8B | 30k | Fine-Tune | 0.793 | 0.775 | 0.835 | 0.768 | 0.734 | 0.820 | 0.822 | 0.821 | 0.850 | 0.720 |
| Qwen-14B | 3k | Zero-Shot | 0.668 | 0.703 | 0.825 | 0.476 | 0.910 | 0.825 | 0.336 | 0.573 | 0.824 | 0.816 |
| Qwen-14B | 3k | RAGTAG | 0.742 | 0.728 | 0.831 | 0.668 | 0.818 | 0.807 | 0.558 | 0.656 | 0.855 | 0.833 |
| Qwen-14B | 3k | Fine-Tune | 0.739 | 0.747 | 0.793 | 0.676 | 0.784 | 0.733 | 0.694 | 0.714 | 0.865 | 0.658 |
| Qwen-32B | 3k | Zero-Shot | 0.702 | 0.715 | 0.835 | 0.557 | 0.882 | 0.843 | 0.422 | 0.602 | 0.827 | 0.818 |
| Qwen-32B | 3k | RAGTAG | 0.778 | 0.780 | 0.837 | 0.716 | 0.892 | 0.817 | 0.630 | 0.693 | 0.857 | 0.829 |
| Qwen-32B | 3k | Fine-Tune | 0.735 | 0.725 | 0.837 | 0.642 | 0.784 | 0.879 | 0.558 | 0.675 | 0.798 | 0.756 |

*Qwen-14B and Qwen-32B 30k zero-shot predictions not saved; Qwen-32B 30k FT still pending on NRP.*

### 10b. Key Findings from Recall Analysis

**1. Zero-shot reveals extreme bug bias across all models:**
- Bug recall: 0.88–0.95 (all datasets, all models)
- Question recall: 0.19–0.44 (3k), 0.43–0.44 (30k)
- The models default to "bug" for any issue describing a problem — this is the parametric prior

**2. RAGTAG partially corrects the bias but doesn't go far enough:**
- Bug recall decreases from zero-shot (0.90→0.80 for Llama-3B on 30k) — fewer false bug predictions
- Question recall increases (0.44→0.60 for Llama-3B on 30k) — retrieval helps the model see "question" examples
- But the correction is incomplete — question recall still trails bug recall by 0.20–0.37

**3. Fine-tune achieves the most balanced recall (on 30k):**
- Bug recall: 0.73–0.87 (lower than zero-shot/RAGTAG — stops over-predicting bug)
- Question recall: 0.74–0.82 (much higher — learns the intent boundary from data)
- The recall gap between bug and question shrinks to 0.09–0.13 (vs 0.20–0.37 for RAGTAG)

**4. Fine-tune on 3k overcorrects (Llama-8B anomaly):**
- Llama-8B 3k FT: bug recall drops to **0.419** — massively swings toward question
- With only ~1,498 training examples, FT can't learn the boundary properly and overcorrects
- On 30k (27k training), it finds balance (0.734 bug recall, 0.822 question recall)
- This means FT with limited data can be WORSE than RAGTAG at calibration

**5. Question precision tells a different story:**
- Zero-shot question precision is high (0.72–0.82) — when the model predicts question, it's usually right
- RAGTAG maintains high question precision (0.70–0.83)
- The problem is exclusively recall — the model doesn't predict question often enough, but when it does, it's accurate
- This suggests the model CAN recognize questions; it just has a high threshold for doing so

### 10c. Gains Over Zero-Shot (Delta Analysis)

| Model | Dataset | RAGTAG Δ Macro | FT Δ Macro | RAGTAG Δ Question | FT Δ Question |
|---|---|---|---|---|---|
| Llama-3B | 3k | +0.100 | +0.093 | +0.256 | +0.264 |
| Llama-3B | 30k | +0.032 | +0.097 | +0.082 | +0.187 |
| Llama-8B | 3k | +0.094 | +0.069 | +0.199 | +0.307 |
| Llama-8B | 30k | +0.038 | +0.086 | +0.087 | +0.201 |
| Qwen-14B | 3k | +0.074 | +0.071 | +0.192 | +0.200 |
| Qwen-32B | 3k | +0.076 | +0.033 | +0.159 | +0.085 |

**Key insight:** On 3k, RAGTAG's macro gains match or beat FT. On 30k, FT's gains are 2–3x larger than RAGTAG's. The divergence is sharpest on question F1: FT gains +0.19 to +0.20 on 30k while RAGTAG gains only +0.08.

### 10d. Implications for Intervention Design

The recall analysis reveals a specific mechanism:
1. The models have **high question precision but low question recall** — they recognize questions when they bother to classify them as such, but their threshold for doing so is too high
2. Any intervention should aim to **lower the model's threshold for predicting question** without destroying its precision
3. This is fundamentally different from teaching the model what a question looks like — it already knows. The issue is that "bug" features in the input overwhelm "question" features in the activation space
4. **Logit-level interventions (Batch Calibration, Contrastive Decoding)** directly address this by mathematically reducing the bug logit's dominance — they lower the threshold without changing the model's representations
5. **The 3k FT overcorrection** on Llama-8B (bug recall 0.419) shows that even gradient-based methods struggle with calibration on small datasets — a logit correction that achieves balanced recall without training would be a genuine contribution

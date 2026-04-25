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

## 8. Phase 1 Results: Debiased Retrieval Heuristic

### 8a. The Idea

Since the LLM already defaults to "bug" from its parametric prior, bug-labeled neighbors in the prompt *reinforce* the bias rather than providing useful signal. The model doesn't need help predicting bug — it needs permission to predict question. Removing bug neighbors from the prompt when question evidence is present forces the model to see only non-bug examples, breaking the reinforcement loop.

### 8b. Mechanism

In `_debias_neighbors()` in `llm_labeler.py`:
1. Count bug and question neighbors in the retrieved set
2. If `bug_count > 0` and `bug_count - question_count <= margin`: remove all bug neighbors
3. Otherwise: keep the original neighbor set unchanged

The margin parameter controls aggressiveness — higher margin triggers debiasing on more issues.

### 8c. Full Results

| Model | Data | Type | F1_macro | F1_bug | F1_feat | F1_ques | R_bug | R_feat | R_ques | P_bug | P_feat | P_ques |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3B | 3k | baseline | 0.6743 | 0.688 | 0.773 | 0.562 | 0.778 | 0.801 | 0.468 | 0.617 | 0.747 | 0.703 |
| Llama-3B | 3k | debias_m3 | 0.6971 | 0.693 | 0.774 | 0.625 | 0.697 | 0.811 | 0.590 | 0.688 | 0.740 | 0.664 |
| Llama-3B | 3k | debias_m2 | 0.6968 | 0.691 | 0.775 | 0.625 | 0.695 | 0.817 | 0.586 | 0.687 | 0.736 | 0.668 |
| Llama-8B | 3k | baseline | 0.7115 | 0.727 | 0.822 | 0.586 | 0.864 | 0.787 | 0.464 | 0.627 | 0.860 | 0.794 |
| Llama-8B | 3k | debias_m3 | 0.7556 | 0.747 | 0.822 | 0.697 | 0.791 | 0.804 | 0.673 | 0.708 | 0.841 | 0.724 |
| Llama-8B | 3k | debias_m2 | 0.7594 | 0.760 | 0.823 | 0.696 | 0.814 | 0.802 | 0.665 | 0.712 | 0.845 | 0.730 |
| Llama-3B | 30k | baseline | 0.7222 | 0.739 | 0.784 | 0.645 | 0.801 | 0.780 | 0.593 | 0.685 | 0.787 | 0.706 |
| Llama-3B | 30k | debias_m3 | 0.7270 | 0.730 | 0.775 | 0.676 | 0.720 | 0.780 | 0.681 | 0.740 | 0.771 | 0.671 |
| Llama-3B | 30k | debias_m2 | 0.7270 | 0.732 | 0.777 | 0.672 | 0.726 | 0.780 | 0.675 | 0.739 | 0.775 | 0.668 |
| Llama-3B | 30k | FT | 0.7897 | 0.792 | 0.824 | 0.753 | 0.870 | 0.761 | 0.737 | 0.727 | 0.898 | 0.770 |
| Llama-8B | 30k | baseline | 0.7429 | 0.766 | 0.811 | 0.652 | 0.921 | 0.786 | 0.543 | 0.656 | 0.837 | 0.815 |
| Llama-8B | 30k | debias_m3 | 0.7569 | 0.764 | 0.812 | 0.695 | 0.826 | 0.795 | 0.654 | 0.710 | 0.829 | 0.742 |
| Llama-8B | 30k | debias_m2 | 0.7546 | 0.766 | 0.809 | 0.689 | 0.840 | 0.792 | 0.638 | 0.704 | 0.828 | 0.748 |
| Llama-8B | 30k | FT | 0.7925 | 0.775 | 0.835 | 0.768 | 0.734 | 0.820 | 0.822 | 0.821 | 0.850 | 0.720 |

### 8d. Summary of Gains

**3k dataset — strong wins:**

| Model | Baseline | Best Debias | Δ F1_macro | Δ Q_recall | Best margin |
|---|---|---|---|---|---|
| Llama-3B | 0.6743 | 0.6971 | **+0.023** | +0.122 | m3 |
| Llama-8B | 0.7115 | 0.7594 | **+0.048** | +0.201 | m2 |

Llama-8B on 3k is the headline: **+0.048 F1_macro**, question recall nearly doubles (0.464→0.665). Bug recall drops modestly (0.864→0.814) — a worthwhile trade. This is the largest single RAGTAG improvement achieved in this study.

**30k dataset — modest gains:**

| Model | Baseline | Best Debias | Δ F1_macro | FT gap before | FT gap after |
|---|---|---|---|---|---|
| Llama-3B | 0.7222 | 0.7270 | +0.005 | 0.068 | 0.063 |
| Llama-8B | 0.7429 | 0.7569 | +0.014 | 0.050 | 0.036 |

Debiasing narrows the FT gap but doesn't close it. Llama-8B gets within 0.036 of FT on 30k.

### 8e. Analysis

**Why it works better on 3k:**
- The 3k training pool is smaller, so retrieved neighbors are noisier — bug neighbors are more likely to be topically similar but semantically misleading. Removing them has outsized impact.
- On 3k, RAGTAG already beats FT. Debiasing widens that lead further (+0.073 for Llama-8B over FT).

**Why it's limited on 30k:**
- The 30k pool provides higher-quality neighbors overall. Bug neighbors are more likely to be genuinely informative, so removing them costs more signal.
- The scaling problem remains: FT trains on 27k examples while RAGTAG still only shows a handful per prompt. No neighbor filtering can close that gap.

**Margin sensitivity is low:**
- m2 vs m3 differences are ≤0.004 across all configs. The heuristic is robust to threshold choice.

**The zero-shot fallback problem (Llama-3B, k=3):**
- With k=3 and high margin, many issues have ALL neighbors removed (all 3 are bugs), falling back to zero-shot. Since zero-shot is heavily bug-biased (0.904 bug recall, 0.194 question recall on 3k), this limits gains.
- A **cap approach** (keep at most 1 bug neighbor instead of removing all) would avoid zero-shot fallback while still reducing bug dominance. Not yet tested.

**Invalid rate improves:**
- Llama-8B 3k: 4.1% → 2.7-3.0%. Llama-8B 30k: 2.2% → 1.7-1.8%.
- Fewer bug-dominated prompts may produce less model confusion.

### 8f. Verdict

Debiased retrieval is **paper-worthy for 3k** (largest RAGTAG improvement, novel approach, validates the bias-reinforcement hypothesis). For 30k, it's **incremental** — confirms that text-level interventions have a ceiling against the scaling problem. Phase 2 (logit-level interventions) remains necessary to close the 30k gap.

### 8g. Ablation: "Always Remove" vs Margin-Gated (3k only)

**Question:** Is the margin condition necessary, or should we *always* remove bug neighbors?

The "always" strategy removes all bug neighbors unconditionally and backfills with the next most-similar non-bug neighbors from a larger retrieval pool (k=16 from FAISS, filtered down to k=3 or k=9 for the prompt).

| Model | Type | F1_macro | F1_bug | F1_feat | F1_ques | R_bug | R_ques |
|---|---|---|---|---|---|---|---|
| Llama-3B | baseline | 0.6743 | 0.688 | 0.773 | 0.562 | 0.778 | 0.468 |
| Llama-3B | remove_m3 | 0.6971 | 0.693 | 0.774 | 0.625 | 0.697 | 0.590 |
| Llama-3B | **replace_always** | **0.6431** | 0.563 | 0.751 | 0.615 | **0.485** | 0.651 |
| Llama-8B | baseline | 0.7115 | 0.727 | 0.822 | 0.586 | 0.864 | 0.464 |
| Llama-8B | remove_m3 | **0.7556** | 0.747 | 0.822 | 0.697 | 0.791 | 0.673 |
| Llama-8B | **replace_always** | 0.7467 | 0.710 | 0.817 | 0.713 | **0.658** | **0.758** |

**Result: "always" is too aggressive.** Bug recall collapses:
- Llama-3B: 0.778 → **0.485** (−0.293). F1_macro drops below baseline.
- Llama-8B: 0.864 → **0.658** (−0.206). F1_macro below remove_m3 despite best-ever question recall (0.758).

**Why:** The model *does* use bug examples to correctly predict bugs. When every prompt has zero bug examples, true bugs get shown 2-3 question/feature neighbors that actively push the model away from "bug." Unlike zero-shot (no examples, parametric prior dominates), replacement examples give the model positive evidence for non-bug labels, overriding the prior.

**Key insight:** The margin condition is not arbitrary — it protects true bugs from being overwhelmed by non-bug replacement examples. The margin ensures debiasing only fires on borderline cases (where bug evidence is weak), not on cases where the retrieval correctly identifies strong bug signal.

**Decision:** "Always" mode abandoned. Margin-gated replacement with backfill (replace_m3) remains the strategy to test on 30k.

---

## 9. Phase 2 Results: Logit-Level Interventions — COMPLETE

### 9a. Batch Calibration (BC)

**Mechanism:** Extract logits for the three label tokens (bug/feature/question) via a single forward pass instead of `model.generate()`. Convert to softmax probabilities, compute the marginal distribution across all test items, then subtract the marginal and add 1/3 (uniform prior). Argmax on calibrated scores. Training-free, zero invalid predictions.

**Implementation:** `logit_calibration.py` — standalone script that imports shared code from `llm_labeler.py`. Uses `--method bc`. Label token IDs for Llama tokenizer: bug=2365, feature=13043, question=7998.

**Results:**

| Model | Dataset | Baseline | BC | Δ F1_macro |
|---|---|---|---|---|
| Llama-3B | 3k | 0.6743 | 0.6834 | **+0.009** |
| Llama-8B | 3k | 0.7115 | 0.7419 | **+0.030** |
| Llama-3B | 30k | 0.7222 | 0.7260 | +0.004 |
| Llama-8B | 30k | 0.7429 | 0.7492 | +0.006 |

**Per-class detail (3k):**

| Model | Method | F1_bug | F1_feat | F1_ques | R_bug | R_ques |
|---|---|---|---|---|---|---|
| Llama-3B | baseline | 0.688 | 0.773 | 0.562 | 0.778 | 0.468 |
| Llama-3B | BC | 0.691 | 0.773 | 0.587 | 0.760 | 0.522 |
| Llama-8B | baseline | 0.727 | 0.822 | 0.586 | 0.864 | 0.464 |
| Llama-8B | BC | 0.764 | 0.824 | 0.638 | 0.916 | 0.538 |

**Analysis:** BC provides consistent but modest improvement. It shifts probability mass from bug toward question (the marginal reveals bug is over-represented in the raw softmax distribution). Effect is larger on 3k than 30k, and larger on 8B than 3B. Does not come close to closing the FT gap on 30k.

### 9b. Contrastive Decoding (CD)

**Mechanism:** Run two forward passes — one with RAG context (full prompt with neighbors), one zero-shot (no neighbors). Subtract zero-shot logits from RAG logits: `cd_logits = rag_logits - alpha * zs_logits`. The idea: whatever the model predicts without context is its parametric prior; subtracting it should isolate the signal from retrieval. Alpha controls subtraction strength.

**Results (3k, alpha sweep):**

| Model | Baseline | CD α=0.5 | CD α=0.75 | CD α=1.0 |
|---|---|---|---|---|
| Llama-3B | 0.6743 | 0.6220 | 0.5354 | **0.4203** |
| Llama-8B | 0.7115 | 0.6469 | 0.4661 | **0.3125** |

**Results (30k, α=1.0 only):**

| Model | Baseline | CD α=1.0 |
|---|---|---|
| Llama-3B | 0.7222 | **0.3881** |
| Llama-8B | 0.7429 | **0.2637** |

**CD is catastrophically destructive at every alpha tested.** Higher alpha = worse. The 8B model at α=1.0 on 30k drops to 0.264 — near random. Bug recall collapses (0.11 for 8B on 30k) while question recall rises (0.50-0.67), confirming the mechanism is "working" directionally but far too aggressively.

**Why CD fails here:** Unlike language modeling (where CD was designed), classification logits for 3 tokens don't have the entropy headroom for contrastive subtraction. The zero-shot logits and RAG logits are highly correlated — both are dominated by the same label space geometry. Subtracting one from the other doesn't isolate retrieval signal; it mostly adds noise and destroys the decision boundary.

### 9c. BC+CD Combined (30k)

| Model | Baseline | BC+CD α=1.0 |
|---|---|---|
| Llama-3B | 0.7222 | **0.3884** |
| Llama-8B | not completed (run killed) | — |

CD's destructive effect dominates the combination. BC cannot rescue logits that CD has already corrupted.

### 9d. Phase 2 Verdict

**BC is mildly useful but insufficient.** Best single gain: +0.030 (Llama-8B, 3k). On 30k, gains are negligible (+0.004 to +0.006).

**CD is harmful and should not be used for classification tasks.** The method was designed for open-ended generation where the logit space has thousands of tokens with rich distributional structure. In a 3-class classification setting, the logit space is too constrained — contrastive subtraction destroys more signal than it isolates.

**Phase 3 (activation steering) was not pursued.** Given that BC barely moves the needle and CD actively hurts, and that the overall research direction has shifted away from gap-closing interventions toward data efficiency analysis, activation steering was deemed not worth the implementation complexity.

### 9e. Intervention Summary (All Phases)

| Intervention | Level | Best Δ (3k) | Best Δ (30k) | Verdict |
|---|---|---|---|---|
| Debiased retrieval (m3) | Prompt/retrieval | +0.048 | +0.014 | Paper-worthy for 3k |
| Batch Calibration | Logit | +0.030 | +0.006 | Marginal |
| Contrastive Decoding | Logit | −0.052 to −0.399 | −0.334 to −0.480 | Destructive |
| BC+CD | Logit | not tested | −0.334 | Destructive |
| Vote prior | Prompt | 0.000 | 0.000 | No effect |
| Enhanced prompt | Prompt | −0.003 | 0.000 | No effect |
| Post-hoc ensemble | Output | +0.013 | +0.013 | Marginal |
| Model ensemble (3B+8B) | Output | — | +0.007 | Marginal |

**Conclusion:** No training-free intervention closes the FT gap on 30k. The gap is structural — FT learns the decision boundary from thousands of gradient updates; RAGTAG is limited to k examples per inference. The research direction has pivoted from gap-closing to data efficiency analysis (see Section 12).

---

## 10. Reference: Complete 30k Results Table

| Model | Approach | F1_macro | F1_bug | F1_feat | F1_ques | Bug R | Ques R | Inv% |
|---|---|---|---|---|---|---|---|---|
| Llama-3B | RAGTAG k=3 | 0.7222 | 0.739 | 0.784 | 0.645 | 0.801 | 0.593 | 0.0% |
| Llama-3B | Debias m3 | 0.7270 | 0.730 | 0.775 | 0.676 | 0.720 | 0.681 | 0.0% |
| Llama-3B | FT | 0.7897 | 0.792 | 0.824 | 0.753 | 0.870 | 0.737 | 0.1% |
| Llama-8B | RAGTAG k=9 | 0.7429 | 0.766 | 0.811 | 0.652 | 0.921 | 0.543 | 2.2% |
| Llama-8B | Debias m3 | 0.7569 | 0.764 | 0.812 | 0.695 | 0.826 | 0.654 | 1.7% |
| Llama-8B | FT | 0.7925 | 0.775 | 0.835 | 0.768 | 0.734 | 0.822 | 0.0% |
| Qwen-14B | RAGTAG k=9 | 0.7790 | 0.783 | 0.840 | 0.714 | 0.855 | 0.628 | — |
| Qwen-14B | FT | 0.7668 | 0.733 | 0.823 | 0.745 | 0.669 | 0.845 | — |
| Qwen-32B | RAGTAG k=3 | 0.7669 | 0.762 | 0.822 | 0.716 | 0.808 | 0.655 | 4.7% |
| Qwen-32B | FT | 0.8103 | 0.812 | 0.849 | 0.770 | — | — | — |

*Note: Llama-8B RAGTAG baseline updated from k-study (0.7429). Qwen-14B numbers from NRP run. Qwen-32B FT is best overall at 0.8103.*

---

## 11. Cross-Dataset Per-Label Analysis: Zero-Shot vs RAGTAG vs Fine-Tune

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

---

## 12. Fine-Tune Generalization Test: 30k FT → 3k Test Set

**Question:** Does the 30k fine-tuned Llama-3B generalize to a different test distribution (the 3k test set)?

**Setup:** Loaded saved LoRA adapters from `results/issues30k/unsloth_Llama_3_2_3B_Instruct/finetune_fixed/adapters_unsloth_Llama-3.2-3B-Instruct/` with `--skip_training`. Ran inference on the 3k test split (1,497 issues). No retraining.

**Results:**

| Config | Test Set | F1_macro | F1_bug | F1_feat | F1_ques | R_bug | R_ques |
|---|---|---|---|---|---|---|---|
| 30k FT on 30k (home turf) | 30k | 0.790 | 0.792 | 0.824 | 0.753 | 0.870 | 0.737 |
| 30k FT on 3k (generalization) | 3k | 0.680 | 0.700 | 0.770 | 0.570 | 0.880 | 0.470 |
| RAGTAG on 3k (no training) | 3k | 0.674 | 0.688 | 0.773 | 0.562 | 0.778 | 0.468 |
| RAGTAG + BC on 3k | 3k | 0.683 | 0.691 | 0.773 | 0.587 | 0.760 | 0.522 |

**Key findings:**

1. **FT degrades -0.110 F1_macro on out-of-distribution data.** The model learned the 30k distribution but doesn't generalize cleanly. The biggest casualty is question recall: 0.737 → 0.470.

2. **The parametric bug bias resurfaces.** On its home turf (30k), FT achieves balanced recall (R_bug=0.87, R_ques=0.74, gap=0.13). On the 3k test set, the recall imbalance returns (R_bug=0.88, R_ques=0.47, gap=0.41) — nearly identical to the zero-shot pattern. Fine-tuning suppresses the bias on its own distribution, but the bias is not "cured" — it resurfaces on unfamiliar data.

3. **30k FT on 3k barely beats RAGTAG on 3k.** The gap is just +0.006 (0.680 vs 0.674). With BC calibration, RAGTAG actually surpasses FT (0.683 vs 0.680). A model fine-tuned on 10x more data, tested on a different dataset, ties or loses to training-free RAGTAG on that dataset.

4. **Implication for practitioners:** FT's advantage is partly in-distribution memorization. If your deployment data doesn't perfectly match your training data (common in real-world software projects where issue patterns evolve), FT's edge disappears. RAGTAG, being retrieval-based, naturally adapts to whatever neighbors are available — it doesn't overfit to a training distribution.

---

## 13. Research Direction Pivot: Data Efficiency Analysis

**Date:** 2026-04-20

### Why we're pivoting

All training-free interventions (Sections 5, 8, 9) have failed to close the FT gap on 30k. The gap is structural: FT learns from thousands of gradient updates; RAGTAG is limited to k examples per inference. No logit correction, prompt engineering, or retrieval heuristic can bridge that fundamental asymmetry at scale.

However, the project already has a strong narrative without closing the gap:
- RAGTAG wins on 3k across all 4 models (no training needed)
- RAGTAG wins on 30k for Qwen-14B (larger models may never need FT)
- FT doesn't generalize well (Section 12: -0.11 on out-of-distribution data)
- FT is unstable on small data (Llama-8B 3k: bug recall collapses to 0.419)

### The new question

Instead of "can we close the gap?", the paper should ask: **"At what data scale does fine-tuning overtake RAGTAG?"**

### Planned experiment: Data Efficiency Curve

- Take the 30k training pool (~27k examples)
- Subsample at log-spaced intervals: 1k, 3k, 9k, 27k (already have)
- For each size: build FAISS index (RAGTAG) / fine-tune from scratch (FT)
- Same held-out 3,000 test issues throughout
- Run on Llama-3B (primary) + Llama-8B at 2-3 validation points
- FT with multiple random seeds (2-3) to capture variance — RAGTAG is deterministic

**Why not Qwen models:** Qwen-14B shows no crossover (RAGTAG wins at both endpoints). Qwen-32B requires A100 hardware, outside the consumer GPU target scenario. Endpoint behavior for both Qwen models is already established in Section 10/11.

### Expected output

A crossover plot showing:
- RAGTAG F1 curve (expected: gentle upward slope, limited by k-example ceiling)
- FT F1 curve with error bars (expected: steep rise, eventually overtaking RAGTAG)
- The crossover point = the practical recommendation ("below X labeled examples, use RAGTAG")

### Supporting arguments already in hand

- FT instability at small pool sizes (Llama-8B 3k overcorrection, Section 10b finding 4)
- FT generalization failure (Section 12)
- RAGTAG's compute advantage (no training, shared FAISS index across models)
- RAGTAG's model portability (same index serves 4 different LLMs)
- RAGTAG's instant adaptability (add new data to index, no retraining)

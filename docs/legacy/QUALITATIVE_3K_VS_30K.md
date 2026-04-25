# Per-Class Analysis: 3k vs 30k — Why the Gap Emerges at Scale

**Date:** 2026-04-17
**Goal:** Determine whether RAGTAG's question misclassification problem exists on the 3k dataset or only emerges at 30k scale.

---

## 1. Per-Class Comparison on 3k (ctx=8192, best k per model)

| Model | Approach | F1_macro | F1_bug | F1_feature | F1_question |
|---|---|---|---|---|---|
| Llama-3B | RAGTAG k=3 | 0.6743 | 0.688 | 0.773 | 0.562 |
| Llama-3B | Fine-tune | 0.6669 | 0.697 | 0.735 | 0.570 |
| Llama-3B | **Δ (R−FT)** | **+0.007** | -0.009 | +0.039 | -0.008 |
| | | | | | |
| Llama-8B | RAGTAG k=9 | 0.7115 | 0.727 | 0.822 | 0.586 |
| Llama-8B | Fine-tune | 0.6868 | 0.556 | 0.810 | 0.694 |
| Llama-8B | **Δ (R−FT)** | **+0.025** | +0.171 | +0.012 | -0.108 |
| | | | | | |
| Qwen-14B | RAGTAG k=9 | 0.7423 | 0.728 | 0.831 | 0.668 |
| Qwen-14B | Fine-tune | 0.7387 | 0.747 | 0.793 | 0.676 |
| Qwen-14B | **Δ (R−FT)** | **+0.004** | -0.019 | +0.037 | -0.007 |
| | | | | | |
| Qwen-32B | RAGTAG k=9 | 0.7775 | 0.780 | 0.837 | 0.716 |
| Qwen-32B | Fine-tune | 0.7347 | 0.725 | 0.837 | 0.642 |
| Qwen-32B | **Δ (R−FT)** | **+0.043** | +0.055 | -0.000 | +0.074 |

## 2. Per-Class Comparison on 30k (ctx=8192, best k per model)

| Model | Approach | F1_macro | F1_bug | F1_feature | F1_question |
|---|---|---|---|---|---|
| Llama-3B | RAGTAG k=3 | 0.7222 | 0.739 | 0.784 | 0.645 |
| Llama-3B | Fine-tune | 0.7895 | 0.792 | 0.824 | 0.753 |
| Llama-3B | **Δ (R−FT)** | **-0.067** | -0.053 | -0.041 | -0.108 |
| | | | | | |
| Llama-8B | RAGTAG k=9 | 0.7350 | 0.753 | 0.807 | 0.645 |
| Llama-8B | Fine-tune | 0.7923 | 0.775 | 0.834 | 0.768 |
| Llama-8B | **Δ (R−FT)** | **-0.057** | -0.022 | -0.028 | -0.122 |
| | | | | | |
| Qwen-14B | RAGTAG k=9 | 0.7693 | 0.770 | 0.834 | 0.704 |
| Qwen-14B | Fine-tune | pending | — | — | — |
| | | | | | |
| Qwen-32B | RAGTAG k=3 | 0.7669 | 0.762 | 0.822 | 0.716 |
| Qwen-32B | Fine-tune | pending | — | — | — |

## 3. Key Finding: The Gap Is a Scaling Problem

On the 3k dataset:
- RAGTAG **wins overall** for 3 of 4 models (all except marginally on Llama-3B)
- Question F1 is **not a clear bottleneck** — deltas are small and mixed (−0.008 to +0.074)
- Qwen-32B RAGTAG actually **beats FT on question** by +0.074
- The only model where FT clearly wins on question is Llama-8B (−0.108), but RAGTAG compensates with a massive bug F1 advantage (+0.171)

On the 30k dataset:
- Fine-tuning **wins overall** for both models with complete results
- **Question becomes the dominant bottleneck**: Δ = −0.108 (Llama-3B), −0.122 (Llama-8B)
- The question gap accounts for ~50% of the total macro-F1 gap
- Bug and feature gaps are smaller (−0.02 to −0.05)

## 4. Why This Happens

**Fine-tuning scales with training data.** On 3k, FT trains on ~1,498 examples — not enough to learn the subtle question-vs-bug intent boundary. On 30k, FT trains on ~27,000 examples and can learn nuanced patterns through gradient updates across thousands of boundary cases.

**RAGTAG's prompt budget is fixed.** Whether the training pool is 1.5k or 27k, RAGTAG still only shows k=3-9 examples in the prompt. The retrieval pool is better (higher-quality neighbors from a larger corpus), but the model still only sees a handful of examples per inference.

**The question-vs-bug boundary is the hardest one.** Both describe problematic behavior — the distinction is user intent (seeking help vs reporting defect). With few training examples (3k), even FT can't learn this well, so RAGTAG's general reasoning ability is competitive. With many examples (30k), FT learns the intent boundary through sheer volume of gradient updates.

## 5. Implications for Enhancement

The enhancement strategy must address the **fixed prompt budget** problem:
- RAGTAG can't scale by simply adding more examples (context window and truncation limits)
- Must find ways to inject more information from the larger training pool into the prompt *without* adding more examples
- Candidate approaches: retrieval vote prior, label definitions, intent-focused instructions, or ensemble methods that leverage the full retrieval set
- The 3k results prove RAGTAG's approach is sound — the problem is purely about scaling the signal, not about a fundamental flaw

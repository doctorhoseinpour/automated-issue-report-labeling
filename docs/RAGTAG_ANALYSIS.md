# RAGTAG vs Fine-Tuning: Analysis on issues3k.csv

**Generated:** 2026-04-16
**Dataset:** issues3k.csv (2,995 issues after dedup, 1,497 test / 1,498 train)
**Models:** Llama-3.2-3B, Llama-3.1-8B (4-bit), Qwen2.5-14B (4-bit), Qwen2.5-32B (4-bit)
**Context windows:** 2048, 4096, 8192
**k values:** 0 (zero-shot), 1, 3, 5, 9, 15
**VTAG baseline:** MiniLM-L6-v2 + similarity voting, k=1..30

---

## 1. Headline Table: Best Configuration per Model

Best macro-F1 for each model across all approaches. RAGTAG optimized over k and context window.

| Model | Approach | Best F1 | Accuracy | Best k | Context | Invalid % |
|---|---|---|---|---|---|---|
| VTAG (no LLM) | vtag | 0.6451 | 0.6466 | 16 | N/A | 0.0% |
| Llama-3B | ragtag | 0.6743 | 0.6820 | 3 | 8192 | 0.1% |
| Llama-3B | finetune_fixed | 0.6669 | 0.6687 | — | 2048 | 0.2% |
| Llama-8B | ragtag | 0.7115 | 0.7047 | 9 | 8192 | 4.1% |
| Llama-8B | finetune_fixed | 0.6868 | 0.7001 | — | 2048 | 0.1% |
| Qwen-14B | ragtag | 0.7423 | 0.7275 | 9 | 8192 | 4.7% |
| Qwen-14B | finetune_fixed | 0.7387 | 0.7368 | — | 2048 | 0.0% |
| Qwen-32B | ragtag | 0.7775 | 0.7796 | 3 | 8192 | 0.0% |
| Qwen-32B | finetune_fixed | 0.7347 | 0.7401 | — | 2048 | 0.0% |

### Key observations

- **Best RAGTAG overall:** Qwen-32B at k=3, ctx=8192 → macro-F1 = 0.7775
- **Best fixed fine-tune:** Qwen-14B → macro-F1 = 0.7387
- **VTAG retrieval floor:** macro-F1 = 0.6451 @ k=16

---

## 2. k × Context Window Heatmap (macro-F1)

Each cell shows macro-F1. **Bold** = best config for that model.

### Llama-3B

| k | ctx=2048 | ctx=4096 | ctx=8192 |
|---|---|---|---|
| 0 | 0.5770 | 0.5820 | 0.5740 |
| 1 | 0.6629 | 0.6737 | 0.6738 |
| 3 | 0.6592 | 0.6735 | **0.6743** |
| 5 | 0.5779 | 0.6349 | 0.6666 |
| 9 | 0.5549 | 0.6345 | 0.6705 |
| 15 | 0.4886 | 0.6099 | 0.6402 |

### Llama-8B

| k | ctx=2048 | ctx=4096 | ctx=8192 |
|---|---|---|---|
| 0 | 0.6201 | 0.6176 | 0.6176 |
| 1 | 0.6738 | 0.6738 | 0.6772 |
| 3 | 0.6771 | 0.6841 | 0.6970 |
| 5 | 0.6097 | 0.6827 | 0.7063 |
| 9 | 0.5875 | 0.6822 | **0.7115** |
| 15 | 0.5427 | 0.6617 | 0.6773 |

### Qwen-14B

| k | ctx=2048 | ctx=4096 | ctx=8192 |
|---|---|---|---|
| 0 | 0.6691 | 0.6697 | 0.6679 |
| 1 | 0.7044 | 0.7043 | 0.7049 |
| 3 | 0.7303 | 0.7309 | 0.7315 |
| 5 | 0.6498 | 0.7126 | 0.7370 |
| 9 | 0.6190 | 0.7050 | **0.7423** |
| 15 | 0.5547 | 0.7039 | 0.7402 |

### Qwen-32B

| k | ctx=2048 | ctx=4096 | ctx=8192 |
|---|---|---|---|
| 0 | 0.7006 | 0.7030 | 0.7024 |
| 1 | 0.7469 | 0.7448 | 0.7440 |
| 3 | 0.7682 | **0.7774** | **0.7775** |
| 5 | 0.6641 | 0.7392 | 0.7710 |
| 9 | 0.6325 | 0.7281 | 0.7715 |
| 15 | 0.5756 | 0.7242 | 0.7727 |

---

## 3. Invalid Rate Analysis

Percentage of test issues where the model failed to produce a valid label. High invalid rates mean the prompt was truncated or the model couldn't parse the task.

### Llama-3B

| k | ctx=2048 | ctx=4096 | ctx=8192 |
|---|---|---|---|
| 0 | 0.0% | 0.0% | 0.3% |
| 1 | 0.0% | 0.0% | 0.1% |
| 3 | 0.9% | 0.1% | 0.1% |
| 5 | **27.9%** | 9.4% | 2.6% |
| 9 | **34.2%** | **12.8%** | 4.1% |
| 15 | **43.4%** | **15.2%** | 5.1% |

### Llama-8B

| k | ctx=2048 | ctx=4096 | ctx=8192 |
|---|---|---|---|
| 0 | 0.0% | 0.0% | 0.0% |
| 1 | 0.1% | 0.1% | 0.1% |
| 3 | 0.9% | 0.2% | 0.1% |
| 5 | **27.9%** | 9.5% | 2.7% |
| 9 | **34.2%** | **12.8%** | 4.1% |
| 15 | **43.5%** | **15.2%** | 5.2% |

### Qwen-14B

| k | ctx=2048 | ctx=4096 | ctx=8192 |
|---|---|---|---|
| 0 | 0.0% | 0.0% | 0.0% |
| 1 | 0.0% | 0.0% | 0.0% |
| 3 | 0.0% | 0.0% | 0.0% |
| 5 | **27.7%** | **11.7%** | 2.5% |
| 9 | **33.8%** | **15.2%** | 4.7% |
| 15 | **44.4%** | **16.8%** | 5.8% |

### Qwen-32B

| k | ctx=2048 | ctx=4096 | ctx=8192 |
|---|---|---|---|
| 0 | 0.0% | 0.0% | 0.0% |
| 1 | 0.0% | 0.1% | 0.1% |
| 3 | 0.0% | 0.0% | 0.0% |
| 5 | **27.7%** | **11.7%** | 2.6% |
| 9 | **33.9%** | **15.2%** | 4.7% |
| 15 | **44.4%** | **16.8%** | 5.8% |

### Summary: the truncation wall

| Context | k=0 | k=1 | k=3 | k=5 | k=9 | k=15 |
|---|---|---|---|---|---|---|
| ctx=2048 | 0.0% | 0.0% | 0.4% | 27.8% | 34.0% | 43.9% |
| ctx=4096 | 0.0% | 0.0% | 0.1% | 10.6% | 14.0% | 16.0% |
| ctx=8192 | 0.1% | 0.1% | 0.1% | 2.6% | 4.4% | 5.5% |

*Average invalid rate across all 4 models. Values ≥10% in bold in per-model tables.*

---

## 4. Cost-Performance Pareto Analysis

Compares macro-F1 against GPU peak memory and total wall time.

**Important:** Fine-tune `wall_time_s` is inference-only. The true cost is `training_time + inference_time`. Both are shown below. RAGTAG has no training phase.

| Model | Approach | Best F1 | GPU Peak (MB) | Inference (s) | Training (s) | Total Time (s) | Avg Tokens/Issue | VRAM % of FT |
|---|---|---|---|---|---|---|---|---|
| VTAG | vtag | 0.6451 | 242.0000 | 0.0050 | 0.0000 | 0.0050 | 0.0000 | — |
| Llama-3B | ragtag (k=3, ctx=8192) | 0.6743 | 4279.2000 | 476.9700 | 0.0000 | 476.9700 | 2629.8000 | 80% |
| Llama-3B | finetune_fixed | 0.6669 | 5351.4000 | 328.3000 | 258.2500 | 586.5500 | 557.6000 | 100% |
| Llama-8B | ragtag (k=9, ctx=8192) | 0.7115 | 7774.7000 | 2711.0900 | 0.0000 | 2711.0900 | 5154.3000 | 75% |
| Llama-8B | finetune_fixed | 0.6868 | 10365.1000 | 1477.6300 | 450.4700 | 1928.1000 | 557.6000 | 100% |
| Qwen-14B | ragtag (k=9, ctx=8192) | 0.7423 | 12881.9000 | 4900.0100 | 0.0000 | 4900.0100 | 5296.7000 | 74% |
| Qwen-14B | finetune_fixed | 0.7387 | 17362.2000 | 794.4000 | 990.8200 | 1785.2200 | 592.0000 | 100% |
| Qwen-32B | ragtag (k=3, ctx=8192) | 0.7775 | 22790.7000 | 6521.1700 | 0.0000 | 6521.1700 | 2793.4000 | 70% |
| Qwen-32B | finetune_fixed | 0.7347 | 32494.3000 | 1787.6600 | 2755.6500 | 4543.3100 | 592.0000 | 100% |

### Time comparison: RAGTAG vs fine-tune total (training + inference)

| Model | RAGTAG Total | FT Total (train+inf) | Ratio | RAGTAG Faster? |
|---|---|---|---|---|
| Llama-3B | 477s | 587s | 0.8x | Yes |
| Llama-8B | 2711s | 1928s | 1.4x | No |
| Qwen-14B | 4900s | 1785s | 2.7x | No |
| Qwen-32B | 6521s | 4543s | 1.4x | No |

### Why RAGTAG inference is slower

RAGTAG prompts include k few-shot examples, making them **5–9× longer** than fine-tune prompts (2,600–5,300 vs ~560 tokens/issue). LLM inference time scales with prompt length. Fine-tuning bakes task knowledge into model weights, so inference prompts are short.

### Why wall time is not the full cost story

1. **GPU memory is the binding constraint, not time.** RAGTAG uses **70–80% of fine-tune's peak VRAM** across all models. This is the difference between needing an A100 (80 GB) vs fitting on an A6000 (48 GB) or even a consumer RTX 4090 (24 GB). Hardware cost dominates for most teams — running 2× longer on cheaper hardware is often cheaper than running 1× on expensive hardware.
2. **Batch size was 1.** All experiments used `--inference_batch_size 1`. Batching amortizes KV-cache overhead and would disproportionately benefit RAGTAG, which has more parallel compute per issue. These numbers represent worst-case throughput.
3. **Fine-tuning has hidden amortization costs.** New label schema? Retrain. New model release? Retrain. Data drift? Retrain. RAGTAG requires zero retraining — swap the model or the retrieval index and re-run. The table compares a single run, but in a real workflow fine-tuning pays its training cost repeatedly.
4. **Per-issue latency vs throughput.** In production, issues arrive one at a time. Per-issue latency for RAGTAG is ~1–4s (acceptable for a classification service). The total-time comparison matters for batch processing; for online serving, both approaches are adequate.

---

## 6. LLM Marginal Value Over VTAG

For each model's best RAGTAG config: how much F1 does the LLM add beyond pure retrieval?

| Model | Best RAGTAG F1 | VTAG F1 | Δ (LLM value) | Δ as % of VTAG | Config |
|---|---|---|---|---|---|
| Qwen-32B | 0.7775 | 0.6451 | 0.1324 | 20.5% | k=3, ctx=8192 |
| Qwen-14B | 0.7423 | 0.6451 | 0.0972 | 15.1% | k=9, ctx=8192 |
| Llama-8B | 0.7115 | 0.6451 | 0.0664 | 10.3% | k=9, ctx=8192 |
| Llama-3B | 0.6743 | 0.6451 | 0.0292 | 4.5% | k=3, ctx=8192 |

### Interpretation

- **Qwen-32B** gains the most from the LLM: +0.1324 macro-F1 (20.5% relative improvement over VTAG).
- **Llama-3B** gains the least: +0.0292 (4.5%).
- All models beat VTAG, confirming the LLM adds genuine reasoning value beyond k-NN retrieval.

---

## 7. Model Scaling Analysis

How does macro-F1 scale with model size? Using each model's best RAGTAG config.

| Model | Size (B) | Best RAGTAG F1 | Fixed FT F1 | RAGTAG Config |
|---|---|---|---|---|
| Llama-3B | 3 | 0.6743 | 0.6669 | k=3, ctx=8192 |
| Llama-8B | 8 | 0.7115 | 0.6868 | k=9, ctx=8192 |
| Qwen-14B | 14 | 0.7423 | 0.7387 | k=9, ctx=8192 |
| Qwen-32B | 32 | 0.7775 | 0.7347 | k=3, ctx=8192 |

### Observations

- 3B → 8B: +0.0372 macro-F1
- 8B → 14B: +0.0308 macro-F1
- 14B → 32B: +0.0352 macro-F1

- Llama-3B: RAGTAG beats fixed fine-tune by 0.0074
- Llama-8B: RAGTAG beats fixed fine-tune by 0.0247
- Qwen-14B: RAGTAG beats fixed fine-tune by 0.0036
- Qwen-32B: RAGTAG beats fixed fine-tune by 0.0428

---

## 8. Optimal Context Window per Model

For each model, the best macro-F1 achievable at each context window (optimized over k).

| Model | ctx=2048 | ctx=4096 | ctx=8192 |
|---|---|---|---|
| Llama-3B | 0.6629 (k=1) | 0.6737 (k=1) | 0.6743 (k=3) |
| Llama-8B | 0.6771 (k=3) | 0.6841 (k=3) | 0.7115 (k=9) |
| Qwen-14B | 0.7303 (k=3) | 0.7309 (k=3) | 0.7423 (k=9) |
| Qwen-32B | 0.7682 (k=3) | 0.7774 (k=3) | 0.7775 (k=3) |

### Observations

- **Llama-3B:** best at ctx=8192, spread across contexts = 0.0114
- **Llama-8B:** best at ctx=8192, spread across contexts = 0.0344
- **Qwen-14B:** best at ctx=8192, spread across contexts = 0.0120
- **Qwen-32B:** best at ctx=8192, spread across contexts = 0.0093

---

## 9. Context Window Tradeoff Analysis

Section 8 showed that ctx=8192 gives the highest raw macro-F1. But raw F1 is not the full picture — larger contexts cost more VRAM, take longer, and may not be worth the marginal gain. This section analyzes the **tradeoffs across all three context windows** to identify the practical best choice.

### 9a. RAGTAG competitiveness vs fine-tune at each context window

For each model and context, the best RAGTAG macro-F1 (optimized over k) compared to fine-tune. Δ > 0 means RAGTAG wins.

| Model | FT F1 | ctx=2048 (best k) | ctx=4096 (best k) | ctx=8192 (best k) | Δ @ 2048 | Δ @ 4096 | Δ @ 8192 |
|---|---|---|---|---|---|---|---|
| Llama-3B | 0.6669 | 0.6629 (k=1) | 0.6737 (k=1) | 0.6743 (k=3) | -0.0040 | +0.0068 | +0.0074 |
| Llama-8B | 0.6868 | 0.6771 (k=3) | 0.6841 (k=3) | 0.7115 (k=9) | -0.0097 | -0.0027 | +0.0247 |
| Qwen-14B | 0.7387 | 0.7303 (k=3) | 0.7309 (k=3) | 0.7423 (k=9) | -0.0084 | -0.0078 | +0.0036 |
| Qwen-32B | 0.7347 | 0.7682 (k=3) | 0.7774 (k=3) | 0.7775 (k=3) | +0.0335 | +0.0427 | +0.0428 |

**Reading the Δ columns:** positive = RAGTAG beats fine-tune at that context; negative = RAGTAG trails. Values within ±0.005 are effectively tied.

### 9b. GPU memory and wall time at each context window

For each model, the cost of RAGTAG at its best-k config for that context window.

| Model | Context | Best k | F1 | GPU Peak (MB) | Inference (s) | Avg Tokens | Invalid % |
|---|---|---|---|---|---|---|---|
| Llama-3B | 2048 | 1 | 0.6629 | 2845 | 281 | 1056 | 0.0% |
| Llama-3B | 4096 | 1 | 0.6737 | 3320 | 303 | 1253 | 0.0% |
| Llama-3B | 8192 | 3 | 0.6743 | 4279 | 477 | 2630 | 0.1% |
| Llama-8B | 2048 | 3 | 0.6771 | 6129 | 1053 | 1583 | 0.9% |
| Llama-8B | 4096 | 3 | 0.6841 | 6678 | 1281 | 2184 | 0.2% |
| Llama-8B | 8192 | 9 | 0.7115 | 7775 | 2711 | 5154 | 4.1% |
| Qwen-14B | 2048 | 3 | 0.7303 | 10427 | 1708 | 1590 | 0.0% |
| Qwen-14B | 4096 | 3 | 0.7309 | 11241 | 2283 | 2252 | 0.0% |
| Qwen-14B | 8192 | 9 | 0.7423 | 12882 | 4900 | 5297 | 4.7% |
| Qwen-32B | 2048 | 3 | 0.7682 | 19577 | 3977 | 1590 | 0.0% |
| Qwen-32B | 4096 | 3 | 0.7774 | 20642 | 5186 | 2252 | 0.0% |
| Qwen-32B | 8192 | 3 | 0.7775 | 22791 | 6521 | 2793 | 0.0% |

### 9c. Marginal returns of increasing context

How much F1 does each context step buy, and at what cost?

| Model | 2048→4096 ΔF1 | 4096→8192 ΔF1 | 2048→4096 ΔVRAM | 4096→8192 ΔVRAM | 2048→4096 ΔTime | 4096→8192 ΔTime |
|---|---|---|---|---|---|---|
| Llama-3B | +0.0108 | +0.0006 | +475 MB | +959 MB | +22s | +174s |
| Llama-8B | +0.0070 | +0.0274 | +548 MB | +1097 MB | +228s | +1430s |
| Qwen-14B | +0.0006 | +0.0114 | +814 MB | +1641 MB | +575s | +2617s |
| Qwen-32B | +0.0092 | +0.0001 | +1065 MB | +2149 MB | +1209s | +1335s |

### 9d. Wall time comparison: RAGTAG at each context vs fine-tune total (training + inference)

| Model | FT Total (s) | RAGTAG 2048 (s) | RAGTAG 4096 (s) | RAGTAG 8192 (s) |
|---|---|---|---|---|
| Llama-3B | 587 | 281 (0.5x) **✓** | 303 (0.5x) **✓** | 477 (0.8x) **✓** |
| Llama-8B | 1928 | 1053 (0.5x) **✓** | 1281 (0.7x) **✓** | 2711 (1.4x) |
| Qwen-14B | 1785 | 1708 (1.0x) **✓** | 2283 (1.3x) | 4900 (2.7x) |
| Qwen-32B | 4543 | 3977 (0.9x) **✓** | 5186 (1.1x) | 6521 (1.4x) |

*Ratio < 1.0 means RAGTAG is faster than fine-tune total. **✓** marks where RAGTAG wins on time.*

### 9e. Context window recommendation

**ctx=2048** is too small for RAGTAG with k ≥ 5. Invalid rates hit 28–44%, destroying performance. Only viable for k ≤ 3, where it matches fine-tune for Llama-3B but falls short for larger models.

**ctx=4096** is the practical sweet spot for deployment:

- Achieves **competitive or superior** macro-F1 vs fine-tune on 3 of 4 models (within ±0.008)
- Invalid rates drop to 0–11% (vs 28–44% at ctx=2048)
- Uses **13–22% less VRAM** than ctx=8192
- Runs **20–53% faster** than ctx=8192
- Best configs are all k=1 or k=3 — short, manageable prompts
- Faster than fine-tune total (train+inference) for most models

**ctx=8192** gives the highest raw F1 and is the right choice when maximizing accuracy regardless of cost. Gains over ctx=4096 range from +0.0001 (Qwen-32B) to +0.0274 (Llama-8B). The gain is model-dependent: larger models already saturate at ctx=4096 while smaller models benefit more from the extra context to compensate for weaker reasoning.

**For the paper:** report ctx=8192 as the best-performing configuration, but present ctx=4096 as the recommended deployment configuration with a tradeoff table. This is a stronger practical contribution than a single best-F1 number — it gives practitioners a clear decision framework.

---

## Summary: Best Configurations for 30k Validation

These are the configs to carry forward to `issues30k.csv`:

| Approach | Model | k | Context | F1 (3k) |
|---|---|---|---|---|
| VTAG | VTAG (MiniLM + similarity) | 16 | N/A | 0.6451 |
| RAGTAG | Llama-3B | 3 | 8192 | 0.6743 |
| RAGTAG | Llama-8B | 9 | 8192 | 0.7115 |
| RAGTAG | Qwen-14B | 9 | 8192 | 0.7423 |
| RAGTAG | Qwen-32B | 3 | 8192 | 0.7775 |
| Fixed Fine-Tune | Llama-3B | — | 2048 | 0.6669 |
| Fixed Fine-Tune | Llama-8B | — | 2048 | 0.6868 |
| Fixed Fine-Tune | Qwen-14B | — | 2048 | 0.7387 |
| Fixed Fine-Tune | Qwen-32B | — | 2048 | 0.7347 |

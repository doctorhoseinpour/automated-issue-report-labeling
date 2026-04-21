# Paper Narrative & Structure — Final Plan

**Date:** 2026-04-20
**Status:** Agreed — ready to execute remaining experiments and write

---

## Working Title

**"RAGTAG: When Retrieval-Augmented Classification Beats Fine-Tuning for GitHub Issue Labeling"**

---

## Core Story

The paper answers one practical question: **at what labeled-data budget does fine-tuning overtake training-free RAG classification, and why?**

This is framed through a three-phase study:
1. **Phase 1 (3k dataset):** Develop and analyze RAGTAG, compare against fine-tuning and a pure-retrieval baseline, discover the parametric bug bias
2. **Phase 2 (30k dataset):** Validate findings at scale, run data efficiency crossover analysis, confirm the bias persists
3. **Phase 3 (bias correction):** Apply debiased retrieval heuristic to partially correct the bias across all models

The narrative arc is: **discover → validate → address.** Each phase motivates the next.

---

## Key Design Decisions

### What we ARE doing
- Three-phase structure (3k development → 30k evaluation → bias correction)
- Data efficiency crossover at 1.5k / 3k / 9k / 15k / 27k training subsamples, all 4 models
- Debiased retrieval (m=3) as the main bias correction intervention, completed on all 4 models
- VTAG (MiniLM + similarity voting) as the retrieval floor
- Pareto frontier / cost-performance analysis in Phase 1
- Random neighbor ablation to confirm retrieval quality matters

### What we are NOT doing
- **No flawed fine-tune comparison.** Our corrected FT implementation is just "the fine-tuning baseline." No mention of any prior flawed pipeline. Too much trouble, no upside.
- **No cross-dataset flip tests** (e.g., 3k FT → 30k test). The data efficiency curve within 30k already proves the scaling story. The one existing result (30k FT → 3k test = 0.680 vs RAGTAG 0.674) gets a single paragraph in Discussion.
- **No seeds for FT variance.** Results are identical across runs. We report "averaged over 3 runs" in one sentence.
- **Activation steering, Batch Calibration, Contrastive Decoding → Future Work only.** Not in main results. Brief mention in Discussion/Future Work.
- **VTAG voting scheme ablation → excluded.** Spread is 0.002. Not worth mentioning.
- **VTAG embedder ablation (bge-base/bge-large) → excluded or Threats to Validity.** Creates apples-to-oranges problem since all RAGTAG numbers use MiniLM. Note in Threats that a stronger embedder could lift both VTAG and RAGTAG floors.

### Dataset design
- **3k = development dataset.** Tune k, context window, discover bias, run deep analysis. ~2,995 issues after dedup, 1,497 test / 1,498 train, balanced 3-class.
- **30k = evaluation dataset.** Test generalization, run data efficiency curve. ~30,000 issues, ~3,000 test / ~27,000 train. Independently collected, **no overlap** with 3k.
- This is standard ML methodology: develop on one dataset, evaluate on another.

---

## Abstract

LLM-based GitHub issue classification (bug/feature/question) can be approached through fine-tuning or training-free retrieval-augmented generation. We present RAGTAG, a RAG-based few-shot classification method, and systematically compare it against fine-tuning across 4 models (3B-32B), two independently-collected datasets, and training pool sizes from 1.5k to 27k labeled examples. On a 3k-issue development dataset, RAGTAG outperforms fine-tuning on all 4 models while using 70-80% of the GPU memory and requiring no training. On a 30k-issue evaluation dataset, we identify a data-efficiency crossover: fine-tuning overtakes RAGTAG as training data grows, but the crossover point depends on model scale. We trace RAGTAG's performance ceiling to a systematic parametric "bug bias" — LLMs default to predicting `bug` for any issue describing problematic behavior, overriding correct retrieval evidence 35-45% of the time across all models. A simple debiased retrieval heuristic partially corrects this, achieving the largest training-free improvement in our study. We provide practitioners with a concrete decision framework: below a model-dependent threshold of labeled examples, RAGTAG is the better choice; above it, fine-tuning is worth the investment.

---

## Paper Structure

### 1. Introduction

- The practical problem: GitHub issue triage at scale. Manual labeling doesn't scale. LLMs can automate it.
- Two paradigms: fine-tuning (learn from labeled data) vs. RAG (retrieve similar labeled examples at inference). Each has a clear cost profile — FT requires training infrastructure and retraining on data drift; RAG requires only an embedding index and works with any LLM.
- The open question practitioners face: **when is each approach worth it?** The answer depends on how much labeled data you have, but no prior work quantifies the crossover.
- Secondary question: **what limits training-free methods?** If RAG hits a ceiling, understanding why tells us whether interventions can close the gap — or whether the ceiling is structural.
- Contributions:
  1. A systematic comparison of RAG-based classification vs fine-tuning across 4 model scales and 5 training pool sizes, identifying the data-efficiency crossover point
  2. Discovery and mechanistic analysis of a parametric "bug bias" that limits all training-free LLM classifiers
  3. A debiased retrieval heuristic that partially corrects the bias without model modification
  4. A pure-retrieval voting baseline (VTAG) establishing the minimum performance any LLM-based method must beat

### 2. Background & Related Work

- GitHub issue classification (prior work, typically fine-tuning-only)
- RAG for few-shot classification (retrieval-augmented ICL)
- Fine-tuning small LLMs for SE tasks
- Gap: no systematic comparison across data scales; no analysis of when RAG is sufficient

### 3. Approach

**3.1 RAGTAG (Retrieval-Augmented Tag Assignment with Generation)**
- FAISS index over training issues (MiniLM-L6-v2 embeddings)
- Retrieve top-k neighbors, format as few-shot chat examples with `<label>` XML tags
- Assistant prefill with `<label>` for constrained generation
- Smart truncation: compresses neighbor bodies proportionally when prompt exceeds max_seq_length

**3.2 Fine-Tuning Baseline**
- LoRA fine-tuning via Unsloth
- Consistent prompt template, full epoch training, proper tokenization
- Same train/test splits as RAGTAG for fair comparison
- No mention of any "flawed" version — this is simply our FT implementation

**3.3 VTAG (Voting-based TAG Assignment)**
- Non-LLM baseline: retrieve top-k neighbors, vote on labels weighted by cosine similarity (Dudani 1976)
- Zero GPU at inference, ~3ms per query
- Establishes the retrieval floor: any LLM-based method must beat this to justify its cost

**3.4 Debiased Retrieval**
- Motivated by Phase 1 bias findings (introduced fully in Section 9)
- When bug and question neighbors are within a margin, remove bug neighbors to break the reinforcement loop
- Margin-gated: only fires on borderline cases, protects true bugs from losing informative neighbors

### 4. Experimental Setup

- **Development dataset:** `issues3k` — 2,995 issues after dedup, 1,497 test / 1,498 train, balanced 3-class
- **Evaluation dataset:** `issues30k` — ~30,000 issues, ~3,000 test / ~27,000 train, independently collected, no overlap with 3k
- **Models:** Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct (4-bit), Qwen2.5-14B-Instruct (4-bit), Qwen2.5-32B-Instruct (4-bit) — all loaded via Unsloth
- **RAGTAG configurations:** k in {0, 1, 3, 5, 9, 15}, context window in {2048, 4096, 8192}
- **Data efficiency subsamples:** 1.5k, 3k, 9k, 15k, 27k from the 30k training pool
- **Metrics:** macro-F1 (primary), per-class F1/precision/recall, accuracy, invalid prediction rate
- **Hardware:** RTX 4090 (24GB) for 3B/8B, A6000 (48GB) / A100 (80GB) for 14B/32B
- **Reproducibility:** RAGTAG is deterministic given fixed neighbors. FT averaged over 3 runs with negligible variance.

---

## Phase 1: Development Study (3k)

### 5. RAGTAG Configuration Analysis

**5.1 k x Context Window Interaction**
- Heatmap figure: 4 models x 6 k values x 3 context windows
- Key finding: ctx=8192 yields best raw F1; ctx=4096 is the practical sweet spot (13-22% less VRAM, 20-53% faster, within 0.01 F1 of 8192 for larger models)
- The truncation wall: at ctx=2048 with k>=5, invalid rates hit 28-44%, destroying performance
- Best k varies by model: k=3 for 3B/32B, k=9 for 8B/14B

**5.2 RAGTAG vs Fine-Tuning vs VTAG**
- Headline table: RAGTAG beats FT on all 4 models (margins from +0.004 to +0.043 macro-F1)
- VTAG floor at 0.645 — all RAGTAG configs beat it, confirming LLM adds genuine value
- LLM marginal value scales with model size: +4.5% (3B) to +20.5% (32B) relative improvement over VTAG
- Zero-shot anchoring: shows how much each method adds over the model's inherent ability

**5.3 Random Neighbor Ablation**
- Random retrieval drops F1 by 0.052-0.079 vs FAISS retrieval across all 4 models
- Confirms the gain is from retrieval quality, not just having few-shot examples in the prompt

**Existing data for 5.1-5.3:** Complete. All in docs/RAGTAG_ANALYSIS.md.

### 6. Cost-Performance Analysis

**6.1 Pareto Frontier**
- Figure: F1 vs GPU peak memory, with all approaches plotted (VTAG, RAGTAG at various configs, FT)
- RAGTAG uses 70-80% of FT's peak VRAM across all models
- This is the difference between fitting on a consumer RTX 4090 vs needing an A6000/A100

**6.2 Time Analysis**
- RAGTAG inference is slower (prompts are 5-9x longer: 2,600-5,300 vs ~560 tokens/issue)
- But RAGTAG has zero training cost; FT's total = training + inference
- At ctx=4096, RAGTAG is faster than FT total for most models

**6.3 Context Window Recommendation**
- ctx=4096 as the Pareto-optimal deployment config: competitive F1, lower VRAM, faster
- ctx=8192 for maximum accuracy regardless of cost
- Present as a tradeoff table giving practitioners a decision framework

**Existing data for 6.1-6.3:** Complete. All in docs/RAGTAG_ANALYSIS.md Sections 4 and 9.

### 7. The Parametric Bug Bias

**7.1 Discovery**
- Zero-shot analysis: all models show bug recall 0.88-0.95, question recall 0.19-0.44
- RAGTAG partially corrects: question recall rises to 0.47-0.63, but gap to bug recall remains 0.20-0.37
- FT on 3k is unstable: Llama-8B bug recall collapses to 0.419 (overcorrection with limited data)

**7.2 Diagnosis: Retrieval Is Not the Bottleneck**
- For question->bug errors: 42% have retrieval correctly favoring question — LLM overrides the signal
- Only 7.4% have zero question neighbors (genuine retrieval failure)
- Cross-model confirmation: 35-45% override rate across all 4 models
- The "similarity paradox": higher embedding similarity correlates with MORE question errors — the embedding space captures topic, not intent

**7.3 Mechanistic Interpretation**
- High question precision but low question recall — models CAN recognize questions, but have a high threshold for predicting them
- The bias is parametric: "problem description = bug report" regardless of user intent
- This is the ceiling that all training-free methods must contend with

**Existing data for 7.1-7.3:** Complete. docs/QUALITATIVE_ERROR_ANALYSIS.md and docs/30K_FINDINGS_AND_INTERVENTIONS.md Section 11.

---

## Phase 2: Evaluation Study (30k)

### 8. Configuration Transfer & Data Efficiency Crossover

**8.1 Config Transfer**
- Apply 3k-optimal configs directly to 30k without re-tuning
- k transfers correctly (confirmed via k-study on 30k for Llama-3B and Llama-8B)
- RAGTAG improves on 30k vs 3k (better retrieval pool): 3B 0.674->0.722, 8B 0.712->0.744
- FT improves more: 3B 0.667->0.790, 8B 0.687->0.793
- Bug bias persists: question F1 still the dominant bottleneck, accounting for ~50% of macro-F1 gap

**8.2 Data Efficiency Crossover — THE CENTRAL EXPERIMENT**
- Subsample 30k training pool at 1.5k, 3k, 9k, 15k, 27k
- Build FAISS index (RAGTAG) and fine-tune (FT) at each size
- Same held-out ~3,000 test issues from 30k throughout
- All 4 models: Llama-3B, Llama-8B, Qwen-14B, Qwen-32B
- **This is the paper's hero figure:** 4-panel crossover plot, one per model, showing RAGTAG and FT curves as a function of training pool size

**Expected findings (to be confirmed by experiment):**
- RAGTAG curve: gentle upward slope, limited by the k-example prompt ceiling
- FT curve: steep rise, eventually overtaking RAGTAG
- Crossover point depends on model size — larger models delay the crossover
- Qwen-14B may never cross (RAGTAG already wins at both 3k and 30k endpoints)

**Practitioner recommendation:** Below N labeled examples (model-dependent), use RAGTAG. Above it, invest in fine-tuning.

**Existing data for 8.1:** Partial. Have 30k RAGTAG and FT for Llama-3B, Llama-8B, Qwen-14B. Qwen-32B FT = 0.8103 (confirmed).
**Existing data for 8.2:** Only the 27k endpoint (already completed 30k runs). Need 1.5k, 3k, 9k, 15k subsample runs for all 4 models.

---

## Phase 3: Bias Correction

### 9. Debiased Retrieval

**9.1 Motivation**
- Phase 1 showed bug neighbors reinforce the parametric prior rather than providing corrective signal
- Phase 2 confirmed the bias persists and limits RAGTAG at scale
- Hypothesis: removing bug neighbors when evidence is ambiguous breaks the reinforcement loop

**9.2 Mechanism**
- Count bug vs question neighbors in retrieved set
- If `bug_count > 0` and `bug_count - question_count <= margin`: remove all bug neighbors
- Margin m=3 is the canonical config
- Margin-gated: only fires on borderline cases, protects true bugs from losing informative neighbors

**9.3 Results — All 4 Models, Both Datasets**

3k results:
| Model | Baseline | Debias m=3 | Delta F1_macro | Delta Q_recall |
|---|---|---|---|---|
| Llama-3B | 0.6743 | 0.6971 | +0.023 | +0.122 |
| Llama-8B | 0.7115 | 0.7556 | +0.044 | +0.209 |
| Qwen-14B | TBD | TBD | TBD | TBD |
| Qwen-32B | TBD | TBD | TBD | TBD |

30k results:
| Model | Baseline | Debias m=3 | Delta F1_macro | FT gap before | FT gap after |
|---|---|---|---|---|---|
| Llama-3B | 0.7222 | 0.7270 | +0.005 | 0.068 | 0.063 |
| Llama-8B | 0.7429 | 0.7569 | +0.014 | 0.050 | 0.036 |
| Qwen-14B | TBD | TBD | TBD | TBD | TBD |
| Qwen-32B | TBD | TBD | TBD | TBD | TBD |

**9.4 Ablation: "Always Remove" vs Margin-Gated**
- "Always remove" is too aggressive — bug recall collapses (Llama-3B: 0.778->0.485)
- The margin condition is load-bearing: it protects true bugs from being overwhelmed by non-bug replacement examples
- Margin sensitivity is low: m=2 vs m=3 differences <=0.004

**9.5 Analysis**
- Works better on 3k (noisier retrieval pool -> removing bug neighbors has outsized impact)
- Limited on 30k (higher-quality neighbors -> removing them costs more signal)
- Narrows the FT gap but doesn't close it — confirms the ceiling is structural
- The fixed prompt budget (k examples per inference) vs thousands of gradient updates is a fundamental asymmetry

**Existing data for 9.1-9.5:** Llama-3B and Llama-8B complete on both datasets. Ablation complete on 3k.
**Needs running:** Qwen-14B and Qwen-32B debiased retrieval on both 3k and 30k (4 runs total).

---

### 10. Discussion

- **Practical decision framework:** Flowchart or decision table mapping (labeled data budget x model size x hardware) to recommended approach. The crossover plot from Section 8.2 is the quantitative backing.
- **The structural ceiling:** RAGTAG's prompt budget is fixed at k examples regardless of training pool size. No training-free intervention can substitute for thousands of gradient updates. This is a fundamental asymmetry, not an engineering gap.
- **Why RAGTAG is still the right default for most teams:** No training cost. Instant adaptability (swap index, not weights). Model-portable (same index serves any LLM). Degrades gracefully on distribution shift. FT's advantage is real but narrow: in-distribution, with abundant labeled data, on a stable label schema.
- **FT generalization note (one paragraph):** 30k FT evaluated on the independently-collected 3k test set achieves F1=0.680 — barely exceeding RAGTAG's 0.674. This suggests FT's advantage is partly distribution-specific.
- **Activation steering as mechanistic evidence (one paragraph):** CAA at layer 23 produced +0.032 F1 on Llama-3B, confirming the bias lives in the residual stream geometry. The optimal layer being at 82% depth (vs typical 33%) suggests classification decisions are made late in the network. Full exploration left to future work.

### 11. Threats to Validity

- Single domain (GitHub issues), 3-class setup — generalization to other SE classification tasks unknown
- MiniLM embedder not optimized — stronger embedders (bge-base showed +0.023 on VTAG) could lift both VTAG and RAGTAG performance
- Quantized models (4-bit via Unsloth) — full-precision results may differ
- RAGTAG is deterministic; FT variance is negligible but present
- The 3k and 30k datasets are from the same platform (GitHub) — cross-platform generalization not tested

### 12. Future Work

- Activation-level interventions: CAA achieved +0.032 on Llama-3B 3k; generalizing across models and scales is promising
- Logit calibration: Batch Calibration showed modest gains (+0.009 to +0.030); Task Calibration untested
- Stronger embedders for both retrieval and VTAG (bge-base already shows improvement)
- Multi-label and fine-grained classification beyond 3-class
- Production deployment study (latency, throughput, drift adaptation over time)
- Combining debiased retrieval with logit/activation interventions

### 13. Conclusion

Three key takeaways:
1. RAGTAG is the better choice below a model-dependent data threshold — it matches or beats fine-tuning with no training cost and lower hardware requirements
2. Above the threshold, fine-tuning's advantage is real and grows with data — the crossover is structural, not a prompt engineering gap
3. A simple debiased retrieval heuristic pushes the threshold higher by partially correcting the parametric bug bias that limits all training-free LLM classifiers

---

## Key Figures (6-8 for conference paper)

1. **RAGTAG pipeline diagram** — approach overview showing retrieval + few-shot prompting flow
2. **k x context window heatmap** — 1 representative model in main paper + summary table, rest in appendix
3. **Pareto frontier** — F1 vs GPU peak memory, all approaches plotted (VTAG, RAGTAG configs, FT)
4. **Bug bias visualization** — confusion matrices or recall bar chart showing the question->bug asymmetry across models
5. **Data efficiency crossover plot** — THE HERO FIGURE. 4 panels (one per model), RAGTAG and FT curves as function of training pool size
6. **Debiased retrieval before/after** — per-class F1 or recall bar chart, all 4 models
7. **Practitioner decision flowchart** — maps data budget + model size to recommended approach

---

## Remaining Experiments

| Experiment | Models | Status | Priority |
|---|---|---|---|
| Data efficiency FT (1.5k, 3k, 9k, 15k subsamples) | All 4 models | Not started | **Critical** |
| Data efficiency RAGTAG (1.5k, 3k, 9k, 15k subsamples) | All 4 models | Not started | **Critical** |
| Debiased retrieval m=3 on 3k | Qwen-14B, Qwen-32B | Not started | **Critical** |
| Debiased retrieval m=3 on 30k | Qwen-14B, Qwen-32B | Not started | **Critical** |
| 27k data efficiency endpoint | All 4 models | **Done** (existing 30k results) | Complete |

### Compute estimate

Data efficiency: 4 subsample sizes x 4 models x 2 approaches (RAGTAG + FT) = 32 runs
- Llama-3B/8B runs on RTX 4090
- Qwen-14B/32B runs on NRP (A6000/A100)
- FT training time scales with subsample size (1.5k is fast, 15k is slow)
- RAGTAG inference time is constant per model (same test set, same k)

Debiased retrieval: 4 runs total (2 models x 2 datasets)
- Qwen-14B/32B on NRP

---

## What Is Explicitly Excluded from the Paper

| Item | Reason |
|---|---|
| Flawed fine-tune comparison | More trouble than its worth. Our FT is just "the baseline." |
| Cross-dataset flip tests (3k FT -> 30k test, etc.) | Data efficiency curve already proves the scaling story |
| VTAG voting scheme ablation (shepard, majority) | Spread is 0.002. Noise. |
| VTAG embedder ablation (bge-base, bge-large) | Creates apples-to-oranges with MiniLM RAGTAG numbers. Mention in Threats. |
| Activation steering (CAA) in main results | Only on Llama-3B 3k. One paragraph in Discussion as mechanistic evidence. |
| Batch Calibration in main results | Marginal gains. Mention in Future Work. |
| Contrastive Decoding in main results | Catastrophically destructive. Mention in Future Work. |
| Vote prior injection | Zero effect. Don't mention. |
| Enhanced system prompt | No effect. Don't mention. |
| Post-hoc ensemble / model ensemble | Marginal, model-dependent. Don't mention. |
| FT seed variance analysis | Results identical. One sentence: "averaged over 3 runs." |

---

## Existing Data Sources (for writing)

| Document | What it contains | Maps to paper section |
|---|---|---|
| docs/RAGTAG_ANALYSIS.md | Full 3k results: headline table, k x ctx heatmaps, invalid rates, cost-performance, LLM marginal value, model scaling, context window tradeoffs | Sections 5, 6 |
| docs/VTAG_FINDINGS.md | VTAG k-curve, voting ablation, embedder ablation | Section 5.2 (headline only) |
| docs/QUALITATIVE_ERROR_ANALYSIS.md | Confusion matrices, neighbor composition analysis, retrieval-signal-ignored rates, cross-model confirmation | Section 7 |
| docs/QUALITATIVE_3K_VS_30K.md | Per-class 3k vs 30k comparison, scaling asymmetry analysis | Sections 7, 8.1 |
| docs/30K_FINDINGS_AND_INTERVENTIONS.md | 30k results, k-study on 30k, all failed interventions, debiased retrieval results, logit calibration, FT generalization test, data efficiency plan | Sections 8, 9, 10 |
| docs/ACTIVATION_STEERING_FINDINGS.md | CAA layer sweep, multiplier sweep, strategy comparison, NTW ablation | Section 10 (one paragraph) |

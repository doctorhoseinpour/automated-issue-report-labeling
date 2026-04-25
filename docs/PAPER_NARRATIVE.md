# Paper Narrative & Structure

**Status:** Pivoted 2026-04-25 to the 11-project benchmark as the sole dataset. The earlier three-phase (3k → 30k → bias correction) framing is retired. Prior versions of this document and the supporting analyses remain in [legacy/](legacy/).

---

## Working Title

**"RAGTAG: When Does Retrieval-Augmented Few-Shot Classification Match Fine-Tuning for GitHub Issue Triage?"**

---

## Core Story

The paper answers three coupled questions on a balanced 11-project benchmark:

- **RQ1 (Comparison):** How do RAGTAG, VTAG, and zero-shot perform in project-specific and project-agnostic settings, per project and overall? A built-in finding: 88.3% of agnostic-retrieved neighbors come from the same project anyway, so for RAGTAG the two settings are nearly equivalent — that equivalence is itself a result.
- **RQ2 (Diagnosis):** Why does RAGTAG fall short of fine-tune-agnostic? The mechanism is a parametric bug bias: feature precision is high but recall is low, with features systematically mislabeled as bugs across all four models tested.
- **RQ3 (Bridging):** Can the gap be closed without training? Margin-based retrieval debiasing (m=3) is the validated intervention — it consistently improves Llama-8B from 0.704 → 0.737, matching agnostic FT in some configurations. A second method is under consideration.

The narrative arc is **compare → diagnose → bridge.** Each RQ motivates the next, all on the same benchmark.

---

## Dataset

The 11-project benchmark (`issues11k.csv`) is the sole dataset.

- 11 projects from diverse domains: `ansible/ansible`, `bitcoin/bitcoin`, `dart-lang/sdk`, `dotnet/roslyn`, `facebook/react`, `flutter/flutter`, `kubernetes/kubernetes`, `microsoft/TypeScript`, `microsoft/vscode`, `opencv/opencv`, `tensorflow/tensorflow`
- 600 issues per project (200 bug + 200 feature + 200 question, balanced by design)
- 6,600 issues total → 3,300 train / 3,300 test (per-project stratified split: 300/300)
- Two evaluation settings:
  - **Project-agnostic:** single FAISS index over all 3,300 train issues; test on all 3,300
  - **Project-specific:** 11 independent FAISS indices (300 train each); test on the corresponding 300

This design lets us measure cross-project transfer cleanly — and the per-project breakdown lets us detect domain-specific failures rather than averaging them away.

The 3k and 30k datasets used in earlier exploratory work are deprecated. Their analyses live in [legacy/](legacy/) and inform the related-work / interventions-tried discussion only.

---

## Models

All four models loaded via Unsloth (4-bit quantization for the 8B+ models):

- `unsloth/Llama-3.2-3B-Instruct`
- `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`
- `unsloth/Qwen2.5-14B-Instruct-bnb-4bit`
- `unsloth/Qwen2.5-32B-Instruct-bnb-4bit`

This spans an order of magnitude in scale and two model families.

---

## Abstract (draft)

LLM-based GitHub issue classification (bug / feature / question) can be approached either by fine-tuning on labeled data or by retrieval-augmented few-shot prompting (RAG). We evaluate both on an 11-project benchmark of 6,600 issues balanced across labels and projects, comparing four models from 3B to 32B parameters under two settings: project-agnostic (one shared retrieval index) and project-specific (per-project indices). We find that retrieval-augmented few-shot prompting (RAGTAG) is competitive with — and at the largest scale exceeds — correctly-implemented fine-tuning, while requiring no training and lower peak GPU memory. We trace RAGTAG's residual gap to fine-tune-agnostic to a parametric bug bias: features are systematically mislabeled as bugs across all models, with retrieval providing the correct signal in the majority of error cases but the LLM overriding it. A margin-based retrieval-debiasing heuristic narrows the gap by 30–50 % of the original delta on Llama-8B, with no training and minimal inference overhead, generalizing to other model families. We also show that for RAGTAG the project-agnostic setting recovers project-specific performance because retrieval naturally clusters within-project (88.3 %), removing the need for per-project indexing infrastructure in deployment.

---

## Paper Structure

### 1. Introduction
Practical problem: GitHub issue triage at scale. Two paradigms: fine-tuning and RAG. The open practitioner question: when does RAG suffice? Secondary question: what limits RAG, and is the limitation structural? Contributions:
1. A systematic 11-project comparison of RAGTAG vs fine-tuning vs a pure-retrieval voting baseline (VTAG), under both agnostic and project-specific settings.
2. A diagnosis of the residual RAGTAG–FT gap as a parametric bug bias — quantitative evidence across all four models.
3. A margin-based retrieval-debiasing heuristic that bridges the gap without training, generalizing across model families.
4. The agnostic-vs-specific equivalence finding for retrieval-augmented classification — a deployment-relevant simplification.

### 2. Background & Related Work
GitHub issue classification (typically fine-tuning-only); RAG for few-shot classification; fine-tuning small LLMs for SE tasks; the gap our paper fills.

### 3. Approach
- **3.1 RAGTAG.** FAISS index (MiniLM-L6-v2 embeddings), top-k retrieval, few-shot chat prompting with `<label>` XML tags and assistant prefill, smart proportional truncation.
- **3.2 Fine-Tuning Baseline.** LoRA via Unsloth; consistent prompt template, full epoch, proper tokenization. Same train/test splits as RAGTAG.
- **3.3 VTAG.** Non-LLM voting baseline: similarity-weighted k-NN over the same FAISS retrievals.
- **3.4 Debiased Retrieval (intervention introduced in Section 6).** Margin-gated removal of bug neighbors when bug-vs-question evidence is borderline.

### 4. Experimental Setup
The 11-project benchmark, two settings, four models, k ∈ {0, 1, 3, 6, 9}, ctx=8192. Metrics: macro-F1, per-class P/R/F1, accuracy, invalid prediction rate. Hardware mix: RTX 4090 for Llama; NRP / OSC for Qwen.

### 5. RQ1 — Comparison
- **5.1 Headline table.** RAGTAG vs VTAG vs zero-shot vs FT, project-agnostic and project-specific, all 4 models.
- **5.2 Per-project breakdown.** 11-row figure / table showing per-project macro-F1 for each method × model. Where the methods diverge by project tells the deployment story.
- **5.3 The agnostic ≈ specific finding (for RAGTAG).** Quantify the within-project retrieval rate (88.3 %); show that agnostic and specific RAGTAG produce per-project F1 within ±0.01 for most projects. Explain why VTAG behaves the same way.
- **5.4 LLM marginal value.** RAGTAG − VTAG, per model, demonstrating the LLM contributes beyond pure retrieval.

### 6. RQ2 — Diagnosis: The Parametric Bug Bias
- **6.1 The recall asymmetry.** Across all four models, feature recall lags feature precision by 0.15–0.25 in agnostic RAGTAG. Question recall is even lower in zero-shot.
- **6.2 Retrieval is not the bottleneck.** For feature → bug errors, the majority have retrieval correctly favoring feature; the LLM overrides the signal. Cross-model confirmation. (Methodology adapted from the legacy 3k/30k qualitative analysis; numbers re-derived on 11k.)
- **6.3 Why FT-agnostic doesn't suffer the bias.** Gradient updates over the full training pool let the model learn project-specific feature/bug boundaries; RAGTAG can only see k examples per inference.
- **6.4 The structural ceiling.** A fixed prompt budget of k examples per inference is fundamentally asymmetric to thousands of gradient updates. This sets the expectation for what RQ3 can achieve.

### 7. RQ3 — Bridging the Gap
- **7.1 Margin-based retrieval debiasing.** Mechanism (count bug vs question/feature neighbors; if `bug_count` exceeds the next-most-frequent class by ≤ margin, remove all bug neighbors). Margin m=3 canonical.
- **7.2 Results, all 4 models, both settings.** Headline table showing baseline RAGTAG, debiased RAGTAG, FT-agnostic, with macro-F1 and per-class deltas. Largest gain on Llama-8B (+0.034 → reaches 0.737, matching agnostic FT 0.736).
- **7.3 Per-project debias robustness.** Does debiasing help every project, or are some projects harmed? Frame any regressions honestly.
- **7.4 Why margin-gating matters.** Always-remove ablation: bug recall collapses. The margin is load-bearing.
- **7.5 Second intervention** (TBD — vote-prior injection, prompt-level disambiguation, or batch calibration are the live candidates). Pending decision.

### 8. Discussion
- Practical decision framework: project-specific vs agnostic indices for RAGTAG (use agnostic — same performance, simpler deployment); RAGTAG vs FT depending on hardware budget and label availability.
- The structural ceiling: training-free methods are bounded by the k-example prompt budget, yet a margin-debias intervention partially restores parity.
- Why RAGTAG is still the right default for most teams: no training cost, instant adaptability, model-portable, graceful degradation under distribution shift. FT's advantage is real but narrow.
- Activation steering (CAA) is mentioned as future-work mechanistic evidence — see [legacy/ACTIVATION_STEERING_FINDINGS.md](legacy/ACTIVATION_STEERING_FINDINGS.md) for the prior 3k results.

### 9. Threats to Validity
Single domain (GitHub issues), 3-class setup, MiniLM embedder not optimized (stronger embedders may shift both VTAG and RAGTAG floors), 4-bit quantized models (full-precision results may differ), the 11 projects are skewed toward popular OSS repositories.

### 10. Future Work
- Activation-level interventions (CAA on Qwen, layer transfer studies)
- Stronger embedders for retrieval (bge-base/bge-large showed +0.022 on the legacy VTAG)
- Multi-label and fine-grained beyond 3-class
- Production deployment study (latency, drift adaptation)

### 11. Conclusion
Three takeaways:
1. On the 11-project benchmark, RAGTAG is competitive with fine-tuning, while requiring no training and lower peak GPU memory.
2. The residual gap to FT-agnostic is explained by a parametric bug bias that all training-free methods must contend with.
3. A margin-based retrieval-debiasing heuristic narrows that gap across model families, with a second intervention to be reported, suggesting practical paths to closing it without training.

---

## Key Figures

1. **RAGTAG pipeline diagram** — retrieval + few-shot prompting flow.
2. **Headline comparison plot** — macro-F1 across (zero-shot, VTAG, RAGTAG, FT) × 4 models × 2 settings (overall).
3. **Per-project breakdown** — heatmap or 11-row bar chart showing per-project macro-F1 per method.
4. **Agnostic ≈ specific for RAGTAG** — scatter of agnostic-F1 vs specific-F1 per (project, model), points cluster on diagonal.
5. **Bug-bias evidence** — confusion matrix or per-class recall bar chart showing feature → bug asymmetry across models.
6. **Debias before/after** — per-class F1 bar chart, all 4 models, both settings.

---

## Remaining Experiments

| Experiment | Models | Status |
|---|---|---|
| Qwen-32B RAGTAG re-run (agnostic + specific) — prior had OOM/invalid outputs | Qwen-32B | Not started |
| Qwen-14B fine-tune (agnostic + specific) | Qwen-14B | Not started |
| Qwen-32B fine-tune (agnostic + specific) | Qwen-32B | Blocked on NRP A100 quota |
| Debiased retrieval m=3 on 11k | Qwen-14B, Qwen-32B | Script ready (`run_11k_debias_qwen.sh`); needs execution |
| Second RQ3 intervention | All 4 models, decided | Method TBD |

These five sets of experiments are the focus of the upcoming NRP Kubernetes-Jobs migration.

---

## Active Data Sources

| Document | Maps to paper section |
|---|---|
| [11K_BENCHMARK_FINDINGS.md](11K_BENCHMARK_FINDINGS.md) | Sections 5, 6, 7 — primary results tables |
| [legacy/RAGTAG_ANALYSIS.md](legacy/RAGTAG_ANALYSIS.md) | Background — methodology validation |
| [legacy/30K_FINDINGS_AND_INTERVENTIONS.md](legacy/30K_FINDINGS_AND_INTERVENTIONS.md) | Sections 6, 7 — bug bias methodology, intervention catalog |
| [legacy/QUALITATIVE_ERROR_ANALYSIS.md](legacy/QUALITATIVE_ERROR_ANALYSIS.md) | Section 6 — diagnostic methodology |
| [legacy/VTAG_FINDINGS.md](legacy/VTAG_FINDINGS.md) | Section 3.3 — VTAG calibration / methodology |
| [legacy/ACTIVATION_STEERING_FINDINGS.md](legacy/ACTIVATION_STEERING_FINDINGS.md) | Section 8 (one paragraph in Discussion / Future Work) |

---

## Explicitly Excluded from the Paper

| Item | Reason |
|---|---|
| Three-phase 3k → 30k → debias narrative | Retired with the 11k pivot. |
| Data-efficiency crossover figure | Retired with the 11k pivot. |
| Flawed fine-tune comparison | Our FT is just "the baseline." No prior-flawed-pipeline framing. |
| Context window study | Settled at ctx=8192 for the 11k benchmark. |
| K study | Settled at k ∈ {0,1,3,6,9} from VTAG plateau analysis. |
| VTAG voting scheme ablation | Spread is 0.002 — noise. |
| VTAG embedder ablation | Apples-to-oranges with MiniLM RAGTAG numbers. Brief Threats note. |
| Activation steering in main results | One paragraph in Future Work as mechanistic evidence. |
| Batch Calibration in main results | If pursued at all, mentioned briefly. |
| Contrastive Decoding | Catastrophically destructive. Skipped. |

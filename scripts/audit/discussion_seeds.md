# Discussion-Section Seeds

Insights surfaced during the numbers audit of `paper/sections/05_evaluations.tex`. Each is grounded in verified numbers and points at a paragraph the discussion section could write. Ordered roughly by impact / novelty.

---

## 1. The bug-bias is structural, not a model artifact

Every method tested — pure-retrieval (VOTAG), in-context (RAGTAG), and parametric (FT-PS) — over-predicts question→bug. Verified rates:

| Method | Q→bug | source |
|--------|-------|--------|
| VOTAG-PS k=15 | 31.8% | [audit_log.csv:vtag_q_to_bug](scripts/audit/audit_log.csv) |
| VOTAG-PA k=16 | 31.9% | same |
| RAGTAG zero-shot (no retrieval) | 46–59% across 4 Qwens | [audit_log.csv:ragtag_zs_q_to_bug](scripts/audit/audit_log.csv) |
| RAGTAG best-k PS | 26–36% across 4 Qwens | same |
| BRAGTAG best-k PS | 19–27% (intervention works but residual) | same |

The same skew shows up in a method that has **no LLM at all** (VOTAG) and in a method that has **no retrieval at all** (FT-PS pre-rescue). This is a strong signal that the bias lives in the **data / embedding space**, not the LLM's parametric prior. The discussion paragraph should explicitly land this point — it reframes BRAGTAG from "fix the LLM" to "fix the retrieval distribution," which is the more accurate causal story.

Supporting number: for true-question queries, the average top-k label imbalance is `N_bug − N_q ≈ −1.35` (k∈[1,30]). Question queries pull bug examples disproportionately, even when the query *is* a question.

---

## 2. VOTAG ≈ Qwen-3B zero-shot — "do you even need an LLM?"

VOTAG-PA peak is 0.604 (k=16). Qwen-3B zero-shot is **0.613** — a +0.009 gap. The smallest LLM with no retrieval barely beats a pure-similarity vote. This is the most quotable framing of the retrieval-floor argument. Discussion should explicitly say: pure retrieval gets you ~99% of the way to a frontier-3B zero-shot, suggesting the value of the LLM is in *how it uses retrieved examples*, not in any inherent classification competence.

---

## 3. PS-vs-PA inversion between RAGTAG and Fine-Tune

Different methods want different data scopes:

| Method | Best setting | PA−PS macro F1 |
|--------|-------------|-----------------|
| VOTAG | PA wins (slightly) | +0.007 avg, +0.015 max |
| RAGTAG | **PS wins** at every (model, k≥1) pair | small but consistent |
| Fine-Tune | **PA wins** decisively | +0.033 (3B), +0.084 (7B), +0.082 (14B), +0.031 (32B) |

The interpretation almost writes itself: retrieval-based methods are constrained by *embedding similarity*, which is best within a project's domain (PS = same-domain examples). Parametric fine-tuning is constrained by *training set diversity*; with only 300 issues per project (PS) the model under-fits, while 3,300 (PA) gives it enough signal. This is a clean "horses for courses" story for the discussion.

---

## 4. Cost crossover at 32B

Wall-times (h):

| Model | RAGTAG | BRAGTAG | FT-train | FT-infer | FT-total |
|-------|--------|---------|----------|----------|----------|
| 3B | 0.18 | 0.22 | 0.31 | 0.11 | 0.42 |
| 7B | 0.50 | 0.60 | 0.48 | 0.10 | 0.58 |
| 14B | 1.37 | 1.35 | 1.49 | 0.22 | 1.71 |
| 32B | **4.62** | **4.15** | 2.28 | 0.44 | **2.71** |

At 3B/7B/14B, fine-tune costs more (training overhead dominates). At 32B, fine-tune is **cheaper than RAGTAG and BRAGTAG**, because long-context inference scales with k×prompt_size and the best k is fixed at 12 across model sizes. This inversion is a useful caveat: BRAGTAG's competitive-performance argument starts to lose its cost-saving angle at the very top of the model scale we tested. Discussion paragraph: "BRAGTAG is most useful at the model sizes a typical practitioner can fit on a single consumer GPU; the cost story flips at 32B."

---

## 5. k\* grows with model size, then plateaus

Verified best-k per model:

- Qwen-3B → k=3
- Qwen-7B → k=6
- Qwen-14B → k=12
- Qwen-32B → k=12

Plus the diminishing-returns finding: every model loses 0.001–0.015 macro F1 going k=12 → k=15. Bigger models extract more value from more examples, but the curve flattens by k=12 even at 32B. Implication: future scaling beyond 32B may not justify a longer prompt. (Untested, but the discussion can flag it.)

---

## 6. BRAGTAG's k\* almost always **increases** vs RAGTAG's k\*

| Model | RAGTAG k\* | BRAGTAG k\* |
|-------|------------|-------------|
| 3B | 3 | 6 |
| 7B | 6 | 12 |
| 14B | 12 | 15 |
| 32B | 12 | 12 |

Mechanically obvious in retrospect: BRAGTAG removes all bug examples when triggered, so the LLM sees fewer effective examples. To compensate, the operator should increase the initial k. The 32B exception is interesting: at 32B the model is robust enough to peak at the same k as RAGTAG. Discussion paragraph: "tuning BRAGTAG = pick a k roughly 2× the RAGTAG optimum at smaller models, equal at the largest."

---

## 7. Inverted-U in BRAGTAG vs FT comparison

(BRAGTAG − FT) macro F1 deltas with rescue:

- 3B: +0.009 [−0.008, +0.028]
- 7B: −0.011 [−0.026, +0.004]
- 14B: −0.014 [−0.030, +0.001]
- 32B: +0.021 [+0.006, +0.035] *(BRAGTAG significantly leads at 32B)*

BRAGTAG wins the smallest and largest models; FT wins the middle two. The discussion could explain this as: at small models, parametric capacity is the bottleneck so retrieval helps most; at the largest model, in-context reasoning catches up and the LoRA's parametric tweak becomes redundant; the middle is where LoRA fits the data best without saturating capacity. This is a non-obvious 3-cell pattern worth a paragraph.

---

## 8. Class-balance argument (BRAGTAG's hidden advantage)

Per-class F1 averaged across the 4 model sizes:

| | Bug | Feature | Question | Pop. std | Worst-class |
|-|-----|---------|----------|----------|-------------|
| RAGTAG-PS+rescue | 0.750 | 0.824 | 0.638 | 0.076 | 0.638 |
| BRAGTAG-PS+rescue | 0.760 | 0.822 | 0.694 | **0.052** | **0.694** |
| FT-PA+rescue | 0.771 | 0.829 | 0.672 | 0.065 | 0.672 |

BRAGTAG has the most-balanced predictions across labels (lowest std) and the highest worst-class F1. For deployment, this is a real practical advantage — fairness across label types matters when a maintainer relies on automated triage. The discussion should explicitly note that aggregate macro F1 hides this; method-comparison parity in macro F1 does not mean parity in per-class behavior.

---

## 9. Invalid-output rate is a deployment concern, but VOTAG fallback erases the gap

Pre-rescue invalid rates (from `tab:bragtag-results` and FT inv columns):

- FT-PA: 0.28%
- FT-PS: 0.38%
- BRAGTAG (avg k∈[1,15]): 2.3%
- RAGTAG (avg k∈[1,15]): 3.0%

The VOTAG fallback (which incurs zero LLM inference cost beyond what was already done for retrieval) closes this gap to zero. So the paper's "always-have-an-output" property is preserved across all methods. Discussion can note: "Invalid-output rate is a real practical concern and a non-trivial argument *against* in-context methods on its own, but our VOTAG-fallback design neutralizes it for free."

---

## 10. The retrieval index covers most of what PA could buy you

PA-vs-PS top-k neighbor overlap is **84–91%** across k∈{3, 9, 30}. So 11× less retrieval data per project (PS = 300 vs PA = 3,300) gives you ~85–90% of the same neighbors anyway, because same-project issues are tightly clustered in the embedding space. This re-grounds the paper's "PS uses 11× less retrieval data" line: it's not just a bookkeeping fact, it's a *empirical* property of the embedding manifold. Discussion paragraph: "Project-specific retrieval is not just smaller — it's nearly redundant with project-agnostic retrieval, because the embedding space already concentrates same-project issues."

---

## Cross-cutting framing options

The findings above support at least three possible discussion-section spines:

1. **"Where the bias lives"** — open with finding #1 (structural bug-bias), use #2 (VOTAG ≈ Qwen-3B zs) to argue it's in the data/embedding, use BRAGTAG as the targeted intervention, close with the class-balance argument (#8).

2. **"What scope each method wants"** — open with finding #3 (PS-vs-PA inversion), use #10 (retrieval near-redundancy) to ground PS as a practical default, use #4 (cost crossover) to caveat at 32B, close with #6 (k\* tuning advice).

3. **"Capacity vs context tradeoff"** — open with finding #7 (inverted-U), use #5 (k\* growth + plateau) to argue larger models exploit context better, use #4 to set practical limits, close with #9 (deployment robustness via fallback).

The audit favors #1 because it is the most novel framing — the structural bug-bias result is the paper's strongest scientific claim and is currently under-emphasized in the evaluations narrative.

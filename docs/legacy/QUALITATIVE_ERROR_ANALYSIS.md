# Qualitative Error Analysis: RAGTAG Question Misclassification

**Date:** 2026-04-17
**Model:** Llama-3.1-8B-Instruct (4-bit), issues30k, RAGTAG k=9, ctx=8192
**Goal:** Understand why RAGTAG systematically misclassifies `question` as `bug`, and whether the problem is retrieval or LLM reasoning.

---

## 1. Confusion Matrix (RAGTAG, Llama-8B, 30k)

|  | pred_bug | pred_feature | pred_question | pred_invalid | total |
|---|---|---|---|---|---|
| **true bug** | 885 | 28 | 48 | 39 | 1000 |
| **true feature** | 139 | 778 | 73 | 10 | 1000 |
| **true question** | 326 | 123 | 534 | 17 | 1000 |

- **326/1000 questions misclassified as bug** — the dominant error
- Asymmetric: bug→question only 48 times (6.2× fewer)
- Feature is relatively stable (778/1000 correct)
- 66 total invalid predictions (2.2%)

## 2. Is Retrieval the Problem?

For each question→bug error, we counted the neighbor label distribution (k=9 neighbors per issue).

### Neighbor composition for misclassified questions (n=326)

| Metric | bug neighbors | feature neighbors | question neighbors |
|---|---|---|---|
| Mean | 4.54 | 1.19 | 3.27 |
| Median | 5 | 1 | 3 |

### Neighbor composition for correctly classified questions (n=534)

| Metric | bug neighbors | feature neighbors | question neighbors |
|---|---|---|---|
| Mean | 1.76 | 2.11 | 5.13 |
| Median | 1 | 2 | 5 |

### Distribution: how many question neighbors did the errors have?

| Question neighbors | Count | % of errors |
|---|---|---|
| 0 | 24 | 7.4% |
| 1 | 44 | 13.5% |
| 2 | 72 | 22.1% |
| 3 | 55 | 16.9% |
| 4 | 43 | 13.2% |
| 5 | 32 | 9.8% |
| 6 | 26 | 8.0% |
| 7 | 17 | 5.2% |
| 8 | 12 | 3.7% |
| 9 | 1 | 0.3% |

**Key finding:** 42% of question→bug errors (137/326) had question neighbors >= bug neighbors. The retrieval gave the correct signal and the LLM overrode it.

Only 7.4% of errors had zero question neighbors — cases where retrieval genuinely failed.

## 3. The LLM Has a Bug Bias

The model defaults to `bug` when an issue describes problematic behavior, regardless of user intent. This is visible in both zero-shot and RAGTAG:

### Zero-shot → RAGTAG delta (question label, 30k)

| Model | ZS question recall | RAGTAG question recall | Δ |
|---|---|---|---|
| Llama-3B | 0.444 | 0.593 | +0.149 |
| Llama-8B | 0.433 | 0.534 | +0.101 |
| Qwen-14B | 0.535 | 0.628 | +0.093 |
| Qwen-32B | 0.579 | 0.655 | +0.076 |

RAGTAG *improves* question recall vs zero-shot (the examples help), but not enough. The bug bias is partially corrected but not eliminated.

### Sample misclassified titles (true=question, predicted=bug)

These issues describe problems but the user's intent is seeking help, not reporting a defect:

| # | Neighbor labels | Title |
|---|---|---|
| 1 | bug=2, q=6 | Bug or misuse not sure: Issue using envoy with upgrade connect and npm |
| 2 | bug=1, q=6 | Flask restx multipart/form request with file and body documented properly with swagger |
| 3 | bug=0, q=4 | It took quite a long time to send out retry request |
| 4 | bug=3, q=3 | Generated setup getting flagged as a virus by windows defender |
| 5 | bug=7, q=2 | Google Translate Extension proxied endpoint bug or not? |
| 6 | bug=7, q=2 | issue: "Illegal constructor" error when using classValidatorResolver with a file input |
| 7 | bug=4, q=5 | Getting exception "ConnectionException" when trying to Login |
| 8 | bug=6, q=3 | Why can't some commands be displayed. Is it because of the network? |
| 9 | bug=4, q=4 | FFmpeg session crashes |
| 10 | bug=8, q=0 | poetry lock fails after poetry-dynamic-versioning installation |

Common patterns in misclassified questions:
- Title describes an error/crash/failure (looks like a bug report)
- User is actually asking "how do I fix this?" or "is this expected?"
- Words like "issue", "error", "fails", "not working" trigger the bug bias
- Even titles with explicit question framing (#1 "not sure", #8 "Is it because...?") get overridden

### Sample correctly classified question titles (for contrast)

| # | Neighbor labels | Title |
|---|---|---|
| 1 | bug=0, q=6 | Multiple policies |
| 2 | bug=4, q=5 | [CoE Starter Kit - QUESTION] Why in Audit Log there is only LuachPowerApps |
| 3 | bug=1, q=7 | No timestamp-like field when loading markets for FTX |
| 4 | bug=0, q=7 | How to achieve idempotence for incoming events |
| 5 | bug=0, q=9 | How create a ModelMixin? |
| 6 | bug=0, q=6 | How to add LLVM to existing Visual Studio project |

Pattern: correctly classified questions tend to use explicit question framing ("How to...", help-seeking language) and fewer error/crash keywords.

## 4. Comparison with Fine-Tuning

For the 1000 true questions, comparing RAGTAG vs fine-tune predictions:

| Outcome | Count |
|---|---|
| Both correct (question, question) | 510 |
| Only RAGTAG correct | 24 |
| Only fine-tune correct | 312 |
| Both wrong | 154 |

- Fine-tune recovers 230 of 326 question→bug RAGTAG errors (correctly labels them question)
- Fine-tune learns the intent boundary through thousands of gradient updates
- But fine-tune trades this for lower bug recall: 88.5% (RAGTAG) → 73.4% (FT)
- 154 questions are hard for both approaches — likely genuinely ambiguous cases

### Per-label recall comparison (Llama-8B, 30k)

| Label | RAGTAG recall | FT recall | Δ |
|---|---|---|---|
| bug | 0.885 | 0.734 | -0.151 |
| feature | 0.778 | 0.819 | +0.041 |
| question | 0.534 | 0.822 | +0.288 |

Fine-tuning essentially rebalances the model: it sacrifices bug over-prediction for much better question detection. The net effect is +0.057 macro-F1 (0.735 → 0.792).

## 5. Root Cause Summary

| Factor | Evidence | Contribution |
|---|---|---|
| **LLM bug bias (primary)** | 42% of errors have retrieval signal favoring question; LLM ignores it | High |
| **Retrieval noise (secondary)** | 58% of errors have bug-majority neighbors for true questions | Medium |
| **Ambiguous ground truth** | 154 questions wrong by both RAGTAG and FT | Low-medium |

The problem is **not** that retrieval fails — it's that the LLM's prior ("problem description = bug") overrides the few-shot evidence. This is a reasoning-level issue that must be addressed through prompt engineering, not retrieval changes.

## 6. Cross-Model Analysis (30k dataset)

The same analysis across all 4 models confirms this is a universal LLM behavior, not model-specific.

### Question→bug errors: "LLM ignored retrieval signal" rate

| Model | k | question→bug errors | Retrieval correct (q≥bug) | % LLM ignored |
|---|---|---|---|---|
| Llama-3B | 3 | 273 | 95 | 34.8% |
| Llama-8B | 9 | 326 | 137 | 42.0% |
| Qwen-14B | 9 | 272 | 113 | 41.5% |
| Qwen-32B | 3 | 224 | 101 | 45.1% |

Consistent pattern: **35–45% of question→bug errors have retrieval correctly favoring question**, yet the LLM overrides to bug. Larger models (Qwen-32B at 45.1%) are slightly *worse* at following retrieval signal, possibly because their stronger priors are harder to override.

### Full confusion matrices (30k, RAGTAG best-k)

**Llama-3B (k=3)**

|  | →bug | →feature | →question | →invalid | recall |
|---|---|---|---|---|---|
| bug | 801 | 77 | 122 | 0 | 0.801 |
| feature | 95 | 780 | 125 | 0 | 0.780 |
| question | 273 | 134 | 593 | 0 | 0.593 |

**Llama-8B (k=9)**

|  | →bug | →feature | →question | →invalid | recall |
|---|---|---|---|---|---|
| bug | 885 | 28 | 48 | 39 | 0.885 |
| feature | 139 | 778 | 73 | 10 | 0.778 |
| question | 326 | 123 | 534 | 17 | 0.534 |

**Qwen-14B (k=9)**

|  | →bug | →feature | →question | →invalid | recall |
|---|---|---|---|---|---|
| bug | 855 | 43 | 62 | 40 | 0.855 |
| feature | 95 | 801 | 93 | 11 | 0.801 |
| question | 272 | 77 | 628 | 23 | 0.628 |

**Qwen-32B (k=3)**

|  | →bug | →feature | →question | →invalid | recall |
|---|---|---|---|---|---|
| bug | 808 | 41 | 74 | 77 | 0.808 |
| feature | 89 | 785 | 100 | 26 | 0.785 |
| question | 224 | 83 | 655 | 38 | 0.655 |

### All error flows: retrieval signal ignored rate

| Error direction | Llama-3B | Llama-8B | Qwen-14B | Qwen-32B |
|---|---|---|---|---|
| question→bug | 34.8% | 42.0% | 41.5% | 45.1% |
| question→feature | 41.8% | 34.1% | 45.5% | 42.2% |
| bug→question | 45.1% | 27.1% | 51.6% | 56.8% |
| feature→bug | 21.1% | 16.5% | 15.8% | 30.3% |

Key observations:
- **question→bug and question→feature** have the highest "LLM ignored" rates (35–45%) — the model's bug bias overrides retrieval evidence
- **feature→bug** has the lowest ignored rate (16–30%) — when retrieval says feature, the LLM mostly agrees. Retrieval is more reliable for this pair.
- **bug→question** also shows high ignored rates (27–57%) — the reverse confusion is equally an LLM reasoning issue, though it affects fewer cases

### Neighbor composition: errors vs correct (all models)

| Model | Group | Avg bug nb | Avg feat nb | Avg ques nb |
|---|---|---|---|---|
| Llama-3B | question→bug errors (273) | 1.79 | 0.26 | 0.96 |
| Llama-3B | correct questions (593) | 0.47 | 0.62 | 1.92 |
| Llama-8B | question→bug errors (326) | 4.54 | 1.19 | 3.27 |
| Llama-8B | correct questions (534) | 1.76 | 2.11 | 5.13 |
| Qwen-14B | question→bug errors (272) | 4.57 | 1.13 | 3.29 |
| Qwen-14B | correct questions (628) | 1.90 | 2.32 | 4.77 |
| Qwen-32B | question→bug errors (224) | 1.52 | 0.34 | 1.14 |
| Qwen-32B | correct questions (655) | 0.62 | 0.67 | 1.71 |

Error cases always have more bug neighbors and fewer question neighbors than correct cases — but the gap is not as large as expected. Even error cases have substantial question neighbors (avg 1–3 depending on k).

## 7. Implications for RAGTAG Enhancement

Two problems need different solutions:

**Problem A: LLM ignores retrieval signal (35–45% of errors)**
- The retrieval correctly identifies question neighbors but the LLM overrides with bug
- Fix: prompt engineering to make the LLM respect the evidence (label definitions, intent-focused instructions)
- An ensemble that defers to retrieval when confidence is high could also help

**Problem B: Retrieval gives wrong signal (55–65% of errors)**
- Bug neighbors dominate because question and bug issues are topically similar
- Fix: this is harder — the embedding space doesn't capture intent, only topic
- An ensemble approach could use the LLM's zero-shot judgment as a counterbalance

**Ensemble opportunity:** The fact that 35–45% of errors have correct retrieval signal means an ensemble that weights retrieval evidence more heavily could recover a significant portion of errors without any prompt changes. Combined with prompt engineering for the remaining 55–65%, the gap to fine-tuning could be substantially narrowed.

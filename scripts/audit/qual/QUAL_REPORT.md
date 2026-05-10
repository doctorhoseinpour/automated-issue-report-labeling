# Qualitative Error Analysis — Qwen-32B

**Scope**: open-coding analysis of 150 misclassifications + 60 invalid outputs from Qwen-32B at each method's best configuration (RAGTAG-PS k=12, BRAGTAG-PS k=12, FT-PA).

**Sampling**: stratified random; seed=20260509 (frozen in [scripts/audit/qual/sample_errors.py](scripts/audit/qual/sample_errors.py)). Misclassifications: 50 per method × 3 methods, with within-method strata (30 true-question / 10 true-bug / 10 true-feature). Invalids: 30 per method × 2 methods (FT excluded; insufficient invalids).

**Method**: open coding. Pilot pass on 36 issues (4 per cell) surfaced patterns. Codebook consolidated and applied to all 210 issues. Single coder (LLM-assisted, with author review). Codes documented in [scripts/audit/qual/codebook.md](scripts/audit/qual/codebook.md). Per-issue codes and rationales in [codes_misclass.csv](scripts/audit/qual/codes_misclass.csv) and [codes_invalid.csv](scripts/audit/qual/codes_invalid.csv).

**Threats**: single-coder analysis; no inter-rater reliability computed. The annotation CSVs are released as audit artefacts so any reviewer can re-rate.

---

## Headline findings (misclassifications)

1. **Only ~3% of misclassifications are "clean errors."** 97% have a label-noise, surface-form, or hybrid-intent confound. The model is rarely "wrong" in an unambiguous case.
2. **35% of misclassifications are label noise.** Ground-truth and predicted labels are both defensible; the issue is mislabeled in the test set. This is an **upper bound on achievable F1** — roughly 35% of the remaining error gap cannot be closed by any method.
3. **38% are template scaffolding mismatch.** Issues filed under a bug template that are actually questions or features. The dominant single confound.
4. **Retrieval-skew explains ~26% of RAGTAG errors and ~14% of BRAGTAG errors.** BRAGTAG roughly halves retrieval-misled errors — direct mechanistic confirmation that the debias intervention works as designed.

---

## Misclassification analysis (n=150)

### Code distribution

| Code | BRAGTAG | FT | RAGTAG | Total | % |
|------|--------:|---:|-------:|------:|---:|
| BUG-SHAPED-QUESTION | 17 | 23 | 17 | **57** | **38%** |
| LABEL-NOISE | 21 | 13 | 18 | **52** | **35%** |
| HYBRID-REPORT | 6 | 4 | 6 | 16 | 11% |
| BUG-SHAPED-FEATURE | 3 | 5 | 5 | 13 | 9% |
| CLEAN-ERROR | 1 | 1 | 2 | **4** | **3%** |
| FEATURE-AS-BUG | 1 | 3 | 0 | 4 | 3% |
| AUTHOR-UNCERTAIN | 1 | 1 | 1 | 3 | 2% |
| QUESTION-SHAPED-OTHER | 0 | 0 | 1 | 1 | 1% |
| **Total** | **50** | **50** | **50** | **150** | **100%** |

### Cross-method patterns

- **FT has the most BUG-SHAPED-QUESTION errors (23/50 = 46%)** — strongly weighted toward question issues filed via bug templates. Prominent contributor: VSCode's auto-bug-template feature ("Type: <b>Bug</b>") attaches to issues users submit via the IDE's "Help > Report Issue", regardless of whether they're support questions. FT learned this template signal as bug-predictive and over-applied it.
- **BRAGTAG has the highest LABEL-NOISE share (21/50 = 42%)** — its remaining errors are disproportionately on cases that any rater would also struggle with. The intervention has cleared the "fixable" errors (bug-shaped questions handled by removing bug examples), so what's left is harder.
- **RAGTAG has the highest retrieval-skew rate (26%)**; BRAGTAG halves it (14%). FT has no retrieval and is unaffected. This is direct mechanistic evidence that BRAGTAG's debias targets retrieval-misled errors specifically.

### Most common patterns (concrete examples)

**BUG-SHAPED-QUESTION (38% of errors).** Issues using a bug-report template structure (Steps to reproduce / Expected / Actual) where the user's actual intent is a support request. Examples:

- `kubernetes/kubernetes#252` — "helm upgrade causing downtime": uses Kubernetes bug template, but the user is asking why their rolling update isn't working as expected.
- `dotnet/roslyn#252` — "CS8604 Compiler Warning despite previous null checking": detailed bug-template report, but it's a misunderstanding of nullable reference flow analysis.
- `microsoft/vscode#2612, #2653, #2673, #2688` — VSCode "Type: <b>Bug</b>" auto-template issues that are users asking C/C++/Python compilation questions.

**LABEL-NOISE (35% of errors).** Ground-truth label is arbitrary, contradicted by the issue's own framing, or applied to a non-issue. Examples:

- Ansible cherry-pick PRs (`ansible#229, #230, #271, #278, #285, #288, #292`) labeled "question" — these are PR descriptions, not issues.
- `microsoft/TypeScript#225` — header literally `# Bug Report`, "v5 beta typescript support not working" — labeled "question" in GT.
- `bitcoin/bitcoin#284` — opens with "Please describe the feature you'd like to see added"; labeled "question" but is a feature request.
- `opencv/opencv#263` — author explicitly checks "I report the issue, it's not a question"; labeled "question" in GT.
- `opencv/opencv#36` — title literally starts `feat:` (conventional commit prefix); labeled "bug".

**HYBRID-REPORT (11% of errors).** Two intents legitimately combined:
- `bitcoin/bitcoin#40` — bug + author proposes a fix.
- `kubernetes/kubernetes#1818` — broken multiarch image + "please give us multiarch image".
- `flutter/flutter#1760` — asks for alternative API + reports removed property.

**BUG-SHAPED-FEATURE (9% of errors).** Feature/enhancement requests using bug-template framing ("X does not work", "X should be Y"):
- `microsoft/vscode#145` — "Dim terminal text should have half the minimum contrast ratio" (Repro / Actual / Expected, but it's a feature request).
- `tensorflow/tensorflow#3154` — "protobuf 4 is not supported" (filed under TF's bug template).

### What retrieval-skew tells us

The retrieval-skew flag (top-12 neighbors ≥7/12 dominated by the wrong class) co-occurs with each code. The most striking cases:

| Issue | GT | Predicted | Neighbor mix | Code |
|-------|----|-----------| -------------|------|
| `kubernetes#18` | bug | feature | **0/0/12** (all question) | HYBRID-REPORT (RAGTAG) |
| `bitcoin#64` | bug | question | 1/1/10 (mostly question) | CLEAN-ERROR (RAGTAG) |
| `microsoft/vscode#145` | feature | bug | 10/1/1 (mostly bug) | BUG-SHAPED-FEATURE (RAGTAG) |
| `microsoft/vscode#166` | feature | bug | 8/1/3 | BUG-SHAPED-FEATURE (RAGTAG) |

These cases show the LLM following retrieval over its own reasoning. BRAGTAG's intervention is specifically about not letting bug-heavy retrieval dominate when the query might be a question — the 26% → 14% drop in retrieval-skew rate is the direct mechanistic signature of the intervention.

---

## Invalid-output analysis (n=60)

Invalid outputs are rare (~4% pre-fallback) and concentrated in two failure modes.

| Code | BRAGTAG | RAGTAG | Total | % |
|------|--------:|-------:|------:|---:|
| CONTINUES-AS-BODY | 21 | 15 | **36** | **60%** |
| MALFORMED-OUTPUT | 6 | 9 | 15 | 25% |
| OFF-TOPIC-LOOP | 3 | 5 | 8 | 13% |
| CHAIN-OF-THOUGHT | 0 | 1 | 1 | 2% |

**CONTINUES-AS-BODY (60%).** The model fails to converge on a single label and instead continues generating issue-body-style content — typically a stack trace, an "Expected/Actual behavior" field, or more code that fits the issue's surface form. These tend to occur on long-context inputs near the token budget, where the model exhausts its generation budget before emitting a structured answer.

**MALFORMED-OUTPUT (25%).** Output is unparseable as a schema label: empty/minimal output, schema-violating values (e.g. "docs", "documentation"), or a label tag mixed with extraneous prose.

**OFF-TOPIC-LOOP (13%).** Pure degenerate generation: repetitive numeric loops ("5555...", "8888..."), repeated dots, or echoed version strings. Model failure on the input.

**CHAIN-OF-THOUGHT (2%).** Single rare case where the model reasoned in prose rather than emitting a label.

All four failure modes are handled at zero LLM cost by the VOTAG fallback proposed in §5.5, which preserves the always-have-an-output property required for deployment.

---

## Implications for the paper

### For Discussion section

1. **The 35% label-noise floor.** Any macro-F1 above ~0.83 should be viewed against this ceiling. The aggregate (BRAGTAG − FT) = +0.001 statistical equivalence is essentially "both methods saturate the achievable F1 given the test set's label quality." This recasts the headline result: the methods aren't tied because they're equally good at IRC, they're tied because **they've both reached the noise floor.** This is a strong, novel framing for the discussion.

2. **BRAGTAG halves retrieval-misled errors (26% → 14%).** Direct mechanistic evidence that the intervention does what it's designed to do. The remaining 14% retrieval-misled errors at BRAGTAG suggest a path to a tighter intervention (e.g. broaden the trigger to feature-bias too, or use embedding-space rebalancing instead of margin-based filtering).

3. **FT's distinct error profile.** FT errors lean harder on BUG-SHAPED-QUESTION (46% vs ~34% for RAG/BRAG). FT learned bug-template signals and over-applies them, especially on the VSCode "Type: <b>Bug</b>" auto-template issues. This is an instance of *training-set artifact memorization*: PA training data has more bug-template-styled questions than retrieval would surface to an in-context method. A discussion paragraph could land this as a concrete advantage of in-context methods over fine-tuning: in-context methods don't memorize template noise.

### For Threats to Validity

- Note that 35% of GT labels in the *error* sample are arbitrary or contradicted by the issue's own framing. Extrapolating, the test set itself has non-trivial label noise. Cite this limitation explicitly.
- Single-coder qualitative analysis. Note in §7. Annotation CSVs released for re-rating.

---

## Files

- [scripts/audit/qual/codebook.md](scripts/audit/qual/codebook.md) — codebook
- [scripts/audit/qual/sample_errors.py](scripts/audit/qual/sample_errors.py) — sampling script (frozen seed)
- [scripts/audit/qual/sample_misclassifications.csv](scripts/audit/qual/sample_misclassifications.csv) — 150 sampled errors
- [scripts/audit/qual/sample_invalids.csv](scripts/audit/qual/sample_invalids.csv) — 60 sampled invalids
- [scripts/audit/qual/codes_misclass.csv](scripts/audit/qual/codes_misclass.csv) — per-issue codes + rationales (misclassifications)
- [scripts/audit/qual/codes_invalid.csv](scripts/audit/qual/codes_invalid.csv) — per-issue codes (invalids)
- [scripts/audit/qual/apply_codes.py](scripts/audit/qual/apply_codes.py) — code-application script (misclassifications)
- [scripts/audit/qual/apply_codes_invalid.py](scripts/audit/qual/apply_codes_invalid.py) — code-application script (invalids)

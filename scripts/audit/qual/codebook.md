# Open-Coding Codebook — Qwen-32B Misclassification + Invalid Analysis

Codebook developed inductively from a 36-issue pilot batch (4 examples × 3 methods × 3 ground-truth labels). Stable codes were consolidated after the pilot and applied to the full 150-issue misclassification sample and 60-issue invalid sample.

## Misclassification codes (one per issue, mutually exclusive)

| Code | Definition | Decision rule |
|------|------------|---------------|
| **LABEL-NOISE** | The ground-truth label is arbitrary, contradicted by the issue's own framing, or applied to a non-issue (PR/commit description). The predicted label is at least as defensible. | (a) author explicitly classifies as a different label, OR (b) issue uses a template/scaffolding for the predicted label, OR (c) it's a PR/cherry-pick description. |
| **BUG-SHAPED-QUESTION** | Genuine support request that uses bug-report scaffolding (steps to reproduce / expected / actual). Author seeks understanding rather than reporting confirmed defect. | True-question, with explicit bug-template structure but the *intent* is "why doesn't this work?" / "is this expected?". |
| **BUG-SHAPED-FEATURE** | Genuine feature/enhancement request phrased as "X does not work" or "X should do Y", using bug-template structure. | True-feature, with bug-style framing about a missing/limited capability. |
| **QUESTION-SHAPED-OTHER** | True bug or feature where the body's tone is information-seeking ("Is there a way to...?", "What is the future of...?"), making question prediction defensible. | True-bug or true-feature, but body tone reads more like a support thread. |
| **HYBRID-REPORT** | Issue legitimately combines two intents (e.g., bug report + proposed fix; question + feature suggestion); ambiguity is genuine. | Two distinct intents are both clearly present and either could justify the GT label. |
| **AUTHOR-UNCERTAIN** | Author explicitly hedges ("I'm not sure if this is a bug...", "this seems a bug but also future request"). | Explicit uncertainty in the body. |
| **CLEAN-ERROR** | None of the above: surface form clearly matches GT, no hedging, no template confound. The model just got it wrong. Indicates parametric bias in FT or post-retrieval reasoning failure in RAG/BRAG. | All other rules fail. |

## Overlay (RAG/BRAG only, recorded as a binary flag)

| Flag | Definition |
|------|------------|
| **retrieval-skew** | The top-12 neighbor labels are dominated (≥7/12) by the wrong class. Co-occurs with any code above. Helps separate "retrieval pulled the LLM" from "LLM ignored retrieval". |

## Invalid-output codes (one per invalid prediction)

| Code | Definition |
|------|------------|
| **FORMAT-XML** | Model produced text but the `<label>...</label>` wrapper was malformed (missing closing tag, wrong nesting, multiple labels). |
| **HEDGED** | Output contains "either", "or", multiple candidate labels, or qualifiers like "possibly". |
| **ECHOED-NEIGHBOR** | Model copied a neighbor example or its content instead of producing a label. |
| **TRUNCATION** | Generation cut off mid-token; `generated_tokens` close to the cap. |
| **OFF-TOPIC** | Model generated unrelated text (continuation of body, refusal, repetition). |
| **EMPTY** | No parseable content after the `<label>` prefill. |
| **OTHER** | None of the above. |

## Notes on application

- Single coder (LLM-assisted, with author review). Limitation noted in §7 Threats.
- Decisions are recorded per (method, project, test_idx) with a one-line rationale.
- Sample is frozen via `scripts/audit/qual/sample_errors.py` (seed=20260509).
- Codes are mutually exclusive *within* a category. The retrieval-skew flag is independent.

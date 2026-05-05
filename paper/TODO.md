# Paper TODO

Items to circle back to. Not blocking the current writing thread.

## During / after evaluation sections (§5–§7)

- [ ] **Significance testing — DECIDED: add Bootstrap CIs + McNemar's.** Run after the 3-epoch FT campaign finishes (NRP + local both done, all 48 cells synced and evaluated).

  **What to compute:**
  - **Bootstrap 95% CIs** on macro F1 for every method × model × setting cell. 1,000 resamples per cell. Report alongside the F1 point estimate in the leaderboard.
  - **McNemar's test** for the headline pairwise comparisons:
    - Debiased RAGTAG_PS vs FT_PS (per project + aggregated, per model)
    - Debiased RAGTAG_PS vs FT_PA per-project eval (per project + aggregated, per model)
    - RAGTAG_PA vs FT_PA (per model)

  **Why both:** the expected outcome is "RAGTAG slightly below FT; Debiased RAGTAG closes most of the gap, sometimes matching." Both the equivalence claims (matches) and the small-gap claims need stats:
  - Bootstrap CIs put error bars on each metric so reviewers can see where confidence intervals overlap.
  - McNemar's settles whether a small gap or apparent tie is real. Required to defensibly claim "matches" (= p > 0.05 of difference).

  **Implementation:** ~50 lines of Python in `scripts/analysis/significance_tests.py`. Use `statsmodels.stats.contingency_tables.mcnemar` for the test and `numpy` resampling for CIs. Need `y_true` and `y_pred` arrays per method × project to feed both.

  **Documentation in §4.4:** once tests are run, document the methodology there (1,000 bootstrap samples; McNemar's with continuity correction; p < 0.05 threshold; report b vs c counts in supplementary).

- [ ] **Hardware specs in §4.5.** Need exact server specs. Run `kubectl describe node <node-name>` on the cluster (or `kubectl describe pod <pod>` for the running mega-runner) to capture: GPU model + VRAM, CPU model + core count, RAM, OS/CUDA versions. Fill in §4.5 placeholder once known.

## RQ3 — method pivot in progress (2026-05-04)

- [ ] **Pick the new RQ3 context-engineering method.** The originally-planned RQ3 method (margin-based retrieval debiasing, m=3) is being demoted from the headline. Reason: after the 3-epoch FT campaign tightened the FT baseline, margin-debiasing's gains shrank. The user's framing: "debiased RAGTAG basically assumes the few-shot prompt already has 3 bugs in the chamber before we do anything" — i.e. the method only manipulates the *prompt's class distribution*, not the model's bug-prior.

  **Constraints on the replacement** (locked, do not re-litigate):
  - Genre must be **context engineering** (curate/augment/restructure what enters the context window).
  - NOT prompt engineering (instructions, persona, schema, decision trees, CoT, rubrics).
  - NOT output-side / logit math (calibration, PMI, threshold tuning, activation steering — already tried or excluded).
  - NOT class-rebalancing of the few-shot pool (same family as current method; user is tired of it).
  - NOT retrieval-side replumbing (different embedder, BM25 hybrid, cross-encoder rerank).

  **Status:** Gemini brainstorm prompt drafted in the prior session and given to the user. Awaiting Gemini response and user selection. See [SESSION_HANDOFF.md §4](SESSION_HANDOFF.md) for the full constraint list.

- [ ] **Decide what happens to margin-debiasing in the paper.** Options: (a) keep as the headline RQ3 method if the new method doesn't pan out; (b) demote to an early-attempt baseline that the new method improves over; (c) drop entirely. Hold this decision until the new method is chosen and run.

## After §7 evaluation section is drafted

- [ ] **Add debiasing forward-pointer to §3.3 RAGTAG.** Insert after the existing RAGTAG content:
  > *We additionally introduce a retrieval-debiasing intervention applied on top of \ragtag; we describe its algorithm and present its empirical motivation in \cref{sec:rq3}.*

  Replace `\cref{sec:rq3}` with the actual subsection label once §7.1 is written (likely `sec:debias` or `sec:rq3-debias`).

  **Why deferred:** debiasing's algorithm and motivation live in §7.1 alongside its empirical evidence (Option B in the framing discussion). The §3.3 sentence is a one-line signpost so methodologically-conservative reviewers don't wonder where the third contribution went.

  **2026-05-04 caveat:** if the RQ3 method gets replaced (see above), this forward-pointer's wording needs to match whatever lands in §7.1. Revisit phrasing after the method is chosen.

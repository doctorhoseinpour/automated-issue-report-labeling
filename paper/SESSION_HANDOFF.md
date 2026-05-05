# Session Handoff — 2026-05-04

Single source of truth for the next session. Read this first.

The paper is being written; experiments are partially in flight. This doc captures the current state of both threads. Earlier handoff docs (`docs/ANALYSIS_HANDOFF.md`, `docs/NRP_MIGRATION_STATUS.md`) are historical and partially superseded — see the "Superseded numbers" section below.

---

## 1. Where we are

**Paper:** ESEM 2026, LIPIcs template, double-blind. Working title: *"RAGTAG: When Does Retrieval-Augmented Few-Shot Classification Match Fine-Tuning for GitHub Issue Triage?"* Internal draft deadline 2026-05-08, conference deadline 2026-05-18.

**Sections drafted:** [§3 Approach](sections/03_approach.tex) and [§4 Experimental Setup](sections/04_setup.tex). Both reviewed and corrected. §3.5 covers fine-tuning; §3 omits a debiasing subsection by design (the algorithm is trivial; the substantive content lives in §7 RQ3).

**Sections not yet drafted:** §1 Intro, §2 Related Work, §5 RQ1, §6 RQ2, §7 RQ3, §8 Discussion, §9 Threats, §10 Conclusion.

**Held back from §4:** [§4.5 Hardware](sections/04_setup.tex#L37-L41) is a placeholder pending kubectl describe on the NRP node. [§4.4 Significance testing methodology](TODO.md) deferred until the FT campaign finishes — bootstrap CIs + McNemar's planned, not yet wired.

---

## 2. The 3-epoch fine-tuning campaign (in flight)

Prior FT results were trained for 1 epoch. Heo et al. and Aracena et al. both default to 3. A 1-epoch→3-epoch calibration on Qwen-3B PA showed +0.05 macro F1, confirming under-training. The full FT grid (4 models × 12 cells = 48 cells) is being re-run with `num_train_epochs=3`. Code change in [fixed_fine-tune.py:400](../fixed_fine-tune.py#L400), committed as SHA `6570030`.

**Backups in place** (do not lose):
- Local 1-epoch FT data: `results/issues11k_ft_1epoch_backup_20260504/` (~12 GB, 72 eval CSVs).
- NRP-side 1-epoch FT data: PVC `/data/_backup_1epoch_20260504/` (24 dirs).

**Status:**

| Track | Models | Cells | Status |
|---|---|---|---|
| Local (RTX 4090) | Qwen-3B + Qwen-7B | 24/24 | **Done.** Predictions and eval CSVs all present. |
| NRP (mega-runner) | Qwen-14B + Qwen-32B | 0/24 | Pod `mega-runner-p4v9d` **Pending** since ~13:00, A6000 contention. |

**Image:** `ghcr.io/doctorhoseinpour/llm-labler:7ecfc75` (SHA-pinned, includes the 3-epoch change *and* the wave-filter so only FT waves [0, 2, 5] run). Verification protocol from the plan was followed: rebuild → cluster verify-pod confirms `=3` → push → update `plan.yaml` SHA.

**Wave layout (mega-runner):** waves 2 (Qwen-14B FT, 12 cells) and 5 (Qwen-32B FT, 12 cells). Wave 0 is a no-op canary.

**When NRP completes:** `bash scripts/nrp/sync.sh` pulls tarballs to `results/issues11k/`, then run `evaluate.py` over each `preds_finetune_fixed.csv` (the FT script writes preds but not evals). The previous analysis pipeline (`scripts/analysis/*.py` writing into `docs/analysis/`) has been retired along with its outputs — the new analysis approach is TBD as we shape the narrative.

---

## 3. Partial leaderboard (current data)

Computed 2026-05-04 after local FT completed and the per-project cache was rebuilt. NaN = waiting on NRP.

| Model | RAGTAG_PS | **Debias_PS** | FT_PS | FT_PA (per-proj) |
|---|---:|---:|---:|---:|
| Qwen-3B (FT 3ep) | 0.694 | 0.709 | 0.665 | 0.704 |
| Qwen-7B (FT 3ep) | 0.714 | 0.730 | 0.677 | **0.759** |
| Qwen-14B (FT —) | 0.717 | 0.742 | — | — |
| Qwen-32B (FT —) | 0.758 | 0.775 | — | — |

**The robust headline that survives all NRP outcomes:**
> Debiased RAGTAG_PS matches or beats fine-tuning's project-specific counterpart at every model scale, without any training.

Locked-in: +0.044 at 3B, +0.053 at 7B. 14B/32B FT_PS pending.

**The fragile claim (the FT_PA-per-proj column):**
- 3B: Debias_PS edges out by +0.005 — essentially tied
- 7B: Debias_PS *loses* to FT_PA per-proj by 0.029
- 14B/32B blocked

The 7B FT_PA result is the warning shot. If 14B/32B FT_PA show similar 3-epoch lifts, the "matches FT_PA" claim collapses everywhere. Do not commit to this claim until NRP lands.

---

## 4. Strategic pivot — RQ3 method needs to change

The user's read after seeing the 3-epoch FT numbers: **the current margin-debiasing method is not adding much**. It plateaus at +0.02 macro F1, and the gain shrinks against properly-trained FT.

The user has framed the problem precisely:

> "The models are biased towards bug, so we need an approach that debiases the model without taking away from the bug performance. Right now debiased RAGTAG basically assumes that the few-shot prompt already has 3 bugs in the chamber before we do anything."

i.e. current debias acts on the **prompt's class distribution** (yank a bug example out). It does not address the model's underlying bug-prior.

**Constraints on the next intervention** (locked by the user, do not re-litigate):

| Genre | Status | Reason |
|---|---|---|
| **Prompt engineering** (instructions, persona, schema, decision trees, CoT, rubrics) | Out | User explicitly excluded |
| **Output-side / logit / math** (contextual calibration, PMI, threshold tuning, activation steering) | Out | Activation steering already failed (CAA best F1 0.7063); rest excluded by user |
| **Retrieval-side replumbing** (different embedder, BM25 hybrid, cross-encoder rerank) | Out | Out of scope this paper |
| **Fine-tuning / DPO / RLHF / distillation** | Out | Inference-time only |
| **Class rebalancing of the few-shot pool** (top-K-per-class, vote-prior injection) | Out | Same family as current debias; user is tired of it |
| **Context engineering** | **In** | The genre the user wants — curate/augment what enters the context window in ways that do *not* simply rebalance class composition |

The user defines context engineering by analogy to the current method:

> "We are ENGINEERING THE CONTEXT in a way that steers the LLM towards the good output. The current debias decides which neighbors land in the context. We want more ideas of that genre."

Examples of in-scope moves (not exhaustive):
- New selection criteria for which neighbors to include (beyond top-K cosine + margin).
- Auxiliary in-context content that is not a retrieved neighbor — synthetic prototypes, anchor exemplars, summary statistics over retrievals, counterfactual versions of retrieved items.
- Composition of the retrieved set — diversification, hard-negative mining, cross-class contrastive pairing, error-driven selection.
- Metadata alongside neighbors that the model conditions on — provenance markers, similarity scores, vote tallies, "this was almost classified as Y" signals.

A Gemini brainstorm prompt was drafted in the prior session (last assistant turn before this handoff). It contains the full context-engineering vocabulary section, the bug-bias diagnosis, and the explicit forbidden list. **The user is going to feed it to Gemini and bring the response back.** No method has been chosen yet.

---

## 5. Outstanding TODOs

- **Wait on NRP mega-runner.** Pod is Pending. When it schedules, ~1.5–2.5 days wall.
- **Sync + evaluate** when NRP cells land. Then design the new analysis pipeline (the old `docs/analysis/` was deleted; we are not committed to its scripts or output schema).
- **Pick the new RQ3 context-engineering method** once the Gemini brainstorm comes back.
- **Bootstrap CIs + McNemar's tests** ([TODO.md](TODO.md)). Run after all 48 cells land. ~50 lines in `scripts/analysis/significance_tests.py`.
- **Hardware specs in §4.5** ([TODO.md](TODO.md)). Run `kubectl describe node <node>` once the mega-runner pod is scheduled.
- **§3.3 forward-pointer to debiasing** ([TODO.md](TODO.md)) — insert one-line cref to §7.1 once §7 is written. Note: this assumes the chosen RQ3 method goes in §7.1; if the method is novel enough to warrant its own §3.x subsection, revisit.
- **Sections to draft** in order: §5 RQ1 → §6 RQ2 → §7 RQ3 → §8/§9/§10 closing → §1 Intro → §2 Related Work.

---

## 6. Superseded numbers (what to discard from the older docs)

These appear in older handoff docs and are now wrong or partial. Trust this section over the old tables until those docs are rewritten.

- **`docs/ANALYSIS_HANDOFF.md` §5.1 leaderboard, §5.3 FT-vs-debias deltas, §5.5 FT-PS collapse table:** all use **1-epoch FT** numbers. The 3B and 7B FT rows are now superseded by the 3-epoch results in §3 of this doc. The 14B/32B FT rows are about to be superseded as well.
- **`docs/NRP_MIGRATION_STATUS.md` "live progress" and "wave queue":** describe the 2026-04-27 1-epoch campaign which has long since completed. Image SHA there (`:55ba8f1`) is also stale; current is `:7ecfc75`. The "Bugs caught + fixed during the migration" list in §4 is still institutional knowledge worth preserving.
- **`docs/PAPER_NARRATIVE.md` lines 1–180:** pre-Qwen-only refresh. Active model lineup is wrong (Llama listed as primary), Llama-8B numbers cited (paper is now Qwen-only). The "Refresh 2026-04-29" section starting at line 181 is canonical.

---

## 7. Files and what they're for

```
paper/
  main.tex                  -- LIPIcs root, anonymous mode on, booktabs loaded
  refs.bib                  -- user-managed
  sections/03_approach.tex  -- DONE
  sections/04_setup.tex     -- DONE (minus §4.5 hardware)
  sections/{01,02,05-10}.tex -- empty/stub
  TODO.md                   -- deferred items
  NARRATIVE_NOTES.md        -- intro-writing reference (Reasoning X, three-roles VOTAG, Option A)
  REVIEW_PROMPT.md          -- adversarial-reviewer template (planned, may not exist yet)
  SESSION_HANDOFF.md        -- this file

docs/
  11K_BENCHMARK_FINDINGS.md -- old findings doc; internal analysis/ links are now broken (analysis dir deleted)
  PAPER_NARRATIVE.md        -- discussion-thread reference; pre-2026-04-29 sections trimmed
  NRP_MIGRATION_STATUS.md   -- steady-state setup + bug history (live status removed)
  ANALYSIS_HANDOFF.md       -- 2026-04-29 handoff; FT numbers superseded, retrieval numbers still valid
  *.pdf                     -- reference papers (Heo, Aracena, CAA, NoTrainingWheels)

results/
  issues11k/                -- paper-archival, never delete
  issues11k_ft_1epoch_backup_20260504/  -- 1-epoch FT backup, also archival

scripts/nrp/
  plan.yaml                 -- waves filtered to [0, 2, 5]; image SHA :7ecfc75
  runners/run_remaining_cells.py
  sync.sh                   -- pulls _outbox/ tarballs to local results/

run_3epoch_full_campaign.sh -- local 3B/7B campaign script (idempotent skip on preds)
fixed_fine-tune.py:400      -- num_train_epochs=3
```

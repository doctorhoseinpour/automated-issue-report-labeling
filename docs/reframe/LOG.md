# Reframe log

Every session appends one entry at the bottom, in this format:

```
## S<n>: <short title> (<date, time>)
- Changed: <file> → <one line>, ...
- Page state: <pagecheck.py output, the key lines>
- Numbers: <anything about NUMBERS.md, anchors, disagreements>
- Open questions for the user: <list, or "none">
- Notes for the next session: <list>
Status: DONE | PARTIAL (<why>)
```

## S0: setup (2026-09-25)

- Preserved the draft: commits `c6cd613` (SANER2027 fixes) and `b93deee` (follow-up study notes and
  scripts) on `encoder-baselines`, pushed; tag `saner-draft-2026-09-25` pushed.
- Created this worktree on branch `saner-reframe` from the tag, pushed with upstream.
- Wrote `docs/reframe/` (README, BRIEF, S1–S6 prompts, this LOG, `pagecheck.py`).
- Page state at the tag: 12 pages. The main text ends on page 10, right column, with 9 free lines; the
  references start on page 10 and end on page 12 (page 12 holds only the last entry). A test build with
  `\clearpage` before `\balance` puts the references at the top of page 11 and ending on page 12 with
  room to spare (about 100 body-line equivalents). S1 adds that `\clearpage`.
- Known build warnings at the tag: two `Overfull \vbox ... while \output is active` from `\balance` on
  the reference pages (harmless).
- User decisions: names $k$NN voting / RAG / filtered RAG; percent with one decimal and points; three
  RQs in build-up order; exactly 10 full pages of main text with references from page 11; work on the
  paper's text only, with no checks against the implementation.
- Open questions for the user: De Vito et al. (TOSEM 2026), to be revisited by S2; the heatmap's 0.000
  cell (lab machine).
Status: DONE

## S1: names, number format, draft colors, tables and figures (2026-09-25, 16:30)

- Diagram: exported automatically (drawio-desktop 31.5.2 CLI); **no manual step needed**.
- Changed:
  - `SANER2027/SANER2027-main-labeling.tex` → `\clearpage` before `\balance`; draft-color macros removed
    (`\fixed`, `\coauthor`, `\coauthornote`, `coauthorcolor`, `coauthoranswer`, `\change`, `\rev`,
    `\minor`); `\votag/\ragtag/\bragtag` replaced by `\knn`, `\rag`, `\frag`, `\Frag` (`\fixme` kept).
  - `SANER2027/sections/*.tex` → colors unwrapped (answer boxes are plain `myboxi`); new names
    everywhere; old acronym expansions rewritten (intro, approach subsections and captions, end of old
    RQ2, abstract, conclusion); "retrieval-based methods/approaches" and "RAG-based methods" now mean
    "RAG variants" / "RAG and filtered RAG"; "LoRA's" / "for LoRA" → "fine-tuning's" / "for
    fine-tuning"; every score in percent, every difference in points; §III-C gains "We report scores in
    percent and differences between methods in percentage points (points)."
  - `scripts/paper/tab_results_master.py`, `tab_method_comparison_ci.py` → no color; new row/column
    labels; values in % (Table I) and points (Table II); captions reproduce the committed
    hand-shortened captions (the generators had drifted from the committed .tex). Table I bolds the
    best macro F1 per block by printed value (no ties occur).
  - `SANER2027/tables/method_cost.tex` → no color; rows `$k$NN voting`, `RAG`, `Filtered RAG`,
    `Fine-tuning`.
  - `scripts/paper/fig_kcurves.py`, `fig_per_project_diff.py` → new names, y axis / cells in % and
    points, integer colorbar ticks; figures regenerated (`kcurves.*`, `per_project_diff.*`).
  - `SANER2027/figures/src/fao.drawio` (new; extracted from the PDF's Subject metadata) and
    `SANER2027/figures/fao.pdf` → labels "A. kNN voting: retrieval-only baseline", "B. RAG:
    retrieval-augmented few-shot prompting", "C. Filtered RAG", "Retrieval-Based Classifiers" (that
    borderless title box widened symmetrically so the longer title fits on one line; C header
    left-aligned to match A and B). Exported with `--crop --scale 2 --embed-diagram`; a re-export of
    the unmodified source matched the old PDF.
  - `scripts/paper/numbers_sheet.py` (new) → writes `docs/reframe/NUMBERS.md`.
- Page state (`pagecheck.py`): main text ends page 10, right column, y=633; **7 free lines** on page 10
  (was 9; the new §III sentence costs 2). References start page 11, left column; end page 12 with about
  100 body-line equivalents of room. OK within 10 + 2. Build: only the known `\balance` overfull vbox
  (one, not two, in this build), no undefined references.
- Color removal check: the PDF's word sequence is identical before and after. One line moved across the
  page 1/2 break (the RQ2 run-in now starts page 2), because the `\color` whatsits at paragraph starts had
  shifted the page break. Pages 3–12 are identical.
- Numbers:
  - All BRIEF §9 anchors agree with NUMBERS.md.
  - Disagreement with the draft: "Feature $F_1$ changes by at most 0.012" (RQ3 text and answer box). The
    exact maximum is 1.15 points (Qwen-7B), so the text now says "at most 1.1 points".
  - Table II values changed from 3 d.p. to points with one decimal, and bold marks are unchanged.
  - Heatmap: the ansible/Qwen-3B cell is exactly 0.000 in the hand-seeded CSV. The figure prints "0.0",
    and it counts toward "16 of 44 pairs match or exceed". Its sign stays unknown until the lab machine
    regenerates `per_project_diff.csv` (`--from-results`). NUMBERS.md §6 notes it.
  - Values marked "from the draft" in NUMBERS.md: calibration (−1.36, 89.1%, 56.7%, 95th pct = 5), 40%/18%
    empty prompts, the per-model filtered-RAG-vs-RAG CIs, neighbor overlap 84–90%, and the failure
    analysis counts.
- Open questions for the user:
  - The failure-analysis shares still print two decimals (77.33/13.33/9.33%). They are shares, so S1
    left them unchanged. Whole percents (77/13/9%) sum to 99%; 116/150, 20/150 and 14/150 are exact.
    S3 or the user should decide.
  - The heatmap's 0.000 cell (lab machine), as before.
- Notes for S3:
  - Sentences the rename made awkward: intro "We label the query by a similarity-weighted vote over these
    neighbors, without an LLM ($k$NN voting)"; RQ4 question "How do the two RAG variants compare…"; the
    §II intro "the $k$NN voting baseline and our two retrieval-augmented methods"; §II-D now opens "RAG
    shows the query's top-$k$ neighbors…" (the tautological "RAG applies retrieval-augmented few-shot
    prompting" sentence was dropped); Discussion "The RAG variants offer deployment advantages…".
  - Signed numbers remain only in the prose CIs (§IV RQ3 per-model CIs, aggregate CIs in RQ4) and in
    the calibration mean −1.36. The prose CIs are in points now, ready to move to Table II / a footnote.
  - "data scope(s)" still appears (§III-A, §IV RQ2/RQ4 text) and should become "setting(s)" (BRIEF 4.1).
  - The Table II caption still says "(paired bootstrap 95% CI)". BRIEF 4.2 wants the bootstrap details
    in one text footnote only; two footnotes now carry them (RQ3 and RQ4).
  - Legacy, un-input tables (`tables/method_comparison*.tex`, `bragtag_results*.tex`,
    `vtag_peak.tex`) and their generators still use `\votag/\ragtag/\bragtag`. They are not in the
    paper and would break the build if someone `\input` them again.
  - Run the generators with `/usr/bin/python3`: the `python3` on PATH is a virtualenv without pandas.
  - `numbers_sheet.py` parses `method_cost.tex` by its row labels (RAG / Filtered RAG / Fine-tuning).
    Keep them if you edit that table.
Status: DONE

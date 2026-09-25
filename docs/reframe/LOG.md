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

## S2: Related Work research dossier (2026-09-25, evening)
- Changed:
  - `docs/reframe/RW_DOSSIER.md` → new: novelty risks, De Vito et al., accuracy fixes (A1–A6), refs.bib corrections, per-paper facts with page-cited quotes and "how we differ" lines, a 6-paragraph outline with exact contrast sentences (~1,000 words vs ~690 now), open questions.
  - `docs/reframe/refs_candidates.bib` → new: 15 verified entries not in refs.bib (sources in comment lines; test-compiled with IEEEtran, no warnings; no key collisions).
  - No edits under `SANER2027/`, `scripts/` or `paper/`. Commit `5bf6860`.
- Page state: S2 did not build. `pagecheck.py` on the PDF currently on disk (built from another session's uncommitted edits) reports that the main text ends on page 8 with 226 free lines before the end of page 10. That is a transient state, not S2's.
- Numbers: none computed. The outline's contrast sentences use only BRIEF §9 anchors (1.0 point behind, CI includes zero; 46–59%); S4 re-checks them against NUMBERS.md.
- **NOVELTY RISK (S4: read the top of the dossier before writing):**
  - The novelty sentence in §I (and the live abstract) says RAG "has not been systematically evaluated for IRC or compared with LoRA fine-tuning".
  - The second half survived every search. The first half is at risk from De Vito et al. (TOSEM 2026, few-shot LLM IRC with Qwen2.5-32B; full text closed, example selection unknown; their 2024 plan used vector-DB similarity selection) and from LabelMate (arXiv 2026-09, retrieved labeled issues for fine-grained labels, no fine-tuning).
  - Recommended narrowing: "... has not been compared with LoRA fine-tuning of the same LLMs on the same labeled issues."
- Open questions for the user:
  1. Cite De Vito et al. (S2 recommends yes, with a cautious contrast sentence; please check its §3 if you have ACM access).
  2. Approve the narrowed novelty sentence (§I and abstract).
  3. Cite LabelMate (preprint) or not.
  4. The NASA study (`colavito2026issue`): one clause recommended. Its SetFit result (fewer than 20 labeled examples beat zero-shot LLMs) could be raised against the "11× less labeled data" headline.
- Notes for the next sessions:
  - S4:
    - The current Related Work misstates prior work in four places:
      - A1 (the most serious): "state-of-the-art ... fine-tune with LoRA". The best results come from GPT models fine-tuned through OpenAI's API; LoRA was used only for open models, which scored lower.
      - A2: `yu2023retrieval` is a trained RoBERTa classifier with a trained retriever, not in-context examples.
      - A3: `le2023log` is prompt-tuning of RoBERTa, `logan2021` argues for fine-tuning, and `ma2023fairness` is about label bias. None supports "few-shot prompting helps".
      - A4: "encoders need large data" is contradicted by Colavito IST 2025.
    - Fixed wording is in the dossier.
    - `hu2022lora` in refs.bib is garbage (journal "Iclr", wrong author); `heo2025study` pages should be 136--146; `assi2026llm` renders "Llm-cure". The corrected BibTeX is in the dossier.
    - Most important new references: `weyssow2025exploring` (LoRA beat random and retrieved examples for code generation; frame our plain-RAG result as agreeing and filtered RAG as the difference, never "overturn"), `mosbach2023few`, `zhao2021calibrate`, `nashid2023retrieval`. `dincc2025judge` is the closest SE precedent: same MiniLM + FAISS stack, fixed k = 5, and it names a k sweep and neighbor filtering as future work.
  - S3/S5: A1 also appears in §II's LoRA baseline subsection. A2 appears in §I l.13 and §II-D (`yu2023retrieval`). A6:
    - §II l.208 attributes lr 2e-4, paged AdamW 8-bit and grad-accum 16 to Heo and Lee and Aracena et al., but neither paper's text states them.
    - §II l.199 says the template is "verbatim" from both papers, but their templates differ.
    - §III l.7 says "the dataset introduced in Heo et al."; it should be "Heo and Lee", who extended NLBSE'24 with six projects.
    - `sclar2023quantifying` (prompt-format sensitivity, ICLR 2024) suits Threats.
  - One sub-agent put the user's email in a Crossref API `mailto=` parameter once (Crossref's "polite pool"). It was not sent anywhere else; flagged to the user.
Status: DONE

## S2 decisions (user) (2026-09-25)
- **No new papers in Related Work.** Do not cite De Vito et al. (TOSEM 2026), LabelMate, the NASA study (`colavito2026issue`), or any other entry in `docs/reframe/refs_candidates.bib`. The file and the dossier's "NEW" entries stay for the record only. S4 must not add them to `refs.bib` or the text.
- **Novelty sentence unchanged.** Keep the novelty sentence in §I and in the abstract as it is. Ignore the rewording options (A/B) in RW_DOSSIER.md.
- **Still to be fixed by S4:**
  - The accuracy findings about papers the paper already cites, A1–A4 in RW_DOSSIER.md:
    - A1: the best results come from GPT models fine-tuned through the API, and LoRA was used only for open models.
    - A2: `yu2023retrieval` does not support "similar examples beat random ones".
    - A3: `le2023log`, `logan2021...` and `ma2023fairness...` do not support "few-shot prompting helps".
    - A4: the "encoders need large training data" claim.
  - The broken existing `refs.bib` entries listed in the dossier's "refs.bib corrections" (for example `hu2022lora`, `heo2025study` pages, `assi2026llm` title).
  - The fixes must use only papers already in `refs.bib`.
- **Email:** never send the user's email address anywhere (including API parameters such as Crossref's `mailto=`) without asking first.
Status: DONE

## S3: headline reframe, three RQs, number diet (2026-09-25)
- Changed:
  - `00_abstract.tex` → rewritten on the S3 draft: context, fine-tuning's costs, gap (novelty sentence unchanged in content), study, filter, headline, plain RAG 2.8 points behind, same-data result, trade-off, closing restatement. **250 words** (counted "macro F1" as two words and "3,300" as one); one colon; no PS/PA, CIs or signed numbers.
  - `01_intro.tex` → ¶2 lists fine-tuning's costs (training run, GPU memory, pooling, retraining); ¶4 study design in words ("We conduct an empirical study that does both", no PS/PA); ¶5 introduces RAG, both reference points ($k$NN voting framed as a test: 59.5%), and the filter hypothesis; three RQ blocks (RQ3 follows the RQ3 box); "Overall" sentence with no numbers; three contributions ($k$NN voting is not one; replication package in bullet 1); one-sentence roadmap.
  - `03_approach.tex` → 2-sentence section intro; $k$NN voting role in one sentence; **filtered RAG calibration moved here** from old RQ2 (hypothesis → rule with a worked example → validation-split calibration → "otherwise identical"); LoRA baseline tightened (no more "faithfully reproduces" twice).
  - `04_setup.tex` (Evaluation Metrics only) → defines "best $k$" = highest test-set macro F1; one bootstrap footnote (paired, 1,000 resamples of the 3,300 test issues); "why macro F1" clause.
  - `05_evaluations.tex` → three RQs. Old RQ1 ($k$NN voting) and old RQ2 (RAG) merged into RQ1; old RQ3 → RQ2; old RQ4 → RQ3 with run-in paragraphs *Same labeled issues*, *Pooled fine-tuning* (per size in words, the fixed-$k$ = 12 check, heatmap), *Labeled data, memory, and time*, *Error profiles*, *Invalid outputs* (mechanism + fallback protocol and result). Stale coauthor/revision comments deleted; prose CIs removed (Table II has them). Opener: 2 sentences.
  - `06_discussion.tex` → failure analysis with counts and whole percents (116/20/14 of 150 = 77/13/9%; invalid 44/14/2 of 60 = 73/23/3%; both sum to 99% by rounding), plus a short paragraph on why templates affect all approaches (hypothesis) and the 34 hybrid/noise cases; "Key Insights" replaced by **Implications**: *Labeled data and retraining* (headline restated), *Where fine-tuning keeps an edge*, *Error profiles*, *Question-to-bug confusion*, *Retrieval alone as a floor*, *Classification and templates*. Commented-out Future Work block deleted.
  - `08_conclusion.tex` → three paragraphs kept (study; findings led by the headline with 4 numbers and the per-size guard; trade-offs + future work). Stale TOST comment block deleted.
  - `02_related.tex` → only `\Cref{sec:rq1,sec:ragtag,sec:method-comparison}` → `\Cref{sec:rq1,sec:rq3}`.
  - `scripts/paper/tab_method_comparison_ci.py` + `tables/method_comparison_ci.tex` → caption no longer says "paired bootstrap" (BRIEF 4.2): "…in percentage points, with 95% CIs, without and with the $k$NN voting fallback."
  - Dossier fixes in my files: A2 (`yu2023retrieval` dropped from §I and §II-D; `liu2022makes` alone), A6.1 (hyperparameters attributed only where the papers report them; `dettmers2023qlora` cited for paged AdamW 8-bit).
- New section labels: `sec:rq1` (RQ1: Classification with Retrieved Examples), `sec:rq2` (RQ2: Filtering the Retrieved Examples), `sec:rq3` (RQ3: RAG versus LoRA Fine-Tuning), `sec:disc-implications` (was `sec:disc-insights`). Removed: `sec:ragtag`, `sec:bragtag`, `sec:method-comparison`. Approach labels (`sec:approach-votag/ragtag/bragtag`) kept unchanged.
- Page state (`pagecheck.py`): main text ends page 10, left column, y=513; **73 free lines on page 10**. References start page 11, end page 12 (about 98 body-line equivalents of room). OK within 10 + 2. Build: no overfull boxes (the `\balance` warning also disappeared in this build), no undefined references.
  - History: the planned cuts alone freed 242 lines. The user chose to restore analysis until about 65–75 lines remain (S4 about 35–45 for Related Work, since no new papers are allowed; S5 about 25–30 for Threats and the final fill). Restored in the user's priority order: RQ3 per-size picture and the reason (fine-tuning's significant leads fall at the sizes where pooling helps it most), the full time trade-off, error profiles with per-label F1, invalid-output mechanism and fallback; RQ1 overlap, k-curve shape, zero-shot errors; RQ2 empty prompts at k ≤ 3, the Qwen-3B over-correction, best-k growth; Discussion implications; Conclusion.
- Numbers:
  - Every prose number is in NUMBERS.md, except setup and failure-analysis counts (300, 270, 30, 150, 60, 2,048, 8,192, 94.9%) and the whole-percent failure shares (77/13/9%, 23% = 34/150), which come from the counts.
  - Statements made **in words only** from `paper/tables/triangulation_all_cells.csv`, with no number in the prose:
    - bug→question share rises at every size under filtered RAG and nearly doubles at Qwen-3B;
    - RAG k-curve shape (smaller models decline beyond their best k, larger ones flat);
    - k = 1 beats zero-shot at every size;
    - zero-shot Qwen-14B is below Qwen-7B;
    - fine-tuning PS beats zero-shot at every size;
    - per-label F1 of fine-tuning PA vs filtered RAG (fine-tuning higher bug F1 at all sizes; filtered RAG higher question F1 at 3B and 32B).
  - "Differences on single projects reach about 10 points in either direction" comes from the hand-seeded heatmap CSV (−9.5 to +10.0).
  - No 3-decimal numbers in the PDF text. No "PS"/"PA" in the abstract or §I. No TOST or equivalence wording.
  - Number diet: results paragraphs carry at most 4 numbers, except the time paragraph (3,300; 0.10–0.44; 0.22–4.15; the $k$NN clause). The abstract has 7 numbers and the §I RQ3 block has 5 (the headline needs them).
- Honesty guards placed:
  1. Plain RAG 2.8 points behind, CI excludes zero: abstract, §I, RQ3 text and box, Discussion, Conclusion.
  2. Per size: RQ3 *Pooled fine-tuning* and Conclusion.
  3. k chosen on the test set, fixed k = 12 gives 1.2 points: §III-C and RQ3. **Threats still needs it (S5).**
  4. "11× per project" everywhere; RQ3 explains 3,300 vs 300 per target project and the total-labeling-effort caveat.
  5. RAG is not faster: abstract, §I, RQ3, Discussion (*Where fine-tuning keeps an edge*), Conclusion.
  6. Question confusion persists and is known: §I, RQ3, Discussion, Conclusion. No favorable comparison with prior work.
  7. Memory is per size, RAG inference vs fine-tuning's training plus inference: RQ3.
- Sentences I was unsure about (S5/S6 please check):
  - RQ2: "which suggests that the smallest model follows the remaining examples most closely" (a hedged interpretation).
  - RQ1: "We hypothesize that larger models make better use of long contexts with many examples."
  - RQ3: "We hypothesize that 300 issues are too few for fine-tuning to learn the task well …"
  - RQ3: "We hypothesize that the long prompts, filled with issue text, lead the model to continue that text rather than answer."
  - Discussion: "so correcting the template when an issue is filed could prevent many of them" and "a false bug label costs a triager's time, whereas a missed bug can delay a fix".
  - RQ3 *Labeled data*: I deliberately did **not** claim that 32B RAG fits a 24 GB GPU (22.3 GB peak), because the 32B k ≥ 12 runs used an L40 per §III.
- Open questions for the user: none new. The heatmap's 0.000 cell (lab machine) is still open.
- Notes for the next sessions:
  - S4:
    - Related Work still cites `\Cref{sec:rq1,sec:rq3}` for "$k$NN voting 32%, zero-shot 46–59%, fine-tuning 23–36%". All three numbers are stated in RQ1/RQ3.
    - "accuracy" appears once in 02_related.tex (BRIEF §6 bans accuracy discussion).
    - The Discussion now has a *Question-to-bug confusion* paragraph that points to `sec:02_related`. Keep the closing Related Work paragraph consistent with it.
  - S5:
    - Threats must state guard 3 (k selected on the test set; fixed k = 12 gives 1.2 points on average).
    - §III-A still says "Heo et al." (3×) and "data scope" (dossier A6.3). §II's "verbatim" template sentence (dossier A6.2) is still to decide: "We adopt the prompt template of these studies~\cite{heo2025study,aracena2025applying} verbatim" cites both, but the two templates differ.
    - Temperature/determinism wording in §II left as is (for S5).
    - The failure-analysis shares sum to 99% by rounding (counts are exact).
Status: DONE

## S4: Related Work with compare-and-contrast (2026-09-25)
- Changed:
  - `SANER2027/sections/02_related.tex` → rewritten as five run-in paragraphs (six blocks; question-to-bug keeps its two), each ending its description of prior work with a contrast to this study and a pointer to our result:
    1. *Supervised IRC*: classic ML, fine-tuned encoders, Siddiq and Santos (800K issues), SetFit on about 200 issues (`colavito2023few`, `colavito2024large`). Contrast: we hold the LLM fixed and vary only how the labeled issues are used, so the baseline is LoRA of the same models; our encoder is frozen and only retrieves. No claim about encoder accuracy or data needs (A4 fixed; the "limits generalizability" sentence is gone).
    2. *LLMs for IRC*: zero-shot varies across datasets; Colavito et al.'s random 1–2 examples per class; **A1 fixed** (best results = GPT fine-tuned through OpenAI's API; open Llama-3.1-8B / DeepSeek-R1-Distill-Llama-8B fine-tuned with LoRA scored lower); Heo and Lee's two settings; cost reported as training/inference time or API prices, not GPU memory; the two future-work suggestions. Contrast: we evaluate both together, up to 15 retrieved examples, LoRA of the same four models in both settings, peak GPU memory; RAG beats zero-shot at every size and k (RQ1). Ends with the novelty sentence in the **same terms as §I** ("has not been systematically evaluated for IRC or compared with LoRA fine-tuning").
    3. *Retrieved in-context examples*: KATE, Milios et al. (frozen encoder, up to 150 labels), Yu et al. as the trained-retriever/trained-classifier alternative (**A2 fixed**), LLM-Cure's five fixed examples, Dinç and Tüzün as the closest SE work (same MiniLM + FAISS, k = 5, encoders ahead, k sweep and neighbor filtering named as future work), Logan et al. (fine-tuning beats ICL for sub-1B masked LMs; **A3 fixed**, moved here). Contrast: three-label IRC, eleven projects, k sweep bounded by the vote, kNN voting and zero-shot references (RQ1), same-model LoRA at 3B–32B; RAG 2.8 points behind, filtered RAG 1.0 point and not significant (RQ3).
    4. *Label bias in the examples*: Ma et al. (similarity selection → label-dominance bias; entropy-scored search with extra LLM calls; **A3 fixed**, moved here), Milios et al. (neutral class rarely retrieved), our matching observation (validation split, §II-E; zero-shot errors, RQ1) and hypothesis, Dinç and Tüzün's judge call. Contrast: filtered RAG keeps per-query retrieval, count rule on the neighbors, no extra LLM call or output probabilities; 1.5–2.4 points over RAG at the cost of bug recall (RQ2).
    5. *Question-to-bug misclassification* (kept, updated): Aracena et al.'s two remedies (retrieval augmentation; a first stage that separates questions) now contrasted with the filter; our reduction stated only against our own zero-shot and RAG (guard 6); kNN voting's 32% dropped for the number diet. Explanations paragraph: new contrast that our benchmark is balanced and the confusion remains, so imbalance alone does not explain it.
    - `le2023log` is no longer cited (A3; it drops out of the bibliography). The word "accuracy" no longer appears in Related Work.
  - `SANER2027/refs.bib` → dossier corrections: `hu2022lora` (ICLR 2022, correct authors), `heo2025study` pages 136--146, `assi2026llm` title casing, `ma2023...` (NeurIPS 2023), `logan2021...` (Findings of ACL 2022), `milios2023context` (GenBench 2023), `reimers2019sentence` (EMNLP-IJCNLP 2019), `sclar2023quantifying` (ICLR 2024; S5 may cite it in Threats), `liu2022makes` author "Dolan, Bill", DOIs for `colavito2023few`, `aracena2024...`, `siddiq2022bert`; `colavito2024large` gets CEUR vol. 3762 and loses the trailing period in its title; `siddiq2022bert` title casing; stray `=====` line removed. No entry from `refs_candidates.bib` added.
- Citations newly used in the paper (all already in refs.bib): `siddiq2022bert`, `colavito2023few`, `milios2023context`. No new papers (user decision).
- Decisions: novelty sentence unchanged in §I and the abstract (user decision); Related Work repeats it in the same terms. De Vito et al., LabelMate, the NASA study: not cited (user decision). `01_intro.tex` untouched.
- Page state (`pagecheck.py`): main text ends page 10, right column, y=346; **31 free lines on page 10** for S5 (Related Work used 42 of S3's 73, inside the user's 35–45 allotment). References start page 11 (top), end page 12 with about 86 body-line equivalents of room. OK within 10 + 2. Build: only the known `\balance` overfull vbox, no undefined references, bibtex 0 warnings.
- Numbers: every number in the section is in NUMBERS.md or BRIEF §9 (2.8, 1.0, 1.5–2.4, 46–59, 26–36, 19–27; 15 = largest k). Prior-work numbers (800,000; about 200; 150 labels; k = 5; 46–54%) come from the dossier or the existing verified text.
- Open questions for the user: none new.
- Notes for S5:
  - Sentences worth a second look for tone: "We see a similar pattern for questions" (links Milios et al.'s neutral-class finding to our validation-split observation); "so imbalance alone does not explain it" (the benchmark is balanced per §III).
  - `sclar2023quantifying` is fixed in refs.bib and uncited; the dossier suggests it for Threats (prompt-format sensitivity).
  - Still outside my files (from S2/S3): §III-A "Heo et al." (3×) and "data scope"; §II's "verbatim" template sentence (A6.2).
Status: DONE

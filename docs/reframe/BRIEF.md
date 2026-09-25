# BRIEF: shared context for every reframe session (read this first, in full)

You are one of six Claude sessions revising the SANER 2027 paper "Can Retrieval-Augmented Few-Shot
Prompting Match LoRA Fine-Tuning for Issue Report Classification? An Empirical Study". Read this file,
then `docs/reframe/LOG.md` (what earlier sessions did and the current page state), then your own
session file `docs/reframe/S<n>_*.md`. Your session file says what to do; this file says what must
always stay true.

## 1. Situation

- **Venue:** SANER 2027 Research Track. `\documentclass[10pt,conference]{IEEEtran}`, double-blind,
  **10 pages of main text** (everything through the Data Availability statement, all figures and tables
  included) **plus at most 2 pages that contain only references**. The PDF may have at most 12 pages.
  Violations are desk rejects.
- **The user's layout requirement for the final draft:** the main text fills **exactly 10 pages**.
  Page 10 is full to the bottom of its right column (at most 3 free lines), and **References start at
  the top of page 11**. A `\clearpage` before the bibliography (added by S1) guarantees the page break,
  so any space left on page 10 shows as a visibly empty gap. Fill it with substance (see §8), never
  with padding. `python3 docs/reframe/pagecheck.py --final` checks this target.
- **Deadline:** paper due 2026-09-25 AoE (about 2026-09-26 07:00 CDT). Work efficiently. Leave the paper
  buildable and within the page limit when you finish.
- **History:** the ESEM 2026 version was rejected. The decisive reviewer was a skeptical expert, so
  overclaims, unsupported statements and sloppy numbers are the biggest risk. Every claim needs a number
  or a citation behind it.
- **Why this revision:** the user and their supervisor gave four problems. They are the success
  criteria for all six sessions (verbatim, typos included):

> 1- there are too many rag-based approaches and too many names and all the names are capitalized and read like one another. and they read very silly like RAGTAG and BRAGTAG.
>
> 2- my supervisor also said that Related work is a good survey, but you should also treat it as an opportunity to differentiate yourself from prior work. Instead of just describing prior work, describe it and then compare and contrast the approach with your own. This might be an opportunity to expand the paper a bit.
>
> 3- I think your current framing of the results emphasizes the slight increase in F1 score, but I don't think the increase is significant enough for a reader to really be convinced by. The key finding is that we empirically analyzed RAG for IRC and found that preserves F1 score while taking 11x less training data and 1/3 less GPU resources. Framing the paper around that and then using the F1 scores to back up the narrative makes a more convincing storyline. I think this note from my supervisor also ties into the fact that the diffferences between the approaches is pretty small which leads the reported numbers to have a lot of decimals in them. which might be hard for the reviewrs to read and follow.
>
> 4- some sections of the paper were written by AI which sometimes makes them look unnatural and read very badly and maybe not clearly or ambigus. the wording and tone in those parts must be fixed as well.

## 2. Sessions and order

| Session | File | Does | Depends on |
|---|---|---|---|
| S1 | `S1_names_numbers.md` | New method names, percent format, no draft colors, regenerated tables and figures | none |
| S2 | `S2_related_work_research.md` | Research dossier for Related Work (no paper edits) | none (runs in parallel with S1) |
| S3 | `S3_reframe.md` | Headline reframe, 3 RQs, number diet (abstract, intro, approach, evaluation, discussion, conclusion) | S1 |
| S4 | `S4_related_work.md` | Related Work rewrite with compare and contrast | S2, S3 |
| S5 | `S5_tone_accuracy_threats.md` | Tone and clarity pass, accuracy fixes, Threats rewrite | S4 |
| S6 | `S6_verify.md` | Independent verification and reviewer simulation, change-view PDF | S5 |

Before starting, check in `LOG.md` that the sessions you depend on have logged "DONE". If one has not,
stop and tell the user.

## 3. Workspace rules

- **Scope (user instruction): work on the text of the paper.** Do not check the paper's statements
  against the implementation (`llm_labeler.py`, `fixed_fine-tune.py`, run scripts, experiment folders,
  raw data such as `issues11k*.csv`). Take the experimental setup as the paper describes it.
  - The float generators in `scripts/paper/` and their CSVs in `paper/tables/` are part of the paper
    (they produce its tables and figures) and are the source of its numbers; use them for that.
  - The cited papers (local PDFs in `docs/`, DOIs) are fair game for checking what the paper says about
    prior work.

- Work **only** in `/home/alireza/Desktop/my_projects/saner-reframe` (git worktree, branch
  `saner-reframe`). Before your first edit, run `pwd` and `git branch --show-current` and confirm both.
- **Never write to `/home/alireza/Desktop/my_projects/automated-issue-report-labeling`.** That folder
  holds the preserved draft (branch `encoder-baselines`, tag `saner-draft-2026-09-25`). Reading files
  there is allowed; writing, building or committing there is not.
- **Baseline for diffs:** tag `saner-draft-2026-09-25`, for example
  `git diff saner-draft-2026-09-25 -- SANER2027/sections/`.
- **Build:** `SANER2027/build.sh` (pdflatex, bibtex, then pdflatex twice; prints pages, overfull boxes
  and undefined references). **Page check:** `python3 docs/reframe/pagecheck.py`. It prints where the main
  text ends, the free lines left on page 10, where the references start and end, the room left on page
  12, OK or OVER, and whether the final target is met ("page 10 full, references from page 11"). Add
  `--final` to make an unmet final target an error. Run both after every section you change.
- The baseline build already shows two `Overfull \vbox ... while \output is active` warnings. They come
  from `\balance` on the reference pages and are known and harmless. Any other overfull box is yours to
  fix.
- **Tables and figures are generated.** The generators in `scripts/paper/` read committed CSVs in
  `paper/tables/` and run on this machine with the system `python3` (pandas and matplotlib are
  installed). Edit the generator and rerun it; never hand-edit `SANER2027/tables/results_master.tex` or
  `method_comparison_ci.tex`. The exception is `SANER2027/tables/method_cost.tex`, which is hand-written.
- **Where the data are:** `paper/tables/triangulation_all_cells.csv` has every (method, setting, model,
  k) cell with exact P/R/F1, confusion rates and bug shares. `paper/tables/method_comparison_ci.csv` has
  the paired-bootstrap differences against pooled fine-tuning (raw and with fallback).
  `paper/tables/per_project_diff.csv` is the heatmap data (hand-seeded at 3 d.p.).
  `SANER2027/tables/method_cost.tex` has memory (GB) and times (h). After S1 these values are collected in
  `docs/reframe/NUMBERS.md`.
- **No lab machine, no GPU.** `results/` does not exist on this PC. If a change would need new
  experiments or the lab machine, do not do it; record it in LOG.md as an open item. **Never invent or
  estimate a number.**
- `paper/` is the frozen ESEM version. Do not edit it (reading its CSVs under `paper/tables/` is
  expected).
- The user edits in VS Code, which builds on save. If a file you edited changes under you, re-read it
  before writing again.
- Use your session's scratchpad directory for temporary files, not the repo.
- **Git:** commit only the files you changed, **by explicit path** (never `git add -A` or `git add .`,
  because another session may be running). Use commit messages like `SANER reframe S3: ...` and end
  every message with the line
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Push with `git push origin saner-reframe`.
  Commit at sensible checkpoints, not only at the end.
- **LOG.md:** when you finish, append an entry in the format given at the top of LOG.md. Include what
  changed (file → one line), the page state (`pagecheck.py` output), every open question for the user,
  and anything the next session must know. The last line of the entry is `Status: DONE` (or
  `Status: PARTIAL` with reasons).
- **If unsure,** ask the user directly if they are present. Otherwise leave that text unchanged and put
  the question in LOG.md. Do not guess on facts about the experiments or on the user's preferences.

## 4. Decisions already made by the user (do not reopen)

### 4.1 Names

| Old | New (macro → output) | Role |
|---|---|---|
| VOTAG (`\votag`) | `\knn` → "$k$NN voting" | A retrieval-only **baseline**, a similarity-weighted $k$-nearest-neighbor vote over the neighbors' labels. It has no LLM. It is not a proposed method and not a contribution. |
| RAGTAG (`\ragtag`) | `\rag` → "RAG" | The approach studied: retrieval-augmented few-shot prompting. The first mention in the abstract and body is "retrieval-augmented few-shot prompting (RAG)". |
| BRAGTAG (`\bragtag`) | `\frag` → "filtered RAG", `\Frag` → "Filtered RAG" (sentence start) | A variant of RAG that removes bug-labeled examples when the neighbors suggest the query is a question. |
| Fine-Tune / FT | "fine-tuning" ("LoRA fine-tuning" at first mention and in headings) | LoRA fine-tuning of the same Qwen2.5-Instruct model. Never write "Fine-Tune" or "FT" in prose; tables may use "Fine-tuning". |
| Zero-shot | "zero-shot prompting" / "Zero-shot" (tables) | RAG with $k=0$, the LLM without retrieval. |

**Fixed terms** (use one word for one thing):
- *neighbors*: the labeled issues retrieved for a query.
- *examples*: the neighbors shown in the prompt.
- *query issue* or *query*: the issue being classified (not "target issue", not "test issue" outside
  the evaluation setup).
- *target project*: the project the query comes from.
- *setting*: PS or PA (not "data scope").
- *label*: bug, feature or question; *class* only in "per-class" metrics.
- *invalid output*: an output that contains none of the three labels.
- *fallback*: $k$NN voting labels the issues whose LLM output is invalid.

**Acronyms** allowed in prose: IRC, LLM, LoRA, RAG, CI, and PS/PA in the body only. PS and PA never
appear in the abstract or the Introduction. There, spell out "the target project's labeled issues" and
"all 3,300 labeled issues of the eleven projects". In the body, use PS/PA only where the setting is not
clear from context. After §IV establishes that RAG uses PS and fine-tuning uses PA, drop the tags where
they are redundant.

### 4.2 Number format

- Macro F1, precision and recall are **percentages with one decimal**: "76.7%" in prose, "76.7" in
  tables with "(%)" in the caption or header.
- Differences between methods are **percentage points**, written "points" ("1.0 point behind", "2.1–4.0
  points higher"). §III (Evaluation Metrics) defines this once: "We report scores in percent and
  differences in percentage points (points)."
- Shares and error rates (for example, the share of questions labeled as bugs) are whole percents
  ("26–36%") unless one decimal is needed to distinguish values. Invalid rates keep one decimal.
- **No three-decimal numbers anywhere.** No signed numbers such as "+0.1" or "−2.8" in prose; say "ahead"
  or "behind". Confidence intervals appear only in Table II; prose says whether an interval "includes
  zero" or "excludes zero". The bootstrap details (paired, 1,000 issue-level resamples) go in one text
  footnote, never in a caption.
- **Every number must come from `docs/reframe/NUMBERS.md`** (written by S1 from the CSVs). Round from
  exact values. Never compute a difference or a range from values that are already rounded. For example,
  0.738 − 0.677 = 0.061 from the rounded table, but the exact difference is 6.0 points.
- **Number diet:** a prose paragraph carries at most 3–4 numbers, and one claim gets one number or one
  range. Per-class detail belongs in Table I, and CIs belong in Table II. Answer boxes carry at most 3
  numbers.

### 4.3 Research questions (three, in build-up order)

- **RQ1:** How well does retrieval-augmented few-shot prompting classify issue reports, and how many
  examples does it need? (It merges the old RQ1 on VOTAG and RQ2 on RAGTAG. Zero-shot prompting is the
  LLM-only reference and $k$NN voting is the retrieval-only reference.)
- **RQ2:** Does filtering the retrieved examples reduce question-to-bug errors? (The old RQ3.)
- **RQ3:** Can RAG match LoRA fine-tuning with less labeled data and GPU memory? (The old RQ4, reordered
  to lead with the headline.)

Section labels after S3: `sec:rq1`, `sec:rq2`, `sec:rq3`.

### 4.4 Other decisions

- The **title stays** unchanged. It was registered on EasyChair and already asks the headline question.
- **Encoder baselines** (SetFit, RoBERTa) stay out of the paper (`tables/encoder_baselines.tex` stays
  un-input).
- **Draft colors are gone** (S1 removes `\fixed`, `\coauthor` and the blue tables). The change record is
  the git diff and a latexdiff PDF against the tag (S6).
- **Related Work stays after the Discussion** (user decision; it lets the Related Work point back to
  results and to the failure analysis).

## 5. The headline and the facts behind it

**Headline (the frame for the whole paper):** *With a simple filter on its examples,
retrieval-augmented few-shot prompting matches LoRA fine-tuning in macro F1 while using 11× less
labeled data per project, no training, and a third less peak GPU memory.*

The facts that back it (verify against NUMBERS.md):
- **F1 parity.** Filtered RAG (project-specific retrieval, best k) trails pooled fine-tuning (PA) by
  1.0 point in macro F1 on average over the four model sizes, and the 95% CI includes zero. With the
  fallback applied to both methods, filtered RAG is 0.1 point ahead, and the CI includes zero.
- **Data.** RAG uses only the target project's 300 labeled issues. Fine-tuning reaches its best score
  only with all 3,300 labeled issues pooled from the eleven projects, 11× more per target project. Given
  the same 300 issues, RAG beats fine-tuning at every size by 2.1–4.0 points, and filtered RAG by
  3.8–6.0 points.
- **Training.** RAG has no training phase. A new labeled issue is only indexed, and the LLM can be
  replaced without retraining.
- **GPU memory.** Peak GPU memory is 33% lower on average (47/31/25/29% at 3B/7B/14B/32B), measured
  against fine-tuning's training plus inference.

**Honesty guards.** These must stay true and must appear where the corresponding claim is made.
1. **Plain RAG does not match.** It trails pooled fine-tuning by 2.8 points on average, and the CI
   excludes zero. Say so plainly, including in the abstract. "Matches" belongs to filtered RAG, the
   approach with its example filter.
2. **Per size,** fine-tuning is significantly ahead of filtered RAG at 7B and 14B (2.4 and 2.9 points).
   Filtered RAG is ahead at 3B and 32B, but not significantly. RQ3 states this.
3. **k is selected on the test set.** Each method's "best k" is the one with the highest test-set macro
   F1. With a single k = 12 for every size, filtered RAG trails pooled fine-tuning by 1.2 points on
   average. RQ3 and Threats state this.
4. **"11×" is per target project, not total annotation effort.** The eleven project-specific indexes
   together hold the same 3,300 issues as the pooled training set. Never write "11× less annotation" or
   "11× fewer labels" without "per project" or an equivalent.
5. **RAG is not faster.** Its inference is slower (longer prompts). Total time including training is
   lower than fine-tuning at 3B and 14B, similar at 7B, and higher at 32B (4.15 h vs 2.71 h). Never
   claim RAG saves time; the saving is memory, data and retraining.
6. **Question-to-bug confusion is known.** Prior IRC work reports it, and it persists under every
   method we test, fine-tuning included. Never claim that RAG or filtered RAG beats prior work on the
   question class. Claim only reductions relative to our own zero-shot and RAG results.
7. **Memory comparisons are per model size,** RAG inference against fine-tuning training plus inference,
   on the hardware stated in §III.

## 6. Content rules from earlier rounds with the user

- **Never report accuracy** (on this balanced test set it equals macro recall; the user wants no
  accuracy discussion).
- **Never use TOST, "equivalence", "statistically equivalent" or equivalence margins.** Use paired
  bootstrap 95% CIs only.
- Write **"Heo and Lee"** (two authors), never "Heo et al.".
- **Prior work's best LLM results for IRC come from GPT models fine-tuned through an API.** LoRA was
  used for open models (Heo and Lee: Llama-3.1-8B with 4-bit PEFT/LoRA; Aracena et al.:
  DeepSeek-R1-Distill with Unsloth and LoRA). Never call LoRA fine-tuning "the state of the art". Say it
  is the standard way to fine-tune open LLMs for IRC in prior work.
- **$k$NN voting has no LLM.** Never list it among the methods compared with fine-tuning "of the same
  models".
- The question-to-bug confusion is placed in prior work: a cited clause at first mention, and the
  closing paragraph of Related Work.
- The NLBSE'24 benchmark equals five of our eleven projects. Prior question-class numbers (for example,
  Heo and Lee's GPT-4o fine-tune reaching question F1 of about 0.85 on our benchmark) are higher than
  ours, so never compare our question results with prior work favorably.
- De Vito et al. (TOSEM 2026) was set aside by the user earlier. S2 re-examines it; nobody cites it
  without the user's approval.

## 7. Writing rules (from the user's feedback on earlier drafts)

1. Use the **shortest wording** that says it. No filler, no restating.
2. **One line of argument per paragraph.** An added sentence must follow from the paragraph it joins;
   the user deletes sentences that "appear out of nowhere". Check that the paragraph leads up to the
   claim before inserting it.
3. **Introduce a method by what it does, with its name in parentheses**, for example "a
   similarity-weighted vote over the neighbors' labels ($k$NN voting)". Never open a paragraph or the
   abstract with a bare method name the reader has not met.
4. **Present assumptions as tests with results.** Say what we test, how, the result, and what it means.
   Never voice a doubt about our own premise. For example, write "the neighbors' labels alone reach 59.5%
   macro F1", not "similarity does not guarantee the same label".
5. **Give the real reason for each design choice,** and phrase hypotheses as hypotheses ("we
   hypothesize that bug-labeled examples reinforce this error").
6. **Recaps restate results and trade-offs, not recommendations** about who should use which method.
   "Competitive" or "matches" is allowed only when the same or the next sentence gives the facts.
7. **No defensive or reviewer-answering tone.** Avoid phrases such as "cannot fully explain", "is
   consistent with this explanation", "our observations bear on each", or "rather than a new
   observation". Rewrite so the text simply no longer has the problem.
8. **Avoid the "claim: explanation" colon pattern.** At most one per paragraph, and only when there is
   no other way.
9. Write "**we conduct an empirical study**", not "In an empirical study, we ...".
10. When citing a **cause from prior work,** follow it at once with our own matching observation and a
    pointer to where it comes from (for example, the failure analysis).
11. Write differences in words or points, **never as signed numbers** in prose.
12. Every change must leave **no knock-on inconsistency** elsewhere in the paper. When you change a
    claim, grep for its other occurrences (abstract, intro, answer boxes, discussion, conclusion).

## 8. Page rules

- Main text through Data Availability must end on page 10; references run from the top of page 11 and
  may continue to page 12.
- **Exact fill (user requirement).** The finished paper has exactly 10 full pages of main text. Budget
  across the sessions:
  - **S3** frees at least 45 lines on page 10.
  - **S4** (Related Work) uses about 30 of them and leaves about 15.
  - **S5** writes the Threats section and then lands the text **exactly at the bottom of page 10**
    (`pagecheck.py --final` passes).
  - **S6** verifies this and fixes it only by a line or two.
  If space remains after the planned work, fill it in this order of value, and never with filler:
  1. a sharper compare-and-contrast sentence in Related Work that the dossier supports;
  2. a missing Threats item;
  3. a result the reader needs but that was cut (for example, one per-class number that supports a
     trade-off);
  4. a clarifying sentence in the Discussion's implications.
  If the text runs over, cut the weakest sentence, not a float.
- Do not shrink figures, fonts or captions. Do not set `\arraystretch` below 0.97, and do not add
  negative `\vspace` or other spacing hacks. Pay for added lines by cutting prose.
- Adding one line of main text also pushes the references down by one line; `pagecheck.py` reports the
  usable slack as the smaller of the two limits.
- Floats: `figures/kcurves.pdf` and `tables/results_master.tex` are `figure*`/`table*` at the top of
  §IV. Savings in page-7 floats move the final break, while savings in page-8 floats often do not.
  Always check with `pagecheck.py`.
- Baseline state (tag `saner-draft-2026-09-25`): 12 pages; the main text ends on page 10, right column,
  with about 9 free lines. The references then start on page 10. After S1's `\clearpage` they start on
  page 11 and end on page 12 with room for roughly 30–40 more entries.

## 9. Anchor numbers (percent, from exact CSV values; S1's NUMBERS.md is authoritative)

Macro F1 (%) at each method's best configuration, pooled over the 3,300 test issues:

| | Qwen-3B | Qwen-7B | Qwen-14B | Qwen-32B |
|---|---|---|---|---|
| Zero-shot | 61.3 | 66.2 | 64.5 | 68.8 |
| RAG (PS; best k 3 / 6 / 12 / 12) | 69.7 | 71.8 | 73.2 | 76.7 |
| Filtered RAG (PS; best k 6 / 12 / 15 / 12) | 71.4 | 73.8 | 75.6 | 78.1 |
| Fine-tuning (PS) | 67.6 | 67.7 | 70.4 | 74.0 |
| Fine-tuning (PA) | 70.8 | 76.2 | 78.5 | 77.1 |
| Filtered RAG minus fine-tuning PA (points) | +0.5 | −2.4 | −2.9 | +1.0 |

- $k$NN voting: 59.5% (PS, k = 15) and 60.4% (PA, k = 16); it labels 32% of questions as bugs.
- Questions labeled as bugs: 46–59% (zero-shot), 26–36% (RAG), 19–27% (filtered RAG), 23–36%
  (fine-tuning PA).
- Differences against pooled fine-tuning, mean over the four sizes [95% CI] (points): RAG −2.8
  [−3.9, −1.8]; filtered RAG −1.0 [−2.0, +0.1]. With the fallback: RAG −2.0 [−3.0, −1.0]; filtered RAG
  +0.1 [−0.9, +1.1]; filtered RAG at 32B +2.1 [+0.6, +3.5].
- Filtered RAG over RAG at each method's best k: 1.5–2.4 points (all four CIs exclude zero).
- Fixed k = 12 for all sizes: filtered RAG minus pooled fine-tuning is −1.2 points on average (−0.2,
  −2.4, −3.1, +1.0).
- Peak GPU memory (GB), RAG vs fine-tuning: 2.9 vs 5.5, 6.8 vs 9.9, 12.6 vs 16.7, 22.3 vs 31.5, that is
  47/31/25/29% less and 33% less on average.
- Time (h): fine-tuned models label the 3,300 test issues in 0.10–0.44 h; filtered RAG takes 0.22–4.15 h.
  Total at 32B: 4.15 h (filtered RAG) vs 2.71 h (fine-tuning). $k$NN voting: under 4 s and 0.2 GB.

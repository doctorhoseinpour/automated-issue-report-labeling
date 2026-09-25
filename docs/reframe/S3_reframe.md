# S3: reframe the paper around the headline, restructure into three RQs, put the prose on a number diet

**Read `docs/reframe/BRIEF.md` in full first, then `docs/reframe/LOG.md`.** S1 must have logged
`Status: DONE` (new names, percent format, `docs/reframe/NUMBERS.md`). If not, stop and tell the user.
Read `NUMBERS.md` completely before writing: it is your only source of numbers.

## Goal

Problem 3 from BRIEF is your problem. Right now the paper reads as "our method scores a little higher".
After S3 it reads as: *"We asked whether retrieval-augmented few-shot prompting can replace LoRA
fine-tuning for IRC. With a simple filter on its examples, it matches fine-tuning's macro F1 while
using 11× less labeled data per project, no training, and a third less GPU memory."* The F1 results
then back that story up. Each RQ builds toward it, and the abstract, introduction, RQ3, discussion and
conclusion all state it in the same terms with the same numbers.

You also restructure §IV into three RQs (BRIEF §4.3) and cut the numbers in the prose (BRIEF §4.2,
"number diet"). The cuts must also **free page space for S4 (Related Work) and S5 (Threats)**. The
target is at least **45 lines** of usable slack from `pagecheck.py`, and more is better (the baseline has
about 9).

**Files you own:** `SANER2027/sections/00_abstract.tex`, `01_intro.tex`, `03_approach.tex`,
`04_setup.tex` (Evaluation Metrics paragraph only), `05_evaluations.tex`, `06_discussion.tex`,
`08_conclusion.tex`. In `02_related.tex` and `07_threats.tex`, change only `\cref`/`\ref` targets that
your relabeling breaks; S4 and S5 rewrite those sections.

Tone matters too. Write plain, precise academic English (BRIEF §7). S5 does a dedicated tone pass, but
do not create new work for it.

## Before writing

1. Read the whole current PDF text (`pdftotext -layout`) once as a reader, so you know the argument end
   to end.
2. In your scratchpad, list every sentence in the files you own that states a result. Mark each as
   KEEP (needed for the argument), MOVE (to a table, or it is already in one), or CUT (repeated
   elsewhere or not needed for the argument). Use the list to do the number diet systematically.

## 1. Abstract (`00_abstract.tex`)

**Constraints:**
- at most 250 words (IEEE limit; count "macro F1" as two words and "3,300" as one; use a small script);
- one paragraph;
- at most about 6 numbers;
- no PS/PA, no CI brackets, no signed numbers.

**Order:**
1. context;
2. fine-tuning's costs: a training run, pooling labeled issues from many projects to reach its best
   score, GPU memory, and retraining for new labels or models;
3. the gap;
4. "we conduct an empirical study" with the scale;
5. what RAG does, the filter and its reason (RAG's most frequent error, questions labeled as bugs, is
   known from prior work);
6. **the headline with its three facts**;
7. plain RAG 2.8 points behind, and the same-data result;
8. the trade-off (slower inference, different errors);
9. one closing sentence that restates the finding, not a recommendation.

**Starting draft** (about 260 words). Tighten it to 250 or fewer and check every number against
NUMBERS.md:

> Classifying issue reports as bugs, feature requests, or questions helps developers triage them. The best reported results come from fine-tuned large language models (LLMs), but fine-tuning needs a training run, reaches its best scores only when labeled issues are pooled across projects, and must be repeated to learn from new labels or to adopt a newer model. Retrieval-augmented few-shot prompting (RAG) needs no training: it shows the LLM the labeled issues most similar to the query as examples. To our knowledge, it has not been systematically evaluated for issue report classification or compared with Low-Rank Adaptation (LoRA) fine-tuning. We conduct an empirical study with four Qwen2.5-Instruct models of 3B to 32B parameters and 6,600 issue reports from eleven projects. Like prior methods, RAG most often labels questions as bugs, so we also evaluate a variant (filtered RAG) that removes bug-labeled examples when the retrieved issues suggest a question. Using only the target project's 300 labeled issues, filtered RAG comes within 1.0 point of the macro F1 that LoRA fine-tuning reaches with all 3,300 labeled issues of the eleven projects, a difference that is not statistically significant, while needing no training and 33% less peak GPU memory on average. Without the filter, RAG trails fine-tuning by 2.8 points. Given the same 300 labeled issues, both RAG variants outperform fine-tuning at every model size. RAG's inference is slower, and the two approaches make different errors: fine-tuning finds more bugs but over-predicts the bug label. Retrieval-augmented few-shot prompting thus preserves the macro F1 of LoRA fine-tuning with 11× less labeled data per project, no training, and a third less GPU memory.

Notes:
- The colon after "needs no training" is the one allowed per paragraph. Remove one of the two colons in
  the draft.
- If a sentence has to go, drop the same-data sentence before any sentence of the headline.
- Keep the title unchanged.

## 2. Introduction (`01_intro.tex`)

Keep the current structure where it works, and change the parts listed here.

- **¶1 (problem)** stays; minimal edits only.
- **¶2 (fine-tuning)** makes fine-tuning's costs concrete: a training run per model; LoRA on open models
  still needs GPU memory for training; the best scores come from pooling labeled issues across projects
  (Heo and Lee's project-agnostic setting); and retraining whenever labels are added or the model is
  replaced. Keep the accurate "best reported results ... with LoRA for open models" (BRIEF §6).
- **¶3 (RAG and the gap)** stays, with light edits: a new labeled issue only needs indexing, and the
  model can be swapped; Heo and Lee and Aracena et al. suggest it as future work; "to our knowledge"
  gap.
- **¶4 (study design)** in words, with no PS/PA: four Qwen2.5-Instruct sizes; Heo and Lee's benchmark;
  the target project's 300 labeled issues vs all 3,300 labeled issues of the eleven projects, for both
  retrieval and fine-tuning.
- **¶5 (replaces the old naming paragraph).**
  - RAG shows the LLM the target project's labeled issues most similar to the query as examples.
  - Two reference points separate what retrieval and the LLM each contribute: zero-shot prompting
    (the LLM without retrieved examples) and a similarity-weighted vote over the neighbors' labels
    (the $k$NN voting baseline, retrieval without the LLM). Frame $k$NN voting as the test of how much
    label information the neighbors carry; its 59.5% shows they carry a lot, which motivates giving
    them to the LLM (BRIEF §7 rule 4).
  - Then the filter: RAG's most frequent error is labeling questions as bugs, as in prior IRC methods
    (keep the citations). We hypothesize that bug-labeled examples retrieved for a question reinforce
    this error, and filtered RAG removes them when the neighbors suggest a question.
- **"We answer three research questions."** Each RQ block is a run-in heading, the question in italics,
  and a 1–2 sentence answer with at most 2 numbers. **RQ3's block carries the full comparison** and
  follows the RQ3 answer box in order and wording, because the user dislikes results scattered between
  blocks. Starting drafts (check the numbers):
  - **RQ1** *How well does retrieval-augmented few-shot prompting classify issue reports, and how many
    examples does it need?* With examples from the target project, RAG outperforms both zero-shot
    prompting and $k$NN voting at every number of examples, reaching 69.7–76.7% macro F1 at its best
    k, between 3 and 12. The remaining errors are mostly questions labeled as bugs (26–36% of
    questions).
  - **RQ2** *Does filtering the retrieved examples reduce question-to-bug errors?* Filtered RAG lowers
    the share of questions labeled as bugs to 19–27% and raises macro F1 over RAG by 1.5–2.4 points,
    at the cost of bug recall.
  - **RQ3** *Can RAG match LoRA fine-tuning with less labeled data and GPU memory?* Given the same 300
    labeled issues, both RAG variants outperform fine-tuning at every model size. Fine-tuning needs
    all 3,300 labeled issues of the eleven projects to catch up. It then leads filtered RAG by 1.0
    point on average, a difference that is not statistically significant, and leads RAG by 2.8
    points. Both RAG variants use 33% less peak GPU memory and need no training, but their inference
    is slower. Fine-tuning finds more bugs but over-predicts the bug label.
- **Closing sentence ("Overall ...")**: the headline and the trade-off with **no new numbers**, e.g.
  "Overall, retrieval-augmented few-shot prompting preserves the macro F1 of LoRA fine-tuning with less
  labeled data per project, no training, and less GPU memory, and the two approaches make different
  errors."
- **Contributions** (3 bullets; $k$NN voting is not one):
  1. the first systematic evaluation of retrieval-augmented few-shot prompting for IRC, against LoRA
     fine-tuning of the same four models on the same labeled issues, covering macro F1, per-class
     errors, labeled data, peak GPU memory and time;
  2. evidence that it matches fine-tuning's macro F1 with 11× less labeled data per project, no training
     and a third less GPU memory, and beats fine-tuning when both use the same labeled issues;
  3. a training-free example filter that reduces the question-to-bug confusion known from prior IRC
     work, with an analysis of the error trade-offs between the approaches.
  Add a replication-package clause to bullet 1 or 3 only if it fits.
- **Roadmap:** one short sentence, or cut it if space is tight. Keep the section references correct
  (Approach §II, Setup §III, Evaluation §IV, Discussion §V, Related Work §VI, Threats §VII, Conclusion
  §VIII).

## 3. Approach (`03_approach.tex`)

- **Section intro:** replace the current paragraph, which lists the subsections one by one, with 2–3
  sentences. "All retrieval-based classifiers share one retrieval step (§II-B). The $k$NN voting
  baseline labels the query from its neighbors' labels alone; RAG shows the neighbors to the LLM as
  labeled examples; filtered RAG removes bug-labeled examples for suspected questions. We compare them
  with LoRA fine-tuning of the same LLMs (§II-F); \cref{fig:approach-overview} shows all four." Keep the
  figure.
- **Problem formulation and retrieval:** keep, and tighten if there is filler.
- **$k$NN voting:** keep Eq. 1 and the tie rule. Say its role in one sentence: it measures how much
  label information the neighbors carry, its peak sets the largest k we give the LLM, and it labels
  invalid LLM outputs in the fallback. Keep the footnote on the other voting schemes if space allows.
- **RAG:** keep the prompt figure and the implementation details (prompt parts, 8,192-token limit,
  70/30 split, truncation, XML label, regex, invalid outputs, Unsloth, temperature). Remove the sentence
  that re-quotes 59.5% if §I already motivates it, or keep one short clause. Leave temperature and
  determinism wording to S5.
- **Filtered RAG: move the calibration here.** It is part of the method's definition and was fixed on a
  validation split before testing. It currently sits at the end of old RQ2 in `05_evaluations.tex`
  (validation split: 30 queries stratified by label and 270 indexed issues per project, mirroring PS;
  true questions retrieve nearly balanced bug and question neighbors, mean $N_{bug}-N_{question} =
  -1.36$ over $k\in[1,15]$; the sweep $m\in\{1,\dots,5\}$, capped at the 95th percentile; $m = 3$ fires
  for 89.1% of true questions and 56.7% of true bugs). Write the subsection as:
  1. the motivation as a hypothesis: RAG's most frequent error is questions labeled as bugs, known in
     prior work (keep the citations); the models show it even at zero-shot; we hypothesize that
     bug-labeled examples reinforce it;
  2. the rule: remove the bug-labeled examples when $N_{bug}-N_{question}\le m$;
  3. calibration on the validation split, with the numbers above;
  4. "otherwise identical to RAG".
  Delete the circular "the empirical motivations ... are fully detailed in Section ..." and delete the
  calibration paragraph from §IV.
- **LoRA fine-tuning baseline:** keep. Tighten the first paragraph, which says "faithfully reproduces
  the established methodology" twice.

## 4. Evaluation Metrics (`04_setup.tex`, that paragraph only)

S1 added the percent/points sentence; keep it. Make sure the paragraph also says that each method's
best k is chosen by test-set macro F1. Moving this definition here from the §IV opener is fine.

## 5. Evaluation (`05_evaluations.tex`): three RQs

Keep the floats and their order: Fig. `kcurves` and Table `results_master` at the top, the heatmap and
Table II in RQ3, and the cost table in RQ3. Relabel: the merged RQ1 → `sec:rq1`; old `sec:bragtag` →
`sec:rq2`; old `sec:method-comparison` → `sec:rq3`. Remove `sec:ragtag`. Then run
`grep -rn 'sec:ragtag\|sec:bragtag\|sec:method-comparison' SANER2027/sections/` and fix every reference
in every section.

**Opener:** at most 2 sentences (floats, and where "best" is defined). Delete the coauthor comment
blocks that describe old revisions (lines starting with `%` about 2026-09-19/21/22 decisions); they are
stale. Keep the generator comments.

### RQ1: How well does RAG classify issue reports, and how many examples does it need?

Merge the old RQ1 ($k$NN voting) and old RQ2 (RAG) into about three paragraphs and one answer box.
Roughly 350–400 words; the old sections were about 750 without the calibration.

1. **Two reference points.**
   - Zero-shot prompting (the LLM, k = 0) reaches 61.3–68.8%.
   - The $k$NN voting baseline (retrieval, no LLM), evaluated at $k\in\{1,\dots,20,25,30\}$, peaks at
     59.5% with the target project's issues (k = 15) and 60.4% with all projects' issues (k = 16). Its
     peak sets the top of the k grid for RAG.
   - Keep the fact that most neighbors come from the query's own project even with the pooled index
     (84–90% overlap), since it explains why the settings barely differ. Say it once.
2. **RAG beats both references everywhere.**
   - All 48 configurations with $k \ge 1$ (four sizes × six k values × two settings) outperform both
     references in macro F1, macro precision and macro recall.
   - Best k is 3, 6, 12 and 12, giving 69.7–76.7%, and nothing improves from 12 to 15.
   - Retrieving from the target project scores slightly higher than retrieving from all projects at
     almost every size and k, so the rest of the paper uses the target project's issues for retrieval.
3. **Where the gain comes from and what remains.**
   - The gain comes mostly from questions (the question-F1 share of the gain, from NUMBERS.md).
   - Questions labeled as bugs fall from 46–59% at zero-shot to 26–36%.
   - Question remains the weakest class, and question-to-bug errors are the largest source of false bug
     predictions. This leads to RQ2.
   Cut the per-model precision/recall averages ("bug precision increases from ... to ...") unless the
   argument needs one of them.

Answer box (at most 3 sentences, at most 3 numbers). Example: "RAG outperforms both zero-shot prompting
and $k$NN voting at every number of examples, reaching 69.7–76.7% macro F1 at its best k (3 to 12). The
gain comes mostly from questions, but 26–36% of questions are still labeled as bugs."

### RQ2: Does filtering the retrieved examples reduce question-to-bug errors?

About two paragraphs and a box, roughly 250–300 words; the old section was about 480.

1. **Macro F1.**
   - Filtered RAG scores higher than RAG at every $k\ge 6$ for all four sizes.
   - It scores lower at k ≤ 3, because with so few neighbors the rule always fires and can leave no
     examples at all (one sentence).
   - At each method's best k it is 1.5–2.4 points higher. The CIs of all four differences exclude zero;
     give them in a footnote together with the bootstrap description, or drop them if Table II's
     footnote already covers the method.
   - Its best k is larger (6, 12, 15, 12) because removing examples needs a larger initial k.
2. **The trade-off.**
   - Questions labeled as bugs fall to 19–27%, and question recall rises.
   - Bug precision rises but bug recall falls at every size, most at 3B, the only size whose bug F1
     drops (the over-correction).
   - Feature is almost unchanged.
   - Point to Table I for per-class values instead of listing ranges for all six P/R changes. Keep at
     most 3–4 numbers in this paragraph.

Answer box: at most 3 sentences.

### RQ3: Can RAG match LoRA fine-tuning with less labeled data and GPU memory?

This subsection carries the headline. Lead with it, then back it up, and keep the honesty guards
(BRIEF §5).

**Run-in structure** (use `\myparagraph` headings: *Same labeled issues*, *Pooled fine-tuning*, *Labeled
data, memory and time*, *Error profiles*, *Invalid outputs*):

(a) **Same labeled issues.**
- With the target project's 300 labeled issues, both RAG variants outperform fine-tuning at every size
  (RAG by 2.1–4.0 points, filtered RAG by 3.8–6.0).
- Fine-tuning gains substantially from pooling all 3,300 labeled issues (PA − PS range from NUMBERS.md,
  consistent with Heo and Lee), whereas retrieval does not (RQ1).
- So we compare each method in its better setting: RAG with the target project's issues, fine-tuning
  with the pooled issues.
- Keep the one-sentence caveat: the eleven project-specific indexes together hold the same 3,300 issues,
  so the settings differ in the labeled data available per target project, not in the total labeling
  effort.

(b) **Pooled fine-tuning (Table II).**
- Averaged over the four sizes, filtered RAG trails fine-tuning by 1.0 point and the CI includes zero;
  RAG trails by 2.8 points and the CI excludes zero.
- Per size, fine-tuning is significantly ahead at 7B and 14B; filtered RAG is ahead at 3B and 32B, but
  not significantly.
- **Robustness to the choice of k:** each method's best k is chosen on the test set. With a single k = 12
  for every size, filtered RAG trails fine-tuning by 1.2 points on average (NUMBERS.md fixed-k section).
- The per-project heatmap: one sentence (16 of 44 pairs, e.g. all four sizes on opencv; keep the
  draft's facts).

(c) **Labeled data, memory and time (Table III; this is the other half of the headline, so it comes
before the error analysis).**
- Filtered RAG needs 11× less labeled data per target project and no training.
- Peak GPU memory is 25–47% lower, 33% on average.
- Its inference is slower (0.22–4.15 h vs 0.10–0.44 h for the 3,300 test issues). Including training,
  total time is lower at 3B and 14B, similar at 7B, and higher at 32B.
- Fine-tuning must be repeated for new labels or a new model, while retrieval only indexes new issues.
- One clause on $k$NN voting: under 4 s and 0.2 GB at 59.5%.

(d) **Error profiles** (at most 5 sentences; the old paragraph was about 190 words):
- Fine-tuning over-predicts bug (39–46% of predictions vs 33% true bugs), with higher bug recall
  (0.86–0.93, now in %) and lower bug precision.
- Filtered RAG predicts bug for 31–38% and makes fewer false bug predictions at three of the four sizes
  (not at 14B).
- The question-to-bug confusion persists under both (23–36% vs 19–27%).
- Which profile is preferable depends on whether a missed bug or a false bug label costs a project
  more. Remove "A bug share close to the true proportion is not by itself evidence of quality; what
  matters is ...": state the trade-off directly.

(e) **Invalid outputs and the fallback** (one paragraph; the old text was three paragraphs, about 280
words):
- Invalid outputs are 1.8–4.7% for filtered RAG vs at most 0.8% for fine-tuning. They count as errors
  and explain most of filtered RAG's lower macro recall at 7B and 14B.
- When $k$NN voting labels the invalid outputs of both methods (no extra LLM call; the retrieval-based
  pipelines reuse their neighbors), filtered RAG is 0.1 point ahead of fine-tuning on average, the CI
  includes zero, and it is significantly ahead at 32B.
- Keep the fallback protocol detail (PS voting at k = 15 for RAG, PA voting at k = 16 for fine-tuning)
  as a short clause or in the Table II caption.

**Answer box** (at most 3 sentences, leading with the headline). Example: "With only the target
project's 300 labeled issues, no training and 33% less peak GPU memory, filtered RAG comes within 1.0
point of the macro F1 of fine-tuning on all 3,300 labeled issues, a difference that is not statistically
significant, while RAG trails by 2.8 points. Given the same labeled issues, both RAG variants outperform
fine-tuning. RAG's inference is slower, and fine-tuning finds more bugs but over-predicts the bug label."

## 6. Discussion (`06_discussion.tex`)

- **Failure analysis:** keep the method and the three categories with their examples. Replace
  two-decimal percents with counts and whole percents: 116 of 150 (77%), 20 (13%), 14 (9%). For the 60
  invalid outputs: 44 (73%), 14 (23%), 2 (3%). Check that these counts add up and match the draft
  percentages. Tighten the wording, and leave deeper tone fixes to S5.
- **Replace "Key Insights and Practical Implications"** with a subsection titled, for example,
  "Implications", in 2–3 run-in paragraphs. Each follows from a result and does not recommend who
  should use what (BRIEF §7 rule 6):
  1. *Labeled data and retraining.* A project that has only its own labeled issues reaches, without
     training, the macro F1 that fine-tuning reaches only with issues pooled from eleven projects (RQ3).
     New labels are an index update, and the model can be replaced without retraining. Fine-tuning keeps
     an advantage in inference time, most clearly at 32B.
  2. *Choosing between error profiles.* The approaches differ less in macro F1 than in which errors they
     make (RQ3). Fine-tuning finds more bugs; filtered RAG raises fewer false bug alarms.
  3. *Classification and templates.* Keep the existing "Combining classification and validation" idea
     (bug templates mislead classifiers; LLM-assisted triage on GitHub and GitLab; report validation),
     and tighten it.
  Remove the old "For extremely resource-constrained scenarios, even VOTAG PS is a defensible baseline"
  sentence (its content is in RQ3's cost paragraph), unless it fits in point 1 as one clause.
- Delete the commented-out Future Work block only if it is truly unused (the Conclusion has its own
  list).

## 7. Conclusion (`08_conclusion.tex`)

The user rejected an earlier full rewrite of the Conclusion ("changed too much"). Keep its structure:
¶1 what we did, ¶2 what we found, ¶3 trade-offs and future work. Replace the content with the new
names, the headline, and the same facts and numbers as the abstract (reusing the abstract's wording is
fine). ¶2 leads with the headline and gives at most 4 numbers. ¶3 keeps the future-work list, trimmed
to what fits. Delete the stale commented-out block at the top (it mentions TOST).

## 8. Page budget and consistency

1. Build and run pagecheck after each section. **Target: at least 45 free lines on page 10, and
   ideally 60** (S4 needs about 30 and S5 about 15; S5 then fills page 10 exactly, since the user wants
   10 full pages with references from page 11). If you are short, cut §IV prose further, never float
   readability or the honesty guards. Do not try to fill page 10 yourself.
2. **Consistency sweep**, because every claim appears in several places:
   - The headline facts (1.0 point and not significant; 11× per project; no training; 33%; RAG 2.8
     points) are identical in the abstract, §I RQ3 block, RQ3 answer box, Discussion and Conclusion.
   - RQ1/RQ2 numbers in §I match their answer boxes.
   - Every number in prose appears in NUMBERS.md. Write a small script that extracts numbers from the
     pdftotext of the main text and checks each against the set of values in NUMBERS.md; hand-check
     what it cannot match.
   - No paragraph has more than 4 numbers (the script can count).
   - `grep -rnE '[0-9]\.[0-9]{3}'` on the PDF text returns nothing.
   - No "PS"/"PA" in the abstract or §I.
   - No "accuracy", "TOST" or "equivalen".
3. Build with no undefined references. The only overfull boxes allowed are the two known `\balance`
   warnings.

## 9. Finish

Commit in logical steps (for example: abstract+intro; approach+setup; evaluation; discussion+conclusion),
with messages `SANER reframe S3: ...`, and push. Append your LOG.md entry with:
- the page state and the slack left for S4 and S5;
- the new section labels;
- sentences you were unsure about;
- any honesty guard you could not place;
- `Status: DONE`.
Tell the user, in a short message, the new abstract's word count, the slack, and anything they should
read first.

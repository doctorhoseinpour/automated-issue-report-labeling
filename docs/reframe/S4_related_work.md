# S4: rewrite Related Work to compare and contrast

**Read `docs/reframe/BRIEF.md` in full first, then `docs/reframe/LOG.md`.** S2 (dossier) and S3
(reframe) must both have logged `Status: DONE`. If not, stop and tell the user. Then read
`docs/reframe/RW_DOSSIER.md` and `docs/reframe/refs_candidates.bib` completely.

## Goal

Problem 2 from BRIEF, the supervisor's note: Related Work is "a good survey", but it should be used "to
differentiate yourself from prior work. Instead of just describing prior work, describe it and then
compare and contrast the approach with your own. This might be an opportunity to expand the paper a
bit."

After S4, **every paragraph of Related Work ends its description of prior work with an explicit
contrast to this study** on a concrete dimension, and where useful points to our result. A reader who
reads only Related Work should see what this paper adds to each line of work.

**Files you own:** `SANER2027/sections/02_related.tex` and `SANER2027/refs.bib`. You may add used entries
to `refs.bib`, and you may commit S2's two dossier files if S2 did not. Touch `01_intro.tex` only if the
novelty sentence must change to stay consistent with Related Work (see §4), and then change that
sentence only.

## 1. Before writing

1. **NOVELTY RISKS.** Read that section at the top of the dossier. If S2 found work that already
   evaluates retrieval-selected examples for IRC, or compares them with fine-tuning, **stop and ask the
   user** how to position the paper before writing anything. The "to our knowledge" sentence in §I and
   Related Work depends on it.
2. **De Vito et al. (TOSEM 2026).** Read S2's recommendation. The user decides whether it is cited. If
   the user is present, ask. If not, follow the user's earlier decision (not cited) and log the open
   question.
3. **Budget.** Run `pagecheck.py` and note the free lines on page 10. **Your budget is the free lines
   minus 15** (S5 needs about 15 for Threats and then fills page 10 exactly). Convert it to words: about
   10–11 words per column line. Current Related Work is about 780 words including comments; count the
   live text. Do not exceed the budget. If you have less room than the full outline needs, keep the
   contrast sentences and cut breadth, dropping the lowest-ranked papers from the dossier.

## 2. Structure

Related Work stays after the Discussion (user decision) and keeps run-in `\myparagraph` headings. Follow
the dossier's proposed outline; this skeleton is the default:

1. **Supervised issue report classification.**
   - Describe: classic ML (Antoniol et al., Ticket Tagger); fine-tuned encoders (CatIss/RoBERTa, seBERT,
     BERT-based NLBSE entries); NLBSE tool competitions; SetFit few-shot fine-tuning (`colavito2023few`,
     if verified).
   - Contrast: these methods train a classifier on the labeled issues. This study holds the LLM fixed
     and varies only how the labeled issues are used, as prompt examples or for LoRA weight updates. That
     is why its baseline is LoRA fine-tuning of the same models rather than a different classifier.
   - Make no claim that encoders are weaker or stronger (encoder results are not in the paper).
2. **LLMs for issue report classification.**
   - Describe zero-shot and few-shot prompting. Colavito et al. chose one or two random examples per
     class, which did not beat zero-shot prompting for most models.
   - Describe fine-tuning. **Fix the current inaccuracy:** the best reported LLM results come from GPT
     models fine-tuned through an API; LoRA is how prior work fine-tunes open models (Heo and Lee:
     Llama-3.1-8B; Aracena et al.: DeepSeek-R1-Distill). Use S2's quotes and wording.
   - Contrast (concrete, from the dossier): prior LLM studies use no or randomly chosen examples, do not
     vary the number of examples, do not compare a training-free approach with fine-tuning of the same
     model, and do not measure labeled-data needs or GPU memory. Heo and Lee suggest few-shot prompting,
     and Aracena et al. suggest retrieval augmentation, as future work. This study evaluates both
     together.
3. **Retrieval-selected in-context examples.**
   - Describe: NLP (KATE `liu2022makes`, `yu2023retrieval`, Rubin et al., `milios2023context`) and SE
     (CEDAR, Gao et al., log parsing `le2023log`, LLM-Cure `assi2026llm`, Judge the Votes
     `dincc2025judge`), only those S2 verified.
   - Contrast: other tasks and labels (code tasks, log parsing, app reviews, bug-report validity). This
     study brings retrieval-selected examples to three-label IRC; sweeps k up to where retrieval alone
     peaks; separates the contributions of retrieval and the LLM with $k$NN voting and zero-shot
     references (RQ1); and compares with LoRA fine-tuning including labeled data and memory (RQ3).
   - If `khandelwal2019generalization` (kNN-LM) or the $k$NN voting rule fits, one clause may connect
     $k$NN voting to nearest-neighbor classification (Cover and Hart, Dudani are already cited in §II).
4. **Fine-tuning versus in-context learning.**
   - Describe: Mosbach et al. (fair comparison on NLP tasks); Weyssow et al. (PEFT/LoRA vs in-context
     learning for code generation).
   - Contrast: those comparisons use NLP benchmarks or code generation, mostly with random or fixed
     examples (check the dossier), whereas this study compares on IRC with retrieved examples, with the
     same model and labeled issues on both sides, and reports labeled data per project and peak GPU
     memory. Point to the RQ3 result in one clause.
5. **Label bias in demonstrations.**
   - Describe: Zhao et al. (majority-label and recency bias; calibration); Min et al. (role of the
     demonstration labels); fairness-guided demonstration selection
     (`ma2023fairnessguidedfewshotpromptinglarge`).
   - Contrast: calibration adjusts output probabilities, and fairness-guided selection searches over
     demonstrations with extra LLM calls. Filtered RAG instead removes retrieved examples of one label
     based on the neighbors' label counts, with no extra LLM call, aimed at the question-to-bug
     confusion known in IRC (RQ2).
   - Phrase any link between our filter and "majority-label bias" as our motivation or hypothesis, not
     as a proven mechanism.
6. **Question-to-bug misclassification.**
   - Keep the existing paragraphs (two in the current file) and their argument pattern: a prior-work
     cause followed at once by our own observation with a pointer (BRIEF §7 rule 10).
   - Update the names and numbers to the new format from NUMBERS.md.
   - They already contrast well. Tighten them if the budget is short, and make sure they still say the
     confusion persists under every method we test and that retrieval reduces it relative to our own
     zero-shot baseline, never relative to prior work (BRIEF §5 guard 6).

**Order and flow.** Paragraphs 1–2 are IRC, 3–5 are techniques, and 6 is the error. Give each paragraph
one line of argument: what they do, what they found, how we differ, where our result is. Contrast
sentences should read naturally ("Unlike these studies, ..." at most once; vary it with "In contrast",
"This study instead ...", "We ...", or by putting the contrast in the subject). Never disparage prior
work. Never compare our numbers with numbers from other datasets or splits.

## 3. Citations

- **Cite only what the dossier verified.** Every statement about a paper must be supported by the
  dossier's facts; abstract-only facts support only abstract-level claims.
- Copy **only the entries you actually cite** from `refs_candidates.bib` into `SANER2027/refs.bib`. Keep
  the style consistent with existing entries (IEEEtran; braces around acronyms in titles, e.g.
  `{LLM}`, `{RAG}`, `{FAISS}`).
- Use "Heo and Lee" and "Aracena et al.", etc., and `~\cite{...}`.
- After building, check `SANER2027/SANER2027-main-labeling.blg` for bibtex warnings (missing fields,
  undefined entries) and fix them. Check the reference list in the PDF: new entries must render
  correctly (authors, venue, year). References must still end by page 12 (pagecheck).

## 4. Keep §I consistent

§I ¶3 says, "To our knowledge, however, retrieval-augmented few-shot prompting has not been
systematically evaluated for IRC or compared with LoRA fine-tuning." Related Work must say the same
thing in the same terms. If the dossier shows that a qualification is needed (for example, "with
retrieval-selected examples"), change both places consistently and log it.

## 5. Checks and finish

1. Build; there must be no undefined citations. Run `pagecheck.py`. **Leave about 15 free lines on
   page 10 for S5.** If you have more room left after all six paragraphs are complete, add the next
   most valuable verified contrast from the dossier, but stop at about 15 free lines.
2. Re-read the section in the PDF as a skeptical reviewer who knows the IRC literature.
   - Is every contrast concrete and true?
   - Is any prior work misdescribed?
   - Does any sentence overclaim novelty?
   Fix what you find.
3. Commit `02_related.tex`, `refs.bib` (and `01_intro.tex` only if changed) with the message
   `SANER reframe S4: related work with compare-and-contrast`. Push.
4. Append your LOG.md entry:
   - the page state and the free lines left for S5;
   - the new citations with keys;
   - the decisions on novelty and De Vito et al.;
   - any open question;
   - `Status: DONE`.
5. Tell the user in a short message: how many paragraphs and citations, and whether any decision needs
   them.

# S6: independent verification ("did the fixes work?")

**Read `docs/reframe/BRIEF.md` in full first, then `docs/reframe/LOG.md`.** S5 must have logged
`Status: DONE`. You are a fresh pair of eyes. Your job is to **judge**, not to rewrite.

**Scope rule from the user:** judge the text of the paper (the PDF, its tables and figures, and its
cited sources). Do not check statements against the implementation (code, scripts, data files). For
numbers, the reference is `docs/reframe/NUMBERS.md` and the paper's own tables.

You may fix **objective errors only**: typos, broken references, a number that disagrees with
NUMBERS.md or the tables, an old method name, a citation that renders wrongly, or a line or two to keep
page 10 exactly full. Report everything else as a finding with a proposed fix; the user decides.

## 1. Mechanics

- Build (`SANER2027/build.sh`). The PDF must have at most 12 pages, no undefined references or
  citations, and no overfull boxes except known `\balance` warnings on the reference pages.
- `python3 docs/reframe/pagecheck.py --final` passes: main text fills page 10 (at most 3 free lines) and
  the references start at the top of page 11. Look at pages 10 and 11 in the rendered PDF (use
  `pdftoppm -f 10 -l 11 -r 70 -png` and view the images) to confirm there is no visible gap and no
  stranded heading.
- Abstract at most 250 words (count "macro F1" as two words).

## 2. Problem 1: names

- `grep -rnE 'VOTAG|RAGTAG|BRAGTAG|\\votag|\\ragtag|\\bragtag'` over `SANER2027/sections`,
  `SANER2027/tables` and the PDF text: none, apart from comments in `.tex` files.
- Run `pdftotext` on each figure in `SANER2027/figures/` that the paper includes (check the
  `\includegraphics` lines) and grep for old names. If `fao.pdf` still shows them, S1's MANUAL STEP
  (diagram export) is still open; report it prominently.
- Count every all-caps token of 2+ letters in the PDF text (a short script) and list them with counts.
  The expected set is IRC, LLM(s), LoRA, RAG, CI, PS, PA, plus proper names (FAISS, GPU, GB, IEEE,
  NLBSE, GPT, BERT, ...). Flag anything else, and flag PS/PA in the abstract or Introduction.
- Check that each method has exactly one name throughout: prose, tables, figure legends and captions.

## 3. Problem 2: Related Work compares and contrasts

- For each Related Work paragraph, quote its contrast sentence(s) in the report. Flag any paragraph
  that only describes prior work.
- Diff the cited keys: `git diff saner-draft-2026-09-25 -- SANER2027/refs.bib` and the `\cite` keys
  used in `02_related.tex` before and after. For **every new reference**, fetch its DOI or DBLP page and
  confirm title, authors, venue and year. Report mismatches, and fix the BibTeX if the fix is certain.
- For every sentence that describes a cited paper, check that `docs/reframe/RW_DOSSIER.md` supports it
  (a fact with a page and quote, or marked abstract-level). Flag unsupported statements.
- Check that the "to our knowledge" novelty sentence matches between §I and Related Work.

## 4. Problem 3: framing and numbers

- **Headline consistency.** Extract the headline statement and its facts (1.0 point and not significant;
  11× less labeled data per project; no training; 33% less peak GPU memory; plain RAG 2.8 points behind)
  from the abstract, the §I RQ3 block, the §I closing sentence, the RQ3 answer box, the Discussion and
  the Conclusion. Put them in one table in the report. They must agree.
- **Does the paper lead with the headline?** Say whether a reader of only the abstract, the first page
  and the RQ3 answer box would come away with "matches fine-tuning with 11× less data per project, no
  training, a third less memory" rather than "slightly higher F1". Quote the sentences that decide it.
- **Honesty guards** (BRIEF §5): for each of the seven, give the sentence and section where it appears,
  or FAIL.
- **Numbers.** With a script, extract every number from the main-text PDF (excluding tables, references,
  equation labels, section numbers, k values, model sizes and years) and match each to NUMBERS.md or to
  a table value. List the unmatched ones and check them by hand. Also list:
  - any three-decimal number in the main text (should be none);
  - any signed number in prose;
  - any CI bracket outside Table II;
  - paragraphs with more than 4 numbers.
- **Content rules** (BRIEF §6): no accuracy, no TOST or equivalence wording, "Heo and Lee", no LoRA
  called the state of the art, no favorable comparison with prior work's question-class results.

## 5. Problem 4: tone and clarity

1. Grep the PDF text for the pattern list in `S5_tone_accuracy_threats.md` §2 and report the hits with
   context.
2. List sentences longer than 35 words (script), with section.
3. **Reviewer panel.** Launch three subagents in parallel. Give each the full PDF text (`pdftotext`, not
   layout mode) and ask each to return:
   - a verdict on the SANER scale (strong reject / weak reject / borderline / weak accept / strong
     accept);
   - the top 5 weaknesses;
   - **every sentence that reads machine-written, vague or ambiguous**, quoted, with a one-line reason.
   The personas:
   - **A: skeptical senior SANER PC member** who has published on issue classification and LLMs for SE,
     and who rejected the previous version for overclaiming. Focus: novelty, soundness of the
     comparison, overclaims.
   - **B: methods and statistics reviewer.** Focus: selection of k on the test set, CIs,
     multiple comparisons, fairness of the fine-tuning baseline, cost measurement, threats.
   - **C: SE researcher outside the area,** reading for clarity. Focus: can they follow the
     argument, the names and the numbers; where did they get lost; which sentences sound unnatural.
   The prompts must not reveal which problems were fixed. Ask for an honest review.
4. Merge the three reviews: deduplicate, and keep only findings about the paper's text and claims.

## 6. Change view for the supervisor

Make a latexdiff PDF from tag `saner-draft-2026-09-25` to HEAD.
- If `latexdiff` is not installed, download the `latexdiff` Perl script from CTAN
  (https://ctan.org/pkg/latexdiff; the `latexdiff-so` variant is self-contained) into your scratchpad.
- Export the tag's `SANER2027/` with `git archive saner-draft-2026-09-25 SANER2027 | tar -x -C <scratch>/old`.
- Run `latexdiff --flatten old/SANER2027-main-labeling.tex SANER2027/SANER2027-main-labeling.tex >
  diff.tex` from a copy of the current `SANER2027/` in your scratchpad. The generated tables have
  changed too; if latexdiff chokes on them, use `--exclude-textcmd` / `--config` options or fall back to
  diffing only `sections/`.
- Compile it in the scratchpad. Copy the result to `docs/reframe/changes_vs_draft.pdf` (commit it; it
  is for the supervisor).
- If latexdiff cannot be made to work in about 15 minutes, produce `git diff --word-diff
  saner-draft-2026-09-25 -- SANER2027/sections` as `docs/reframe/changes_vs_draft.txt` instead.

## 7. Report

Write `docs/reframe/REVIEW_REPORT.md`:

```
# Review report (S6)
## Verdict per problem          <- PASS / PARTIAL / FAIL for problems 1-4, one paragraph of evidence each
## Mechanics                    <- pagecheck --final output, pages, warnings, abstract word count
## Headline consistency table
## Honesty guards               <- 7 rows: guard, where, PASS/FAIL
## Numbers audit                <- unmatched numbers, 3-decimal hits, dense paragraphs
## Related Work audit           <- contrast sentences, new references verified, unsupported claims
## Reviewer panel               <- 3 verdicts, merged weaknesses, machine-sounding/ambiguous sentences
## Objective fixes I made       <- file, old -> new
## Remaining fixes, prioritized <- for the user; each with a proposed wording
## Open questions for the user
```

Commit `REVIEW_REPORT.md`, the change view, and any objective fixes (by explicit path) with the message
`SANER reframe S6: verification report`, and push. Append your LOG.md entry with `Status: DONE`. Tell
the user in a short message:
- the verdict per problem;
- the three reviewer verdicts;
- the top 3 remaining fixes;
- that they can bring `docs/reframe/REVIEW_REPORT.md` back to the planning session for a final check.

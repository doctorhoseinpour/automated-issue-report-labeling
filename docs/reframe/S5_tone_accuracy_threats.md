# S5: tone and clarity pass, text-level accuracy, Threats, and filling page 10 exactly

**Read `docs/reframe/BRIEF.md` in full first, then `docs/reframe/LOG.md`.** S4 must have logged
`Status: DONE`. If not, stop and tell the user.

**Scope rule from the user: work on the text of the paper only.** Do not check statements against the
implementation (`*.py`, shell scripts, experiment folders, data files). Judge each sentence by what the
paper itself says: its own results, tables, figures and cited sources. If a statement cannot be judged
from the paper and cannot be worded more carefully without new facts, leave it and note it in LOG.md.

## Goal

Problem 4 from BRIEF: "some sections of the paper were written by AI which sometimes makes them look
unnatural and read very badly and maybe not clearly or ambiguous. the wording and tone in those parts
must be fixed as well."

After S5:
- the whole paper reads as if one careful human researcher wrote it: plain, precise, specific, with no
  machine-sounding phrasing and no ambiguous sentences;
- the Threats to Validity section is a real section;
- page 10 is **exactly full**, with the references starting on page 11 (`pagecheck.py --final` passes).

**Files you own:** every file in `SANER2027/sections/`, and the captions in the table generators and
`method_cost.tex`. Change **wording, not claims or numbers**. The exceptions are the text-level accuracy
fixes in §3, and the Threats section, which you rewrite.

## 1. Read first, then list

1. Extract the PDF text (`pdftotext -layout`) and read the whole paper once, start to finish, as a
   reader seeing it for the first time.
2. In your scratchpad, list every passage that:
   - sounds machine-written;
   - is vague: says little, or uses abstract nouns instead of saying what happened;
   - is ambiguous: "this", "it", "they" or "these" without one clear referent; a sentence that can be read
     two ways; a term used for two things; two terms used for one thing;
   - is hard to follow: sentences over about 35 words, nested clauses, many numbers in one sentence.
3. Fix them section by section in paper order, rebuilding as you go.

## 2. Patterns to fix (check every section)

Machine-sounding patterns seen in this draft and similar ones:

- **Meta and roadmap sentences:** "In this section, we ...", "The remainder of this section is organized
  as follows", "as detailed below", "fully detailed in", "we elaborate on". Keep at most the Introduction's
  one-sentence roadmap. A section should start with its content.
- **Inflated or abstract wording:**
  - "expose the trade-off", "a targeted reduction", "cohesive workflow", "underexplored",
    "emphasizes the importance of", "offer a practical solution", "faithfully reproducing the
    established", "sheds light", "plays a crucial role", "a wide range of", "various", "leverage" (use
    "use"), "utilize", "facilitate", "robust" (unless tested), "comprehensive", "novel";
  - "notably", "importantly", "crucially", "interestingly", "it is worth noting that".
- **Stock constructions:**
  - "Beyond X, ...";
  - "what matters is ...";
  - "X is not by itself evidence of Y";
  - "not only ... but also";
  - "This is not X; it is Y";
  - "Thus, <restatement of the previous sentence>".
- **Rhetorical triplets and em-dash asides.** Lists of three used for rhythm, not content.
- **Paragraph-ending summaries** that repeat what the paragraph just said. Delete them unless the
  paragraph is long and the sentence adds the implication.
- **Stacked hedges** ("may potentially suggest"). Use one hedge, or none if the result is measured.
- **Defensive or reviewer-answering phrasing** (BRIEF §7 rule 7).
- **The "claim: explanation" colon pattern** (BRIEF §7 rule 8).
- **Vague quantities** where the paper has the number ("substantially", "considerably"). Use the number
  from NUMBERS.md or drop the adverb, within the number diet (at most 3–4 numbers per paragraph).
- **Terminology drift** from BRIEF §4.1: neighbors vs examples vs demonstrations; query vs target issue;
  setting vs data scope; "fine-tuning" vs "LoRA" vs "Fine-Tune"; "RAG-based methods" vs "retrieval-based
  methods" vs "both RAG variants". Pick the BRIEF term and apply it everywhere, captions included.
- **Tense:** past tense for what we did ("we evaluated", "we sampled"), present tense for what the
  results show and what tables contain ("filtered RAG trails fine-tuning by 1.0 point"). Be consistent
  within each section.
- **Awkward leftovers from renaming** (S1 and S3 logged some): "the \knn", "\frag's" at a sentence
  start, "RAG and \frag" where "both RAG variants" reads better.

Keep the user's own phrasing where it is already clear. The goal is natural text, not text rewritten
for its own sake. If a sentence is clear, precise and correct, leave it.

## 3. Accuracy fixes that can be made from the text alone

Each of these can be checked against the paper itself or its citations; do not open code or data.

1. **"Heo et al." → "Heo and Lee"** everywhere (the bib entry has two authors).
2. **Claims about prior work:** the best reported LLM results for IRC come from GPT models fine-tuned
   through an API, and LoRA is how prior work fine-tunes open models (BRIEF §6). Fix any remaining
   sentence that calls LoRA fine-tuning the state of the art, for example in the Approach's fine-tuning
   subsection ("Recent IRC approaches have achieved strong performance by supervised fine-tuning ...
   using LoRA").
3. **"The temperature is set at 0.1 for deterministic classifications":** a temperature of 0.1 is low
   but not deterministic. Reword without new facts, e.g. "a low temperature (0.1), so that outputs vary
   little between runs".
4. **Internal consistency.** Every claim that appears in more than one place (abstract, §I, answer boxes,
   §V, §VIII) must say the same thing with the same number. Every "Table X shows ..." must match what
   the table actually shows. Every section reference must point to the right section.
5. **Honesty guards (BRIEF §5):** confirm each is present where its claim is made. Add a missing one in
   the shortest form.

## 4. Threats to Validity (`07_threats.tex`)

The current section has three short paragraphs and one vague sentence ("Assessing how far our findings
generalize ... warrants its own dedicated studies"). Rewrite it into four run-in paragraphs. Each threat
gets one or two sentences: the threat, then what we did about it or what it means for the reader. Use
only facts stated elsewhere in the paper. Target 15–25 lines.

- **Internal validity.**
  - Each method's best k is chosen by test-set macro F1. With a single k = 12 for every model size,
    filtered RAG still comes within 1.2 points of pooled fine-tuning on average (this is already in RQ3;
    refer to it).
  - The filter's margin was calibrated on a validation split drawn from the training issues, not on the
    test set, and is the same for all models and projects.
  - The fine-tuning prompt and hyperparameters follow prior work; other choices could change either
    approach's results (cite as in the current text).
  - Invalid outputs count as errors; the fallback analysis shows their effect.
- **Construct validity.**
  - Macro F1 on a balanced three-label test set; per-class precision and recall are reported so that
    trade-offs are visible.
  - Some ground-truth labels are wrong (label noise was 9% of the sampled errors in the failure
    analysis), which affects all methods alike.
  - Keep the existing sentence about LLM randomness and repeated runs as the paper states it; only
    improve its wording.
- **Conclusion validity.**
  - Differences are tested with paired bootstrap CIs that resample issues.
  - Per-size comparisons are not corrected for multiple comparisons, so single per-size results should
    be read with care, whereas the headline uses the average over sizes.
- **External validity.**
  - One model family (Qwen2.5-Instruct, 4-bit, four sizes);
  - one embedding model;
  - one benchmark of eleven GitHub projects with balanced labels;
  - results may differ with imbalanced labels, other issue trackers or other model families.
  - Keep the cited sentence on writing-style variation across projects.

No new claims about experiments that the paper does not report. Delete the commented-out lines.

## 5. Fill page 10 exactly (user requirement)

The user wants exactly 10 full pages of main text, with references starting on page 11 (a `\clearpage`
before the bibliography forces the break).

1. After the tone pass and Threats, run `python3 docs/reframe/pagecheck.py`.
2. **If free lines remain on page 10,** fill them with substance, in this order (BRIEF §8), and never with
   filler or repetition:
   1. a sharper compare-and-contrast sentence in Related Work supported by `RW_DOSSIER.md`;
   2. a missing Threats item;
   3. a result the reader needs that was cut (for example, one per-class number that makes a trade-off
      concrete);
   4. a clarifying sentence in the Discussion's implications.
3. **If the text runs past page 10,** cut the weakest sentence (a repetition, a paragraph-ending summary,
   or a number the table already shows), not a float.
4. Iterate until `python3 docs/reframe/pagecheck.py --final` passes: page 10 full (at most 3 free lines)
   and references starting at the top of page 11. Look at page 10 in the PDF to confirm that the right
   column ends at the bottom margin with no visible gap, and that no heading is stranded at the bottom
   of a column.

## 6. Final sweep

- The abstract is at most 250 words (count "macro F1" as two words).
- Captions match the text and use the new names and units, and every float is referenced in the text.
- There is no `\fixme`, "TODO" or stray comment text in the PDF.
- Double-blind: no author names or affiliations; the only URLs are GitHub issue links and the anonymous
  replication package; any self-citations are in the third person.
- The build has no undefined references or citations and no overfull boxes other than the two known
  `\balance` warnings (check whether they still occur; with the references now starting on page 11 they
  may change).

## 7. Finish

Commit in steps (for example: tone §I–§IV; tone §V–§VIII; Threats; final fill), with messages
`SANER reframe S5: ...`, and push. Append your LOG.md entry:
- the sections changed, with the kinds of fixes;
- the list of passages you rewrote for being machine-sounding or ambiguous, with old → new for the most
  important 10;
- the `pagecheck.py --final` output;
- open questions;
- `Status: DONE`.
Tell the user in a short message that the draft is ready for S6.

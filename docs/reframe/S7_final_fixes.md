# S7: final fixes from the S6 review (user-approved list only)

**Read `docs/reframe/BRIEF.md` in full first, then the S6 entry in `docs/reframe/LOG.md`, then
`docs/reframe/REVIEW_REPORT.md`.** S6 must have logged `Status: DONE`.

This is the last editing session before submission. Apply **exactly** the fixes listed below, and
nothing else: no new content, no new papers, no restructuring. Work on the paper's text only (BRIEF §3
scope rule). The page layout must stay final: page 10 full, with at most 3 free lines, and References
starting at the top of page 11. Run `python3 docs/reframe/pagecheck.py --final` after every step.

## User decisions (2026-09-25; they override BRIEF §5's headline wording)

1. **Headline verb.** Replace "matches" and "preserves" with **"comes within 1.0 point of" / "comes
   within about one point of"** wherever the headline is stated. Do the same for any other "matches
   fine-tuning", "parity" or "preserves the macro F1" wording.
   - Known places: the abstract's last sentence, the §I "Overall" sentence, contribution 2 (which also
     needs a subject: "it" has no antecedent), the Conclusion ¶2's first sentence, §IV-C "so the parity
     in aggregate does not hold for every project", and §V "so the parity does not depend on tuning k
     per model".
   - Use REVIEW_REPORT "Remaining fixes" item 1 for the proposed wordings.
   - Grep the source for `match`, `preserv` and `parity` and fix every headline use.
   - The title stays. "Comes within one point" answers its question honestly.
2. **Threats, Construct validity: add one sentence.** Fine-tuning's peak memory includes training;
   serving an already trained model needs less memory, so the saving is in producing the classifier, not
   in serving it. Make it one sentence, fitted to the paragraph's wording.
3. **Keep unchanged:**
   - the "three runs, averaged" sentence;
   - the Unsloth sentence (no "4-bit");
   - the filter's "suspected question" wording;
   - everything that needs the lab machine.

## Wording fixes from REVIEW_REPORT (apply all; each is a small local edit)

- **Item 3.** §IV-C fixed-k check: delete the vacuous "and it remains ahead at Qwen-32B" (32B's best k is
  already 12). Do not add a replacement.
- **Item 5.** Repetition:
  - delete "Neither approach removes the question-to-bug confusion." at the end of §V *Error profiles*
    (the next paragraph says it);
  - rename §V's *Error profiles* heading to *Different errors*.
  These cuts pay for decision 2.
- **Item 6.** "labeled between two training runs" (§IV-C and §V) → "classified between two training runs".
- **Item 7.** §IV-C "Fine-tuning's significant leads fall at the two sizes …" → "… occur at the two sizes …".
- **Item 8.** §V "with 0.2 instead of 2.9 GB of GPU memory" → "with 0.2 GB of GPU memory, against 2.9 GB for
  RAG at Qwen-3B" (or equivalent wording that is correct).
- **Item 9.** Conclusion: "Two earlier results lead to this one." → "This result builds on two others." or
  equivalent.
- **Item 10.**
  - §V "Prior studies validate bug reports~\cite{…}" → "Prior studies reproduce or validate bug
    reports~\cite{…}".
  - §VIII: "such as chain-of-thought prompting~\cite{koyuncu2025exploring}" → "such as the
    chain-of-thought prompting used for bug reports~\cite{koyuncu2025exploring}".
  - Leave `huang2025back` and `akhavan2026…` as they are.
- **Item 11.** §I "Fine-tuning pulls ahead only with all 3,300 labeled issues …" → "Fine-tuning scores
  higher only with all 3,300 labeled issues …". It must still read correctly next to "not statistically
  significant" for filtered RAG.
- **Item 13.** The Table III caption says "at the configurations of Table II" → "Table I".
- **Abstract.**
  - "must be repeated for new labels or a newer model" → "must be repeated for newly labeled issues or a
    newer model" (so "labels" is not read as new label classes).
  - Fix the dangling subject in "…a difference that is not statistically significant, and needs no
    training and 33% less peak GPU memory on average": filtered RAG is the subject, so split or recast the
    sentence.
  - Keep the abstract at 250 words or fewer ("macro F1" = two words).
- **Also** (reviewers A, B and C): "the only size at which the filter over-corrects" (§IV-B). Say what
  happens instead, e.g. "the only size at which the filter costs more bug F1 than it gains elsewhere", or
  whatever matches the paragraph's facts.

## Finish

1. Build, and run `pagecheck.py --final`: it must pass. If the edits freed lines, fill them only with the
   next REVIEW_REPORT item that the user approved. None is left, so shorten nothing further; instead,
   restore one sentence of substance that S5 or S6 cut, or, as a last resort, let up to 3 lines stay free.
   If they overflow, cut a repeated sentence.
2. Grep the PDF text: no "matches fine-tuning", "preserves", "parity" (in the headline sense), no
   three-decimal numbers, no old method names.
3. Regenerate the change-view PDF against tag `saner-draft-2026-09-25` the same way S6 did (see S6's LOG
   entry) and overwrite `docs/reframe/changes_vs_draft.pdf`.
4. Commit by explicit path, with the message `SANER reframe S7: final fixes from the review`, and push.
   Append a LOG entry listing each fix (old → new) and the final pagecheck output, ending with
   `Status: DONE`.
5. Tell the user in 3–4 lines: the fixes are done, the page state, and that the PDF is ready at
   `SANER2027/SANER2027-main-labeling.pdf`.

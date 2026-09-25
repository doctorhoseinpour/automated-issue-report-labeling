# Review report (S6)

Scope: the text of the paper (PDF, tables, figures, cited sources). Numbers checked against
`docs/reframe/NUMBERS.md` and the paper's tables. No checks against code or data.
Build checked: HEAD of `saner-reframe` after the S6 fixes listed below.

## Verdict per problem

**Problem 1 (names): PASS.**
- No old name (VOTAG, RAGTAG, BRAGTAG, `\votag`, `\ragtag`, `\bragtag`) appears in the PDF text.
- In `SANER2027/sections/` and `SANER2027/tables/`, old names remain only in `.tex` comments (`method_cost.tex` header) and
  in legacy tables that are not `\input` (`method_comparison_ext.tex` and others; S1 already logged them).
- Figures included by the paper (`fao.pdf`, `kcurves.pdf`, `per_project_diff.pdf`) have no old names.
  `app_ov.pdf` still says VOTAG/RAGTAG/BRAGTAG, but its figure sits inside `\begin{comment}` and is not in the PDF.
- Each method has one name in prose, tables, legends and captions: $k$NN voting, RAG, filtered RAG, fine-tuning, zero-shot.
- All-caps tokens in the main text are all in the expected set (list under Mechanics). No PS/PA in the abstract or §I.
- Residual clarity point from reviewer C: "RAG" also reads as retrieval-augmented *generation*, the index terms say
  "Retrieval-augmented generation", and the title/conclusion use the long form for the family while "RAG" is the unfiltered
  variant (see Remaining fixes, item 9).

**Problem 2 (Related Work compares and contrasts): PASS.** All six blocks end with a contrast to this study and a pointer
to our result (quoted under Related Work audit).
- The four newly cited references check out against Crossref, ACL Anthology and NeurIPS proceedings.
- Every description of a cited paper that I checked is supported by `RW_DOSSIER.md`.
- The novelty sentence has the same content in the abstract, §I and §VI.

**Problem 3 (framing and numbers): PASS, with one open risk.**
- The abstract, the §I RQ3 block, the contributions, the RQ3 box, the Discussion and the Conclusion all lead with the
  headline, and they agree on every number (table below).
- All seven honesty guards are present.
- Numbers:
  - no three-decimal numbers, no signed numbers and no CI brackets in prose;
  - every prose number is in NUMBERS.md or the tables, or is a setup count or prior-work figure verified by S3/S4.
- The open risk: all three simulated reviewers attack "matches" / "preserves" as an equivalence claim drawn from a
  non-significant difference (CI up to 2.0 points behind, significant deficits at 7B and 14B). The two skeptical ones
  vote weak reject mainly for this reason. The wording is a user decision (BRIEF §5), so I did not touch it; see Remaining
  fixes, item 1.

**Problem 4 (tone and clarity): PARTIAL.**
- The S5 pattern list is almost clean. The only hit is one "thus" restatement in RQ3.
- All three reviewers independently flag:
  - formula-level repetition: the headline sentence appears about seven times, and "the question-to-bug confusion
    persists" four times;
  - a few vague or unnatural phrases: "preserves", "pulls ahead", "fall at", "Two earlier results lead to this one",
    "join both strengths" (fixed), "labeled" used for "classified by the model".
- Reviewer C followed the argument and understood the headline from page 1, but got lost in about 20 specific places
  (merged list below).

## Mechanics

- **Build:** `SANER2027/build.sh` gives 12 pages and no undefined references or citations. The only overfull box is the
  known `Overfull \vbox (1.73193pt too high) ... while \output is active` from `\balance` on the reference pages.
- **`python3 docs/reframe/pagecheck.py --final`** (exit 0):
  ```
  Main text ends: page 10, right column, y=696 of 719  (last words: '...this paper.')
    free lines before the end of page 10: 2 body lines
  References start: page 11, left column, y=52
  References end: page 12, lowest line y=204 of 719
    room left before the page-12 limit: about 86 body-line equivalents
  OK: within 10 + 2 pages.
  FINAL TARGET MET: page 10 is full and the references start on page 11.
  ```
- **Visual check:** pages 10 and 11 were rendered and viewed. Page 10's right column ends with Data Availability near
  the bottom margin: no visible gap and no stranded heading. Page 11 starts with REFERENCES at the top.
- **Abstract:** 248 words ("macro F1" counted as two words, "3,300" and "6,600" as one each).
- **All-caps tokens in the main text (count):**
  - RAG 154, LLM 55, LoRA 29, PS 28, GPU 21, PA 18, LLMs 16, IRC 15, CI 7, FAISS 5, GB 5, CIs 4, BERT 4, NLBSE 4,
    API 3, XML 2, NVIDIA 2, VRAM 2, RTX 1, GPT 2, NLP 2, MLP 1 (quoted issue title), VS 1 (VS Code).
  - Proper names: GitHub, GitLab, TypeScript, RoBERTa, SetFit, MiniLM, OpenAI, DeepSeek, AdamW.
  - Roman numerals and small-caps headings.
  - Nothing unexpected.

## Headline consistency table

| Place | F1 gap, filtered RAG vs pooled FT | Significance | Labeled data | Training | GPU memory | Plain RAG |
|---|---|---|---|---|---|---|
| Abstract | "comes within 1.0 point" | "not statistically significant" | "target project's 300"; "11× less labeled data per project" | "needs no training" | "33% less peak GPU memory on average"; "a third less" | "trails fine-tuning by 2.8 points" |
| §I RQ3 block | "leads filtered RAG by 1.0 point on average" | "not statistically significant" | "Fine-tuning pulls ahead only with all 3,300 labeled issues" (no 11×) | "need no training" | "33% less peak GPU memory on average" | "leads RAG by 2.8 points" |
| §I "Overall" | "preserves the macro F1" (no number) | – | "less labeled data per project" | "no training" | "less GPU memory" | – |
| §I contribution 2 | "matches fine-tuning's macro F1" | – | "11× less labeled data per project" | "no training" | "a third less GPU memory" | – |
| RQ3 box | "comes within 1.0 point" | "not statistically significant" | "only the target project's labeled issues" | "no training" | "33% less peak GPU memory on average" | "trails by 2.8 points" |
| §IV-C text | "trails fine-tuning by 1.0 point" | "the CI includes zero" | "11× less labeled data per target project" | "no training phase" | "33% less peak GPU memory on average" | "2.8 points, and the CI excludes zero" |
| §V Implications | "within 1.0 point" | "not statistically significant" | "11× as many issues pooled"; "saving is per project" | "without training" | "33% less peak GPU memory" | "trails by 2.8 points" |
| §VIII Conclusion | "matches" + "trails ... by 1.0 point" | "not statistically significant" | "11× less labeled data per project" | "no training" | "33% less peak GPU memory on average" | "trails by 2.8 points" |

All numbers agree. The verbs vary: "comes within", "preserves" (abstract last sentence and §I Overall), and "matches"
(contribution 2 and Conclusion). The verb is the item the reviewers attack.

**Does the paper lead with the headline?** Yes. A reader of the abstract, page 1 and the RQ3 box comes away with "within
1.0 point of fine-tuning, not significant, with the target project's issues only, no training, 33% less memory". They do
not come away with "slightly higher F1". The deciding sentences:
- Abstract: "Using only the target project's 300 labeled issues, filtered RAG comes within 1.0 point of fine-tuning on
  all 3,300 labeled issues of the eleven projects in macro F1, a difference that is not statistically significant, and
  needs no training and 33% less peak GPU memory on average."
- Abstract: "With this filter, retrieval-augmented few-shot prompting preserves the macro F1 of LoRA fine-tuning with 11×
  less labeled data per project, no training, and a third less GPU memory."
- RQ3 box: "With only the target project's labeled issues, no training, and 33% less peak GPU memory on average, filtered
  RAG comes within 1.0 point of the macro F1 of fine-tuning on the labeled issues of all eleven projects, a difference that
  is not statistically significant; RAG trails by 2.8 points."

Reviewer C's one-paragraph summary after page 1 confirms this reading.

## Honesty guards

| # | Guard | Where (sentence) | Result |
|---|---|---|---|
| 1 | Plain RAG does not match | Abstract "Without the filter, RAG trails fine-tuning by 2.8 points."; §IV-C "RAG trails by 2.8 points, and the CI excludes zero"; RQ3 box; §V; §VIII | PASS |
| 2 | Per size: FT significantly ahead at 7B/14B | §IV-C "Fine-tuning is significantly ahead of filtered RAG at Qwen-7B and Qwen-14B, while filtered RAG is ahead at Qwen-3B and Qwen-32B, but not significantly."; §V "by 2.4 and 2.9 points"; §VIII | PASS |
| 3 | k selected on the test set; fixed k = 12 gives 1.2 | §III-C "the one with the highest macro F1 on the test set"; §IV-C "Each method's best k is chosen on the test set ... trails fine-tuning by 1.2 points on average instead of 1.0"; §VII Internal Validity | PASS |
| 4 | 11× is per target project | Every "11×" carries "per project" / "per target project"; §IV-C "the settings differ in the labeled data available per target project, not in the total labeling effort"; §V "The saving is per project"; §VII | PASS |
| 5 | RAG is not faster | Abstract "RAG's inference is slower"; §IV-C "The longer prompts make inference slower ... and more at Qwen-32B"; §V; §VIII | PASS |
| 6 | Question-to-bug confusion is known; no favorable comparison with prior work | §I cited clause; §VI closing paragraphs; reductions stated only against our own zero-shot and RAG | PASS |
| 7 | Memory is per size, RAG inference vs FT training + inference | §IV-C "comparing their inference with fine-tuning's training and inference of the same model"; hardware in §III-D | PASS |

## Numbers audit

- **Script:** it extracted every number from the prose of the included sections (tables, figures, equations, k values,
  model sizes and years removed) and matched each against NUMBERS.md and the table sources.
- **Unmatched numbers (checked by hand, all fine):**
  - setup counts: 6,600; 600; 270; 8,192; 94.9% (from the draft, per S3's log);
  - 77% = 116/150 (failure analysis);
  - prior-work figures verified by S2/S4: 800,000 (Siddiq and Santos); about 200 (SetFit); 46–54% (Colavito et al.).
- **Three-decimal numbers in the main text:** none.
- **Signed numbers in prose:** none.
- **CI brackets outside Table II:** none.
- **Paragraphs with more than 4 numbers or ranges** (a range counts as one):
  - abstract (7; S3 accepted this for the headline);
  - §I RQ3 block (5);
  - §II-E calibration paragraph (8 setup values);
  - §II-F hyperparameters (setup values);
  - §IV-C *Labeled data, memory, and time*, second paragraph (5: 3,300; 0.10–0.44 h; 0.22–4.15 h; 4 s; 0.2 GB);
  - §V *Labeled data and retraining* (5: 33%, 1.0, 11×, 2.8, 3.8–6.0);
  - §V-A invalid-output sample (8 counts and shares).
  - The results paragraphs otherwise stay at 4 or fewer.
- **Spot checks against Table I and NUMBERS.md** (all correct):
  - RQ1: 5.5–8.7; 56–70%; 76–84%; 74–79%.
  - RQ2: 6.5–15.3; 1.2–5.0; 80.3→66.5; 1.8; 1.1.
  - RQ3 and error profiles: 2.1–4.0; 3.8–6.0; 3.1–8.4; 39–46%; 86.5–92.9 vs 66.5–80.4; 1.8–4.7% and 0.8%; within 1.4;
    6–8%.
  - Cost: 47/31/25/29%; the time totals.
  - Heatmap: 16 of 44 pairs (recounted from the figure); opencv 4/4, bitcoin 3/4.
- **Reviewer-visible rounding.** All three reviewers recomputed ranges from Table I and got 0.1-point mismatches (5.6
  vs 5.5, 1.4 vs 1.5, 8.5 vs 8.4, 4.1 vs 4.0, 6.1 vs 6.0; Table II −5.4 vs 73.2 − 78.5 = −5.3). The prose follows BRIEF
  §4.2 (exact values). I added "computed from unrounded scores" to the §III-C sentence that defines points (no line
  cost).
- **Heatmap check.** Two reviewers read Fig. 4's column labels as k = 6/12/12/15 in the text extraction. The rendered
  figure shows 6/12/15/12, which is correct.
- **Open data point:** the ansible / Qwen-3B heatmap cell is exactly 0.000 (lab machine; unchanged since S1).

## Related Work audit

**Contrast sentences (one or more per block):**
1. *Supervised IRC:* "This study instead holds the LLM fixed and varies only how the labeled issues are used, as examples
   in the prompt or as training data for LoRA, so its baseline is LoRA fine-tuning of the same models rather than a
   different classifier. Unlike SetFit's encoder, ours is not trained and serves only to retrieve the neighbors, whose
   labels the LLM then sees as examples."
2. *LLMs for IRC:*
   - "They average the scores of the eleven projects, whereas we score the test issues of all projects together, so our
     scores are not directly comparable with theirs."
   - "This study evaluates both suggestions together. We show the LLM the labeled issues most similar to the query, up to
     15 of them, and compare this with LoRA fine-tuning of the same four open models on the same labeled issues, in both
     of Heo and Lee's settings, including peak GPU memory."
   - "Pooling the eleven projects' issues helps fine-tuning, as Heo and Lee found, but not RAG, …"
3. *Retrieved in-context examples:*
   - "This study takes both steps for three-label IRC across eleven projects. It sweeps k up to where a vote over the
     neighbors' labels peaks and filters the examples … compares with LoRA fine-tuning of the same 3B to 32B
     instruction-tuned LLMs."
   - "In our study, RAG trails fine-tuning on the pooled issues by 2.8 points; with the example filter, the gap shrinks to
     1.0 point and is not statistically significant."
4. *Label bias:* "Filtered RAG instead keeps per-query retrieval and applies a count rule to the neighbors' labels,
   removing the bug-labeled examples for suspected questions without an extra LLM call or access to output probabilities."
5. *Question-to-bug:* "In our study, retrieved examples lower the share … and filtered RAG lowers it further to 19–27%
   without an extra classification stage, but neither removes the confusion."
6. *Explanations:* "Our models favor the bug label even before they see any labeled issue …"; "Our failure analysis finds
   a related cause …"

No block only describes prior work. The Logan et al. sentence in block 3 has no contrast of its own, but its block does.

**Cited keys, tag → HEAD:**
- Added: `colavito2023few`, `dettmers2023qlora` (§II-F), `milios2023context`, `siddiq2022bert`.
- Removed: `le2023log`.
- No reference is new to `refs.bib` (user decision).

**Verification of the newly cited references:**

| Key | Source checked | Result |
|---|---|---|
| `colavito2023few` | Crossref 10.1109/NLBSE59153.2023.00011 | Title, authors, NLBSE 2023, pp. 16–19: match |
| `siddiq2022bert` | Crossref 10.1145/3528588.3528660 | Title, authors, NLBSE'22 workshop, pp. 33–36, 2022: match |
| `milios2023context` | ACL Anthology 2023.genbench-1.14 | Title, authors, GenBench 2023, pp. 173–184, ACL, Singapore: match |
| `dettmers2023qlora` | NeurIPS 2023 proceedings page | Title, authors, NeurIPS 36 (2023): match. The title **rendered as "Qlora: … llms"**; fixed (below) |

**Rendering fixes in `refs.bib`** (casing only, no content change): GPT/LLMs, API, GitHub (×2), RoBERTa/BERT, seBERT,
QLoRA/LLMs, LLM-Assisted, GPT-3, AssertFlip/LLM, and LoRA in the replication-package title.

**Support in the dossier** (checked for each cited-paper description):
- Supported:
  - Milios et al. (frozen encoder, up to 150 labels, neutral class rarely retrieved, "the retriever may be limiting the
    performance");
  - Ma et al. (entropy on a content-free input, extra LLM calls);
  - Dinç and Tüzün (MiniLM + FAISS, k = 5, Firefox validity, fine-tuned RoBERTa best, future work "sweep k" and
    "relevance filters");
  - Colavito et al. IST (random 1–2 examples per class, no improvement for most models; cost as time and hardware, not
    GB; 46–54%);
  - SetFit on about 200 issues; Siddiq and Santos 800,000;
  - Heo and Lee's per-project averaging (A6.4); Aracena et al.'s two remedies; Logan et al.; LLM-Cure's five fixed
    examples.
- Slight stretches (not errors):
  - "the authors name a sweep over k and filtering the neighbors as future work" / "This study takes both steps".
    Dinç and Tüzün's filter would remove *mislabeled or off-topic* neighbors, whereas ours removes by label. Two
    reviewers call "both steps" vague.
  - "These studies report cost as training time, inference time, or API prices, not as GPU memory". Colavito et al. do
    name GPU classes (2 vs 4 A100 64 GB), though not measured peak memory.
- Citations that reviewers A, B and C say do not support their sentence (outside Related Work; not in the dossier's
  scope):
  - `huang2025back` (issue–commit linking) for "works well on issue report text" (§II-B);
  - `akhavan2026linkanchor…` for writing-style variation (§VII);
  - `koyuncu2025exploring` for chain-of-thought (§VIII);
  - `khatib2025assertflip`, `wang2024aegis`, `ahmed2025otter` (bug reproduction / test generation) for "Prior studies
    validate bug reports" (§V).

**Novelty sentence:** the same content in all three places.
- §I: "To our knowledge, however, retrieval-augmented few-shot prompting has not been systematically evaluated for IRC or
  compared with LoRA fine-tuning."
- §VI: "… has not been systematically evaluated for IRC or compared with LoRA fine-tuning before."
- Abstract: "… for this task or compared with Low-Rank Adaptation (LoRA) fine-tuning."

## Reviewer panel

The three personas got the full `pdftotext` of the PDF and a neutral brief; the prompts did not say which problems had
been fixed.

| Persona | Verdict |
|---|---|
| A: skeptical senior SANER PC member (IRC and LLM4SE) | **Weak reject** ("borderline or weak accept" if the parity claim becomes a bounded difference and k is chosen on the validation split) |
| B: methods and statistics | **Weak reject** ("could move to weak accept" with validation-chosen k, a stronger FT baseline and reworded equivalence claims) |
| C: SE researcher outside the area, clarity | **Borderline** (argument followable; numbers checked and correct to 0.1; headline wording overclaims) |

**Merged weaknesses** (deduplicated; text and claims only; the number of reviewers raising each is in brackets):
1. **"Matches/preserves" rests on non-significance** [A, B, C]. The CI [−2.0, +0.1] allows a 2-point deficit. FT is
   significantly ahead at 7B and 14B. A mean over four sizes is not a quantity anyone deploys. Filtered RAG matches or
   beats FT in only 16 of 44 project pairs. The CIs resample issues, not projects.
2. **Test-set selection** [A, B, C]:
   - What is chosen on the test set: best k, the PS/PA choice for RAG ("scores slightly higher … so we use PS"), the top
     of the k grid (the kNN peak), and the footnoted choices (voting scheme, 8,192 tokens).
   - The filter's target error was observed on the test set; only m is validated.
   - The fixed-k = 12 check has no CI, and "it remains ahead at Qwen-32B" is vacuous, since 32B's best k is already 12.
3. **Fine-tuning baseline strength** [A, B, C]:
   - hyperparameters not tuned;
   - about 56 optimizer steps in PS (300 / 16 × 3), so "300 issues are too few" may mean under-trained;
   - no anchor to Heo and Lee's numbers;
   - 32B FT (77.1) below 14B FT (78.5).
4. **"Given the same labeled issues, both RAG variants outperform fine-tuning"** is true only in PS, since in PA the
   issues are also the same and FT wins [A, B]. *Partly fixed:* now "the same 300 labeled issues" in the abstract, RQ3
   box, contribution 2 and Conclusion, as §I already said. The CIs for the PS comparison are still missing.
5. **Memory comparison uses FT's training peak** [A, B, C]. A deployed fine-tuned model needs only inference memory,
   which is not reported. RAG and filtered RAG have identical memory at different k (A, B, C ask why). The hardware
   differs (L40 or L40S; 14B across GPUs).
6. **Protocol details missing or misplaced** [A, B, C]:
   - quantization (4-bit) is never stated, though §VIII mentions "quantization settings";
   - "we ran the experiments three times and report the average" appears only in Threats and does not say how three
     runs enter a single bootstrap or whether FT was retrained;
   - which runs used L40 vs L40S;
   - the split type;
   - no inter-rater agreement for the failure analysis.
7. **No encoder baseline (RoBERTa/SetFit)** [A, B, C]. It is out by user decision (BRIEF §4.4); the reviewers still ask.
8. **Filter ablations** [A, B]: removing bug examples vs fewer examples vs a label-prior shift; a random-example control;
   m that scales with k; behaviour on imbalanced data.
9. **Failure-analysis arithmetic** [A, C]. Each sample has 30 true-bug errors in total. "Wrong template" (116) holds only
   questions and features, so all 30 true-bug errors must fall in hybrid intent (20) + label noise (14) = 34, leaving 4
   non-bug errors there. Reviewers find this implausible or want it discussed.
10. **Repetition** [A, B, C]: the headline sentence about seven times, the confusion statement four or more times, and
    "Error profiles" as a heading in both §IV-C and §V-B.

**Memory across sizes (user question 3).** No reviewer raised the specific point that fine-tuning's best score overall
(78.5% at Qwen-14B, 16.7 GB) needs less memory than filtered RAG's best (78.1% at Qwen-32B, 22.3 GB). A and B raised a
related one: the 33% compares against fine-tuning's *training* peak, and a fine-tuned model deployed for inference needs
less. C asked why RAG and filtered RAG have identical memory.

**Sentences flagged as machine-written, vague or ambiguous** (merged; the number of reviewers flagging each is in
brackets):

Abstract
- "must be repeated for new labels or a newer model" [A, B, C]: "new labels" reads as new label classes.
- "…a difference that is not statistically significant, and needs no training and 33% less peak GPU memory on average."
  [B, C]: the subject of "needs" dangles.
- "…preserves the macro F1 of LoRA fine-tuning…" [A, B, C]: wrong verb, and it restates the sentences before it.
- "fine-tuning finds more bugs but over-predicts the bug label" [B, C]: colloquial, and it recurs about five times.

§I
- "Two reference points separate what retrieval and the LLM each contribute." [B, C]
- "The vote alone reaches 59.5% macro F1, so the neighbors carry label information that the LLM can use." [A, B, C]:
  the "so … can use" does not follow.
- "when bug-labeled neighbors do not clearly outnumber question-labeled ones" [A, B]: "clearly" stands in for m = 3.
- "Fine-tuning pulls ahead only with all 3,300 labeled issues" [B, C]: informal, and it clashes with "not significant".
- "Overall, … preserves the macro F1 …" [A, B, C]: restatement.
- Contribution 2: "Evidence that, with a simple filter on its examples, it matches …" [A, B, C]: "it" has no antecedent.

§II
- "which is small and fast and works well on issue report text [26]" [A, B, C]: vague, and the citation is off-topic.
- "Its peak sets the largest k we give the LLM …, and it labels invalid LLM outputs in the fallback" [C]: "the fallback"
  is not yet defined.
- Footnote 2, "In preliminary experiments, 8,192 tokens gave the best trade-off …" [A, B, C]: data unstated.
- "chose m=3 as the best trade-off between firing for questions and for bugs" [A, B, C]: criterion unstated.
- "m is a small positive margin" [A, B, C].
- "set a low temperature (0.1), so that outputs vary little between runs" [A, B, C]: why not greedy decoding?
- "Examples that exceed their share are truncated in proportion to their length …" [A, B]: per-example share unclear.
- "since LLMs use information at the end of a long context better than in its middle" vs descending-similarity order [A]:
  the least similar example sits next to the query.

§III
- "four sizes of one family let us study the effect of model size alone" [A, B, C]: "alone" overclaims.
- "one NVIDIA L40 or L40S GPU" [A, B, C].

§IV
- "labels 46–59% of questions as bugs but far fewer as features" [A, B, C]: ambiguous.
- "so low question recall and low bug precision are two sides of the same error" [A, C].
- "All 48 configurations … outperform both reference points in macro F1, macro precision, and macro recall (Fig. 3b)"
  [A, C]: Fig. 3b shows only F1, and the comparison baseline is unclear.
- "We hypothesize that larger models make better use of long contexts with many examples." [A, B, C]
- "which suggests that the smallest model follows the remaining examples most closely" [A, B, C].
- "the only size at which the filter over-corrects" [B, C]: "over-corrects" is undefined.
- "The filter thus turns a significant deficit into a difference that is not significant." [A, B, C]
- "Fine-tuning's significant leads fall at the two sizes where pooling helps it most, by more than 8 points" [A, B, C]:
  "fall" reads as "decrease".
- "so the parity in aggregate does not hold for every project" [A, B, C]: "parity".
- "although the RAG prompts may be four times as long as fine-tuning's" [A, B, C]: "may", and the "although" logic.
- "Conversely, the more issues are labeled between two training runs …" (also §V) [B, C]: "labeled" here means
  "classified by the model".
- "most of them continue the issue body or the model's reasoning" [C]: the setup has no reasoning step.
- "and it becomes significantly ahead at Qwen-32B" [C]: awkward.

§V
- "Fine-tuning keeps three advantages." [B, C]: formulaic, and it repeats the heading.
- "so the parity does not depend on tuning k per model" [A, B, C].
- "Neither approach removes the question-to-bug confusion." directly followed by the paragraph "Question-to-bug
  confusion: The confusion persists …" [A, B, C].
- "with 0.2 instead of 2.9 GB" [C]: 2.9 GB is RAG's Qwen-3B memory; zero-shot memory is not reported.
- "This matters for the LLM-assisted triage now offered on platforms such as GitHub and GitLab, where the reporter
  describes the issue to an LLM that suggests a template and labels." [A, B, C]: a product claim backed only by product
  pages.
- "Prior studies validate bug reports [34]–[37], and a triage workflow could combine such validation with
  classification." [A, B, C]: [34]–[36] are about bug reproduction.
- "so they limit the macro F1 that any classifier can reach on this benchmark" [A, B].

§VI
- "This study evaluates both suggestions together." / "This study takes both steps …" [A, B, C]: a formulaic pattern.
- "we hypothesize that they learned in pretraining that bug reports outnumber questions" [A, C].

§VII
- "other choices could change the results of either approach" [A, B, C]: boilerplate.
- "LLM outputs can vary between runs, so we ran the experiments three times and report the average." [A, B, C]: see
  weakness 6.
- "so a single per-size result should be read with care" [B, C].

§VIII
- "Two earlier results lead to this one." [A, B, C]
- "may join both strengths" [A, B, C]: fixed.
- "such as chain-of-thought prompting [51]" [C]: the citation is not the origin of chain-of-thought.

## Objective fixes I made

| File | Old → new | Why |
|---|---|---|
| `refs.bib` | `Qlora: Efficient finetuning of quantized llms` → `{QLoRA}: … {LLMs}`; likewise `{GPT}-like {LLMs}`, `{API}`, `{GitHub}` ×2, `{RoBERTa}`/`{BERT}`, `{seBERT}`, `{LLM}-Assisted`, `{GPT-3}`, `{AssertFlip}`/`{LLM}`, replication package `{LoRA}` | Titles rendered in lower case ("Qlora", "gpt-like llms", "github", "sebert", …) |
| `05_evaluations.tex` (§IV-C, Invalid outputs) | "while its macro recall is lower at Qwen-7B and Qwen-14B, where …" → "while its macro recall is lower at every size, most at Qwen-7B and Qwen-14B, where …" | Table I: filtered RAG's R_mac is lower at all four sizes (70.8/71.2, 72.9/76.4, 74.2/78.5, 76.7/78.0); flagged by A and B |
| `05_evaluations.tex` (§IV-C, per project) | "while fine-tuning leads at all four sizes on TypeScript." → "… on TypeScript and tensorflow." | Fig. 4: tensorflow is also negative at all four sizes (−0.1, −5.3, −1.4, −2.6); A called the sentence selective |
| `08_conclusion.tex` | "Since fine-tuning finds more bugs and filtered RAG more questions, combining retrieval with a fine-tuned model may join both strengths." → "Since the two approaches make different errors, combining retrieval with a fine-tuned model may combine their strengths." | Table I: filtered RAG's question recall is *below* fine-tuning's at 7B and 14B (55.8/58.8, 57.9/71.0); "join both strengths" was flagged by all three reviewers |
| `04_setup.tex` (§III-C) | "differences between methods in percentage points (points)." → "…(points), computed from unrounded scores." | All three reviewers found 0.1-point mismatches between prose differences and Table I arithmetic |
| `00_abstract.tex`, `05_evaluations.tex` (RQ3 box), `08_conclusion.tex` | "Given the same labeled issues, both RAG variants outperform …" → "Given the same 300 labeled issues, …" | Without "300", the claim is false in the pooled setting (same issues, fine-tuning wins); §I already said "the same 300 labeled issues" |
| `01_intro.tex` (contribution 2) | "beats fine-tuning when both use the same labeled issues" → "… the same 300 labeled issues" | Same reason |

None of these changed the line count. The page state is identical to S5's (2 free lines, final target met).

**User answer (1), "three runs, averaged": KEEP.** The sentence stays unchanged. Keeping it frees no line, so there was
no freed line to fill. Page 10 still has 2 free lines, within the "at most 3" target. A candidate for one of those lines
is item 4 below (a Threats item that A and B both ask for). I did not add it, because its wording touches the headline
and is the user's call.

**User answer (2):** Table III header stays "Mem. (GB)".

**User answer (3):** memory across sizes is unchanged; see the Reviewer panel for what the reviewers raised.

## Remaining fixes, prioritized

1. **Soften "preserves" / "matches" where no numbers follow** [A, B, C; the main reason for both weak rejects].
   Occurrences: abstract last sentence, §I "Overall", contribution 2, Conclusion ¶2 first sentence.
   - Proposed abstract last sentence: "With this filter, retrieval-augmented few-shot prompting comes within about one
     point of LoRA fine-tuning's macro F1 with 11× less labeled data per project, no training, and a third less GPU
     memory."
   - §I Overall: "preserves the macro F1 of" → "comes close to the macro F1 of".
   - Contribution 2: "it matches fine-tuning's macro F1" → "retrieval-augmented few-shot prompting comes within 1.0 point
     of fine-tuning's macro F1".
   - Conclusion: "matches the macro F1 of LoRA fine-tuning while using" → "comes within 1.0 point of LoRA fine-tuning's
     macro F1 while using".
   - The title question can stay: the paper answers it.
2. **State the quantization** [A, B, C]. §II-D: "We load the models with Unsloth to reduce memory use" → "We load the
   models in 4-bit precision with Unsloth to reduce memory use". The user must confirm 4-bit (bnb-4bit per the project
   notes, not the paper). There is no line cost if it fits.
3. **The fixed-k check** [A, B]. Drop the vacuous "and it remains ahead at Qwen-32B" (§IV-C). Optionally replace it with
   "and trails at three of the four sizes" (NUMBERS §4: −0.2, −2.4, −3.1, +1.0). A CI for the fixed-k mean needs the
   lab machine.
4. **Threats: test-informed filter design and FT's inference memory** [A, B]. Possible one-line fill for page 10 (see
   the note on user answer 1), in Internal Validity after the margin sentence: "The filter itself targets RAG's most
   frequent error on the test set (§IV-A), which prior work also reports." And in Construct Validity: "Fine-tuning's peak
   includes training; serving a trained model needs less memory." Each costs about 1 line, so only one fits without a
   cut.
5. **Repetition** [A, B, C].
   - Delete "Neither approach removes the question-to-bug confusion." at the end of §V *Error profiles*; the next
     paragraph says it.
   - Rename §V's *Error profiles* heading to *Different errors* so it is not a duplicate of §IV-C's.
   - Either change frees about 1 line, which can pay for item 4.
6. **"labeled" for model output** [B, C]. §IV-C "the more issues are labeled between two training runs" and §V "the
   number of issues labeled between two training runs": "labeled" → "classified".
7. **"fall at"** [A, B, C]. §IV-C "Fine-tuning's significant leads fall at the two sizes where pooling helps it most" →
   "Fine-tuning's significant leads occur at the two sizes where pooling helps it most".
8. **Zero-shot memory** [C]. §V "with 0.2 instead of 2.9 GB of GPU memory" → "with 0.2 GB of GPU memory, against 2.9 GB
   for RAG at Qwen-3B".
9. **Conclusion connective** [A, B, C]. "Two earlier results lead to this one." → "This result builds on two others."
10. **Citations the reviewers dispute** [A, B, C].
    - §V "Prior studies validate bug reports~\cite{khatib…, wang…, ahmed…, dincc…}" → "Prior studies reproduce or
      validate bug reports~\cite{…}".
    - §VIII: cite chain-of-thought's origin next to `koyuncu2025exploring`, or write "such as the chain-of-thought
      prompting used for bug reports~\cite{koyuncu2025exploring}".
    - §II-B `huang2025back` and §VII `akhavan2026…`: keep only if their text supports the sentence.
11. **§I "pulls ahead"** [B, C]. "Fine-tuning pulls ahead only with all 3,300 labeled issues of the eleven projects." →
    "Fine-tuning scores higher only with all 3,300 labeled issues of the eleven projects."
12. **The filter's firing rate vs "suspected questions"** [A, B, C]. The rule fires for 56.7% of true bugs on validation
    (§II-E). Consider "when the neighbors do not point clearly to bug" instead of "when the neighbors suggest a
    question" in §I, the abstract and §II-E. This is a wording choice for the user.
13. **Table III caption** [C]. "at the configurations of Table II" → "at the configurations of Table I" (Table I lists
    k; Table II only describes it in its note).
14. **Items needing the lab machine** (not text fixes):
    - the heatmap's 0.000 cell;
    - Table I Qwen-3B RAG vs filtered RAG having identical feature P/R (79.4/81.3) at k = 3 vs 6 (A and B suspect a copy
      error; NUMBERS.md has the same values from the CSV);
    - CIs for the same-data (PS) comparison and the fixed-k check;
    - fine-tuning's inference-only memory.

## Open questions for the user

1. Remaining fix 1 ("preserves"/"matches" → "comes within"): this changes the headline wording that BRIEF §5 fixed.
   Apply it?
2. Remaining fix 2: confirm the models are 4-bit quantized so §II-D can say so.
3. Remaining fixes 4 and 5: add one Threats item (paid for by the repetition cut)? Which one: test-informed filter design,
   or FT's training-inclusive memory peak?
4. The "three runs, averaged" sentence is kept as you decided. All three reviewers ask where it belongs and how three runs
   enter the bootstrap. Do you want it moved to §III-C with one clause on how the runs are combined (you need to supply
   the fact)?
5. The failure-analysis category counts (all 30 true-bug errors inside hybrid + noise): please confirm against the
   annotation sheet, since two reviewers find it implausible.

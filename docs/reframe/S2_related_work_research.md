# S2: Related Work research dossier (research only; no paper edits)

**Read `docs/reframe/BRIEF.md` in full first, then `docs/reframe/LOG.md`.** You are S2. You run in
parallel with S1, which is editing the paper in the same folder. **Do not edit anything under
`SANER2027/`, `scripts/` or `paper/`.** Your only outputs are two new files:

- `docs/reframe/RW_DOSSIER.md`: verified facts about prior work and how this study differs from each
  paper.
- `docs/reframe/refs_candidates.bib`: verified BibTeX entries for references not yet in
  `SANER2027/refs.bib`.

Commit only these two files, by explicit path. S4 uses them to rewrite Related Work.

## Why this matters

The supervisor's note (problem 2 in BRIEF): Related Work is "a good survey", but it should
"differentiate yourself from prior work. Instead of just describing prior work, describe it and then
compare and contrast the approach with your own." A contrast is only as good as the facts behind it. A
wrong claim about someone else's paper is worse than none, because the reviewer may be its author.
Hence the strict verification rules below.

## Our study, in contrast-ready terms

Use this list to write the "How we differ" lines.

- **Task:** three-label IRC (bug, feature, question) on Heo and Lee's balanced benchmark: 11 GitHub
  projects × 600 issues, 300 labeled and 300 test per project.
- **Approach:** retrieval-augmented few-shot prompting (RAG). A sentence-embedding model
  (all-MiniLM-L6-v2) and FAISS retrieve the k most similar labeled issues, which become labeled examples
  in a chat prompt. There is no training.
- **k sweep:** k ∈ {0, 1, 3, 6, 9, 12, 15}, bounded by where a retrieval-only $k$NN vote peaks (k = 15–16).
- **Two references** separate the contributions of retrieval and of the LLM: zero-shot prompting (the
  LLM without retrieval) and $k$NN voting (retrieval without the LLM).
- **Filtered RAG:** a label-count rule on the retrieved neighbors removes the bug-labeled examples when
  the query looks like a question. It needs no extra LLM call and is calibrated on a validation split of
  the training data.
- **Baseline:** LoRA fine-tuning of the **same** four Qwen2.5-Instruct models (3B–32B, 4-bit) on the
  **same** labeled issues, per project (PS) or pooled over all eleven projects (PA).
- **Measured:** per-class precision and recall, macro F1, paired bootstrap CIs, **peak GPU memory**,
  **wall-clock time**, and **labeled data per target project**.
- **Headline:** filtered RAG matches pooled LoRA fine-tuning in macro F1 (1.0 point behind, CI includes
  zero) with 11× less labeled data per project, no training, and 33% less peak GPU memory. Plain RAG is
  2.8 points behind.

## 1. Read what we already have

- `SANER2027/sections/02_related.tex` (the current Related Work), `01_intro.tex`, and `SANER2027/refs.bib`.
- `docs/research/QUESTION_BUG_PRIOR_WORK.md`: an earlier verification report with verbatim quotes and
  page numbers for many IRC papers. Reuse its verified facts and do not re-verify them.
- `docs/SANER_REVISION_PLAN.md` §2 (reviewer C's request to justify similarity-based selection) and the
  "§2 Related Work" entry in `paper/TODO.md` (Judge the Votes, LLM-Cure).

## 2. Local PDFs (read the relevant pages with the Read tool, using the `pages` parameter)

| File in `docs/` | Likely bib key |
|---|---|
| `A_Study_on_Applying_Large_Language_Models_to_Issue_Classification.pdf` | `heo2025study` (Heo and Lee) |
| `Applying Large Language Models to Issue Classification: Revisiting.pdf` | `aracena2025applying` |
| `Benchmarking large language models for automated labeling_ The case of issue report classification.pdf` | `colavito2025benchmarking` |
| `Impact of data quality for automatic issue classification using pre-trained.pdf` | `colavito2024impact` |
| `Issue classification with LLMs_ An empirical study of the NASA flight software systems.pdf` | **not in refs.bib**; decide whether to add it |
| `JudgetheVotes.pdf` | `dincc2025judge` |
| `LLM-Cure.pdf` | `assi2026llm` |
| `Log Parsing with Prompt-based Few-shot Learning.pdf` | `le2023log` |
| `NoTrainingWheels_ICML2025.pdf` | check relevance to the filter (bias correction at inference time?) |
| `CAA_rimsky_2024.pdf` | probably irrelevant (activation steering); confirm and skip |

For each relevant paper, extract **with page numbers and short verbatim quotes**:
- task, labels and dataset (and its overlap with our 11 projects, if any);
- method: is there training, and how? Prompt examples: none, random, fixed, or retrieved? How many?
  How were they selected?
- models used, and whether open models were fine-tuned with LoRA;
- best reported result and which configuration produced it;
- whether they measure cost (GPU memory, time, money) or labeled-data needs;
- what they propose as future work, especially few-shot prompting or retrieval augmentation (Heo and
  Lee and Aracena et al. are known to);
- anything that contradicts a claim in our current Related Work or Introduction.

Known fact to confirm and quote: prior work's best LLM results come from GPT models fine-tuned through
an API, while LoRA was used for open models (Heo and Lee: Llama-3.1-8B with 4-bit PEFT/LoRA; Aracena et
al.: DeepSeek-R1-Distill with Unsloth and LoRA). The current Related Work says "state-of-the-art
LLM-based methods ... fine-tune the LLM with LoRA", which is inaccurate. S4 must fix it, so give S4 the
accurate wording with quotes.

## 3. Entries already in refs.bib but not cited

Grep `\cite{...}` in `SANER2027/sections/*.tex`, ignoring commented lines, to list the uncited keys. For
each uncited key relevant to Related Work, say what the paper is and which paragraph it could support.
Candidates:
- `colavito2023few` (few-shot IRC with SetFit, NLBSE'23?);
- `panichella2023summary` and `kallis2024nlbse` (NLBSE tool competitions);
- `vargovich2023givemelabeledissues`;
- `milios2023context` (in-context learning for many-label classification with retrieval?);
- `ma2023fairnessguidedfewshotpromptinglarge` (demonstration selection for fairness);
- `khandelwal2019generalization` (kNN-LM);
- `sclar2023quantifying` (prompt-format sensitivity; useful for Threats);
- `dettmers2023qlora`;
- `joulin2016bag` (fastText, the NLBSE baseline);
- `siddiq2022bert`, `gomes2023bert`;
- `wei2021finetuned`.
Verify each entry's metadata as in §5 if you plan to recommend it. Existing entries can be wrong too.

## 4. New candidate references (add only if verified)

Read at least each abstract, and more when a contrast depends on method details.

1. Nashid, Sintaha, Mesbah, "Retrieval-Based Prompt Selection for Code-Related Few-Shot Learning",
   ICSE 2023 (CEDAR). This is the closest SE precedent for retrieval-selected examples.
2. Gao et al., "What Makes Good In-Context Demonstrations for Code Intelligence Tasks with LLMs?",
   ASE 2023.
3. Zhao, Wallace, Feng, Klein, Singh, "Calibrate Before Use: Improving Few-Shot Performance of Language
   Models", ICML 2021 (majority-label and recency bias of demonstrations). This is directly relevant to
   why removing bug-labeled examples can help.
4. Min et al., "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?", EMNLP 2022.
5. Mosbach et al., "Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation",
   Findings of ACL 2023.
6. Weyssow et al., "Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large
   Language Models", TOSEM. It compares PEFT/LoRA with in-context learning in SE; check whether their
   in-context baseline uses retrieved examples.
7. Rubin, Herzig, Berant, "Learning To Retrieve Prompts for In-Context Learning", NAACL 2022.
8. Tunstall et al., "Efficient Few-Shot Learning Without Prompts" (SetFit), only if the supervised-IRC
   paragraph mentions SetFit.

Do a **web search for other close work** too: LLM-based issue/bug-report/ticket classification or
triage with few-shot or retrieved examples (2023–2026); retrieval-augmented or kNN-based in-context
learning for text classification; fine-tuning vs in-context learning comparisons in SE. Add at most
about 5 more strong candidates. Quality beats quantity: S4 has room for roughly 6–10 new citations.

## 5. Novelty check (highest priority; do this first)

The paper claims, "to our knowledge", that retrieval-augmented few-shot prompting has not been
systematically evaluated for IRC or compared with LoRA fine-tuning.

1. Search for work that retrieves labeled issues as prompt examples for issue report classification or
   closely related issue tasks, 2023–2026. Use search terms such as "retrieval augmented issue
   classification", "in-context learning issue report classification", "few-shot LLM bug report
   classification retrieved examples", "RAG issue labeling GitHub", and "kNN demonstration selection bug
   report".
2. **De Vito et al., TOSEM 2026, doi 10.1145/3815577** (few-shot LLM IRC with Qwen2.5-32B). Fetch the DOI
   page and any accessible full text (arXiv or the authors' page). Find out how they select examples
   (random? fixed? retrieved?), how many, which labels and data, and whether they compare with
   fine-tuning. The user earlier chose to set this paper aside. **Do not decide for the user.** Give a
   clear recommendation in the dossier ("cite and contrast in paragraph X with this sentence" or "leave
   out because ..."), with the facts.
3. Put everything that threatens the novelty sentence at the **top of the dossier** under "NOVELTY
   RISKS", with the exact claim at risk and a proposed rewording. If nothing is found, say so and list
   what you searched.

## 6. Verification protocol (mandatory for every reference you recommend)

- Get BibTeX from DBLP (`https://dblp.org/search?q=...`, or the API
  `https://dblp.org/search/publ/api?q=...&format=json`, then the record's `.bib` link), the publisher's
  DOI page, the ACL Anthology, or arXiv. **Never write a BibTeX entry from memory.**
- Confirm the title, full author list, venue, year and pages against the fetched source, and record the
  URL you used in a comment line above each entry in `refs_candidates.bib`.
- Use keys in the style of `refs.bib` (`firstauthorYEARfirstword`, for example `nashid2023retrieval`).
  Check for collisions with existing keys.
- **Never state a fact about a paper that you did not read.** Mark facts taken only from an abstract as
  "(abstract)". If a paper is paywalled and only the abstract is available, the contrast sentence may
  rely only on the abstract.

## 7. Dossier format (`docs/reframe/RW_DOSSIER.md`)

```
# Related Work dossier (S2)
## NOVELTY RISKS            <- first, even if empty (then say what was searched)
## De Vito et al. (TOSEM 2026): facts and recommendation
## Accuracy fixes for the current Related Work / Introduction   <- e.g. the LoRA-SOTA sentence, with quotes
## Papers
### <bibkey> (in refs.bib | NEW)
- What it is (one line):
- Verified facts (page, quote):
- How we differ (1-2 sentences, concrete):
- Paragraph: <1-6 from the outline below>
- Source checked: <URL or local PDF + pages>
...
## Proposed outline for Related Work
## Open questions for the user
```

"How we differ" must name a **concrete dimension**, not a vague "we focus on". For example:
- they train or fine-tune, while we use labeled issues as prompt examples without training;
- random, fixed or per-class examples vs examples retrieved by similarity;
- a single number of examples vs a sweep of k bounded by a retrieval-only baseline;
- no comparison with fine-tuning of the same model vs LoRA fine-tuning of the same four models on the
  same labeled issues;
- no cost measurement vs peak GPU memory, time and labeled data per project;
- a different task or labels (code generation, log parsing, app reviews, bug validity) vs three-label
  IRC;
- output calibration or extra LLM calls vs a label-count rule on retrieved neighbors with no extra call.

Never disparage prior work, and never claim superiority over results obtained on other data.

## 8. Proposed outline (end of the dossier)

Propose 5–6 run-in paragraphs for Related Work. For each give the heading, the papers it cites, **the
exact contrast sentence(s)** you propose, and a word estimate. Suggested skeleton (change it if the
evidence says otherwise):

1. *Supervised issue report classification*: classic ML, encoders, NLBSE competitions, SetFit few-shot.
   Contrast: each trains a classifier; this study holds the LLM fixed and varies only how the labeled
   issues are used (as prompt examples or for weight updates), which is why the baseline is LoRA
   fine-tuning of the same models. Make no claim about encoder performance.
2. *LLMs for IRC*: zero-shot and few-shot prompting with random examples (Colavito et al.: no gain over
   zero-shot); fine-tuning (best results from GPT fine-tunes; LoRA for open models). Contrast: no
   retrieved examples, no k sweep, no training-free alternative on the same model, no cost measurement;
   Heo and Lee and Aracena et al. name few-shot prompting and retrieval augmentation as future work.
3. *Retrieval-selected in-context examples* in NLP and SE (KATE `liu2022makes`, `yu2023retrieval`,
   Rubin et al., CEDAR, Gao et al., log parsing `le2023log`, LLM-Cure `assi2026llm`, Judge the Votes
   `dincc2025judge`, `milios2023context`). Contrast: task and labels; a k sweep bounded by $k$NN voting;
   retrieval-only and LLM-only references; comparison with LoRA fine-tuning, including cost.
4. *Fine-tuning vs in-context learning* (Mosbach et al.; Weyssow et al.). Contrast: NLP benchmarks or
   code generation with random or fixed examples, vs IRC with retrieved examples and measured labeled
   data and memory.
5. *Label bias in demonstrations* (Zhao et al.; Min et al.; fairness-guided prompting). Contrast: those
   use output calibration or selection that needs extra LLM calls; our filter changes which retrieved
   examples are shown using their labels, with no extra call, and targets the question-to-bug confusion
   known in IRC.
6. *Question-to-bug misclassification*: the existing paragraph stays; note any new supporting evidence
   you found.

Current Related Work: about 780 words including comments; count the live text. S4's budget depends on
the page slack S3 leaves, likely +300 to +450 words. Rank the papers by importance so S4 can cut from
the bottom.

## 9. Finish

Commit `docs/reframe/RW_DOSSIER.md` and `docs/reframe/refs_candidates.bib` only, with the message
`SANER reframe S2: related-work dossier`, and push. Append your LOG.md entry (commit it too, but only
LOG.md in a separate `git add docs/reframe/LOG.md`; if S1 is committing at the same moment and git
reports an index lock, wait a few seconds and retry). Tell the user in a short message:
- whether any novelty risk was found;
- the De Vito et al. recommendation;
- how many new references you verified.

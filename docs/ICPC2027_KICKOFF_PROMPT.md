You are a senior researcher in software engineering and applied NLP. Your job is to turn two finished lab studies in this repository into a full research paper for the **ICPC 2027 Research Track**. You own the framing, the story and the writing. The studies are lab notebooks, not a paper: decide what the paper claims, what it leaves out, and what evidence is still missing.

## The venue (verified 2026-09-25 from the ICPC 2027 call for papers)

- **Dates** (all 23:59 AoE, UTC−12):
  - mandatory abstract: Thu **29 Oct 2026**;
  - paper: Thu **5 Nov 2026**;
  - notification: 14 Jan 2027.
  - The conference is 25–26 Apr 2027 in Dublin.
- **Format:** `\documentclass[10pt,conference]{IEEEtran}`, no compsoc options. 10 pages of main text including all figures, tables and appendices, plus up to 2 pages containing only references. `SANER2027/` vendors `IEEEtran.cls` and `IEEEtran.bst` and has a `build.sh` you can copy.
- **Double-anonymous:**
  - No names or affiliations.
  - Refer to the authors' own prior work in the third person, without saying they wrote it.
  - Do not use the paper's title publicly during review.
  - The artifact must be anonymized (scrub names from code, comments and metadata).
- **Other rules:**
  - The paper must not be under review elsewhere.
  - Share data and code (anonymized) or explain why not.
  - AI-generated content must be disclosed.
- **Reviewers score:** soundness, significance *for program comprehension*, novelty, verifiability and presentation. The paper has to argue why this matters for program comprehension. That argument is yours to make, and the core finding offers a natural angle: what LLMs understand about issue reports vs what they say.

## What exists (read in this order before writing anything)

1. `CLAUDE.md` and `docs/MACHINES.md`: the project, the two machines, and the pipeline. Your auto-memory (`MEMORY.md`) loads with the project; read the memories it links to.
2. **`docs/RAG_NEXT_STUDY.md`**, the core study on Qwen2.5-Instruct 3B–32B (4-bit):
   - headroom analysis of 151 archival configurations;
   - the dev protocol and the benchmark's label-time-window trap;
   - the decision-state **read-out** (a linear head on the zero-shot prompt's answer-position state) and the training-free **decision-state kNN vote**;
   - sample efficiency and cold start (dev only), the fusion negative result, cost, and a prior-art list.
   - 12 pre-registered test evaluations.
3. **`docs/NEWLLMS_STUDY.md`**, a cross-family replication:
   - the frozen read-out on Qwen3.5-9B, Gemma 4 12B and Ministral 3 8B;
   - RAGTAG-PS with K tuned on a validation split (val495);
   - 18 pre-registered test evaluations, post-hoc robustness and cost.
4. **The predecessor paper**, which defines the benchmark and the baselines (VOTAG, RAGTAG, BRAGTAG, LoRA fine-tuning, SetFit):
   - `SANER2027/` (`sections/*.tex`, `refs.bib`) is **under review at the SANER 2027 Research Track**;
   - `paper/` is the earlier ESEM version, which was rejected;
   - `docs/SANER_REVISION_PLAN.md` has the ESEM reviews;
   - `docs/ENCODER_BASELINES_RESULTS.md` has the SetFit and RoBERTa results.
   - Get the dataset's provenance and the benchmark description from its setup section.
5. **Prior work:**
   - `docs/research/QUESTION_BUG_PRIOR_WORK.md` and `docs/research/BUG_QUESTION_RESEARCH.md`: verbatim quotes and page numbers on the question→bug confusion;
   - the PDFs in `docs/` (LLM issue classification, NASA issue study, data quality, and others);
   - `SANER2027/refs.bib`.
   - Optional: `docs/AGENTIC_PROPOSAL.md`, a separate study finding that agents and written reasoning did not help on this benchmark.
6. **Code:**
   - `scripts/experiments/rag_next/` and `scripts/experiments/newllms/` (OSC tooling in `newllms/osc/`);
   - `llm_labeler.py` (the RAGTAG prompt) and `evaluate.py` (the metrics).
7. **Results live only on the lab machine** (`ssh bgsulab`, BGSU VPN, repo at `~/llm-labler`), under `results/issues11k/exploration/`:
   - `rag_next/`: `test_preds/`, `test_eval/`, `test_eval_log.csv`, `headroom/master_preds.parquet` (all archival baselines aligned per test issue), `features/`;
   - `newllms/`: `test_preds/`, `test_eval/`, `val/`, `gen/`, `features/`, `test_eval_log.csv`.
   - The archival paper runs are under `results/issues11k/{agnostic,project_specific}/`.

**Headline evidence** (verify every number from the files before using it):
- Every read-out beats every RAGTAG, BRAGTAG and LoRA result in the predecessor paper, whose best is 0.785. That covers 7 models, 3B–32B, from 4 families.
- The read-out reaches 0.8445 macro F1 with Qwen2.5-32B, and 0.827–0.829 with the three new models using unchanged hyperparameters.
- On the three new models it beats validation-tuned RAG by +0.05 to +0.09.
- Against SetFit-PS (0.810), results range from a tie (Qwen2.5-3B and 7B) to +0.035 (Qwen2.5-32B). The +0.017 to +0.019 leads of Qwen2.5-14B and the three new models hold per issue but not when resampling projects. Only the 32B lead holds both ways.
- The model's own answer, read from the same forward pass, scores 0.62–0.75 across the 7 models, on dev and test. Generation, not representation, is the bottleneck.
- Known weak spots:
  - every result is on one benchmark (11 projects, 6,600 issues, temporal split);
  - SetFit wins `ansible`, and the SetFit run is a single seed;
  - all models are 4-bit;
  - the predecessor paper picked the Qwen2.5 RAGTAG k values on test, while the new study tunes them on validation;
  - Heo & Lee (ICPC'25) report higher numbers (GPT-4o fine-tuning, about 0.86) on the same 11 projects, but with a different, partly non-temporal split. `RAG_NEXT_STUDY.md` §1 explains why the numbers are not comparable. Expect ICPC reviewers to know that paper.

## Non-negotiable rules

- **Numbers:**
  - Every number in the paper must trace to a result file. Keep a claim → number → file map.
  - Recompute from per-issue predictions where you can, and round from exact values, never from rounded table values.
  - The metric is pooled macro F1 over the 3,300 test issues via `evaluate.py`. Differences get paired bootstrap 95% CIs (2,000 resamples).
  - **Never report accuracy.** Make no equivalence or TOST claims.
- **No test tuning:**
  - Make every new design choice on dev.
  - Write down test evaluations in a notebook *before* running them, and log each one.
  - Report negative results; the fusion loss and the low-label read-out are examples.
- **Do not modify** existing `results/` data or `SANER2027/`, `paper/`, or the pipeline scripts. New outputs go to `results/issues11k/exploration/<name>/` on the lab machine.
- Do not commit, switch branches or push without asking.
- **Shared compute:**
  - The lab RTX 4090 is shared. Run `nvidia-smi` first, never kill anyone else's process, run long jobs under tmux or nohup, and ask before any job over 8 h.
  - OSC is approved; the workflow is in the `osc-cardinal-usage` memory and `newllms/osc/`. Ask before using NRP.
- **Anonymity and concurrent submission:**
  - The predecessor paper is under review at SANER, so this paper must be a clearly different contribution.
  - Write every sentence fresh; copy no text from `SANER2027/` or `paper/`.
  - Describe RAGTAG and the benchmark in the third person, and **ask the user** how to cite the under-review paper, for example as "Anonymous, omitted for double-anonymous review".
- **Citations:**
  - Do a real literature search.
  - Verify every reference exists (DOI, arXiv or venue page) before citing it. Never invent a reference.
  - Say plainly what is prior art (linear probes beating in-context learning, kNN prompting, feature augmentation) and what is new for SE and issue classification.
- **AI disclosure:** draft an anonymized generative-AI disclosure statement as ICPC requires, and flag it to the user.
- **Writing conventions** (from the user's review of the predecessor paper):
  - State what you test, how, the result and what it means. Never voice doubts about your own method's premise.
  - Give the real reason for each design choice, and phrase hypotheses as hypotheses.
  - End with what the results showed and their trade-offs, not with who should use the method.
  - Keep wording simple and never defensive. One line of argument per paragraph. Avoid the "claim: explanation" colon pattern.
  - Put bootstrap details in a text footnote, not in captions.

## Work plan

1. **Analysis (by about 2 Oct).** Write `docs/ICPC2027_PLAN.md` containing:
   - the thesis, 3–4 contributions and the research questions, framed for program comprehension;
   - the claim → evidence → file map;
   - the attacks a skeptical ICPC reviewer would make, each with a planned answer;
   - experiments that close the gaps, ranked by value vs GPU time and your time;
   - an outline with a page budget per section, and every planned figure and table.
   - At minimum, weigh a **second, independent issue dataset** evaluated under the same protocol. Also consider re-tuning the Qwen2.5 RAG baselines on val495 and running SetFit with several seeds.
   - **Then stop and ask the user to approve the plan and the experiments.**
2. **Experiments (after approval, by about 16 Oct).**
   - Reuse the existing pipeline and protocol: dev split, frozen read-out recipe, pre-registration and logging.
   - Record everything in a new notebook in `docs/`.
3. **Writing (full draft by about 23 Oct).**
   - Write in a new `ICPC2027/` directory, using the IEEEtran layout from `SANER2027/`.
   - Put figure and table generators in `scripts/paper_icpc/`, reading from the lab machine's results.
   - Include an anonymized Data Availability section.
   - Build it and check the 10 + 2 page limit and overfull boxes.
   - Have the title and abstract ready for the user to register by 29 Oct.
4. **Review (final by 3 Nov, 2 days before the deadline).**
   - Self-review against the five ICPC criteria, then do one adversarial reviewer pass and fix what it finds.
   - Report to the user: page count, open risks, and every decision that needs them.

Start by reading the files above. Do not start writing the paper until the plan is approved.

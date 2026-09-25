# Related Work dossier (S2)

Written 2026-09-25 by S2 for S4 (Related Work rewrite). S3 and S5 should read the section "Accuracy fixes" too, because several
of those sentences are outside Related Work.

**How this was verified.** Every fact about a paper below was read in that paper: the local PDFs in `docs/`, the ACL
Anthology/PMLR/NeurIPS/arXiv full texts, or (marked "(abstract)") the abstract only. Quotes are verbatim and short. Page numbers
are printed pages unless marked "PDF". For Heo and Lee, printed page = PDF page + 135. BibTeX for new references is in
`docs/reframe/refs_candidates.bib` (sources in comment lines; test-compiled with IEEEtran, no warnings). DBLP was unreachable
this session (anti-bot page), so metadata comes from Crossref, the ACL Anthology, PMLR, NeurIPS proceedings, OpenReview (which
carries DBLP's own records for ICLR papers), CEUR and arXiv. Working notes with longer quotes are in S2's scratchpad and were
not committed; everything S4 needs is here.

## NOVELTY RISKS

**Claim at risk** (01_intro.tex l.13; the live abstract, 00_abstract.tex l.10, says the same):
> "To our knowledge, however, retrieval-augmented few-shot prompting has not been systematically evaluated for \irc\ or compared
> with LoRA fine-tuning."

**Verdict.** The second half ("or compared with LoRA fine-tuning") survived every search. The first half ("has not been
systematically evaluated for IRC") is at real risk, mainly from De Vito et al. (TOSEM 2026), and S4 should narrow it. **Per S4's
instructions (§1.1), S4 must ask the user how to position the paper before writing** if the user is present.

| # | Paper | Risk | Why (facts, source) |
|---|---|---|---|
| 1 | De Vito, Starace, Palomba, Di Martino, Ferrucci, "Advancing LLM-Based Issue Report Classification with Explained Few-Shot Learning, Intent Extraction, Ensemble, and Summarization", TOSEM 2026, doi 10.1145/3815577 (`devito2026advancing`, NEW) | **High** for "not systematically evaluated for IRC"; **low** for "compared with LoRA fine-tuning" | Few-shot LLM IRC with GPT-4o, GPT-3.5-turbo and Qwen2.5-32B; "Explained Few-Shot Learning, which implements the example-based strategy with transparent rationales"; baselines "a RoBERTa-based model, SETFIT, and a previous LLM-based method" (abstract, Crossref; re-checked by S2). **Full text closed** (ACM 403, Unpaywall "closed", replication package on a private figshare link), so the example selection is **unknown**. The authors' own 2024 plan for this work selects examples by similarity: "the targeted or directed selection of few-shot examples, achieved using Vector Databases"; "we perform a similarity search between the vector representation of the current issue report to be labelled and those of previously-labelled issue reports"; "This approach avoids fine-tuning the LLM"; "experimenting with different numbers of few-shot examples" (De Vito et al., Ital-IA 2024, CEUR Vol-3762 pp. 48–53, full text, §2.3–2.4; `devito2024italia`, NEW). Weak counter-hint: the TOSEM reference list (Crossref) cites no retrieval-ICL paper and no vector database. |
| 2 | Johnston, Noei, Assi, Zou, "LabelMate", arXiv 2609.04055 (posted 2026-09-03, preprint; `johnston2026labelmate`, NEW) | **Medium** (if a reviewer counts issue labeling as IRC) | Retrieves the k most similar same-repository labeled issues with FAISS and puts them with their labels in the prompt, k = 1 to 19, with gemma-2-9b, Llama-3.1-8B and Qwen2.5-7B (full text read by S2's sub-agent, §2–3). But the task is multi-label assignment from about 275 derived fine-grained labels, it calls bug/feature/question taxonomies "oversimplified", accuracy is judged by an LLM, and there is **no fine-tuning comparison** (abstract re-checked by S2). |
| 3 | Trad and Chehab, "Retrieval-Augmented Few-Shot Prompting Versus Fine-Tuning for Code Vulnerability Detection", FLLM 2025, pp. 615–622 (`trad2025retrieval`, NEW) | **Low** for IRC; relevant to the "RAG vs fine-tuning" framing in SE (the title mirrors ours) | Gemini-1.5-Flash with random vs retrieved examples and a "retrieval-based labeling" baseline (the analogue of our $k$NN voting), k up to 20; retrieval-augmented prompting reaches "an F1 score of 74.05%" at 20 shots, and "fine-tuned CodeBERT demonstrated superior performance" (91.22%) (arXiv abstract, re-checked by S2). Full text (sub-agent): k varied 1–20; Gemini fine-tuned on Vertex AI reached 59.31%. Different task (multi-label CWE detection); fine-tuned baselines are an API model and a different encoder, not LoRA of the same open model. |
| 4 | Gön, Dinç, Sungur, Tüzün, arXiv 2605.17561 (2026, under review) | Low | RAG over historical bug reports and wiki pages to subclassify invalid bug reports; one k (top 20 reranked to 5); no fine-tuning comparison (sub-agent, full text). Not recommended for citation. |

Also checked and not a threat: Colavito et al. NASA study (zero-shot only; `colavito2026issue`); Zhao et al., APSEC 2025
(multi-turn dialogue, abstract); Soltaniani et al., arXiv 2601.22921 (zero-shot prompting vs different fine-tuned models for
security bug reports); a 2026 survey of language models in MSR (arXiv 2604.00787, 177 papers; its IRC paragraph lists no
retrieval-based method); NLBSE'25 and NLBSE'26 (no IRC track); the NLBSE'24 IRC entries (none uses retrieved examples). **Koyuncu,
TOSEM 2026 (`koyuncu2025exploring`, already cited)**: full text blocked; the abstract mentions only "various prompt engineering
strategies" for fine-grained bug report categorization. **Unchecked; do not claim either way.** Judge the Votes (`dincc2025judge`)
retrieves labeled bug reports as examples, but for binary validity on one Bugzilla project, not IRC (see Papers).

Searched: about 30 web queries; the arXiv API (15 title/abstract queries, e.g. `abs:"issue report" AND retrieval AND classif*`,
`abs:"bug report" AND "in-context"`, `abs:"GitHub issues" AND "few-shot"`); Semantic Scholar citations of Heo and Lee, Aracena et
al. (SCP and NLBSE'24), Colavito et al. IST, the NLBSE'24 competition paper and De Vito et al. (every citing title screened);
OpenAlex, Crossref, Unpaywall, Zenodo and figshare. Not reachable: DBLP, ACM DL, ScienceDirect, ResearchGate, CORE.

**Proposed rewording** (S2's recommendation: Option A in §I and the abstract; Option B in Related Work if the user approves
citing De Vito et al.):

- **Option A (safe whatever De Vito et al. did):**
  > "To our knowledge, however, retrieval-augmented few-shot prompting for \irc\ has not been compared with LoRA fine-tuning of the
  > same LLMs on the same labeled issues."

  Abstract (l.10): "... needs no training but, to our knowledge, has not been compared with Low-Rank Adaptation (LoRA)
  fine-tuning of the same models for this task." (S3 owns the abstract and §I; S4 may change only the novelty sentence in §I.)
- **Option B (Related Work, acknowledges the closest work):**
  > "Few-shot prompting has also been applied to \irc\ with intent extraction, ensembles and explained
  > examples~\cite{devito2026advancing}, and retrieved labeled issues have been used to assign fine-grained repository
  > labels~\cite{johnston2026labelmate}. To our knowledge, however, retrieval-augmented few-shot prompting for \irc\ has not been
  > compared with LoRA fine-tuning of the same LLMs on the same labeled issues."

  Do **not** write that De Vito et al. use random or retrieved examples, or the NLBSE'24 data: neither could be verified.

Why "systematically" is not enough of a qualifier: this study's distinguishing features (a k sweep bounded by a retrieval-only
vote, the retrieval-only and LLM-only references, four model sizes, the same-model LoRA baseline with memory and labeled data) are
real, but a reviewer who knows De Vito et al. will read "not systematically evaluated" as a claim about their paper. The narrowed
claim rests on the one thing no found paper does: compare against LoRA fine-tuning of the same models on the same labeled issues.

## De Vito et al. (TOSEM 2026): facts and recommendation

**Verified (Crossref record and abstract, re-checked by S2 at https://api.crossref.org/works/10.1145/3815577):**
- Title: "Advancing LLM-Based Issue Report Classification with Explained Few-Shot Learning, Intent Extraction, Ensemble, and
  Summarization". Authors: Gabriele De Vito, Luigi Libero Lucio Starace, Fabio Palomba (Salerno), Sergio Di Martino (Napoli
  Federico II), Filomena Ferrucci. ACM TOSEM, article 3815577, online 2026-05-13, no volume or pages yet. Its replication-package
  reference names an earlier title: "Beyond Few-Shot Learning: Advancing LLM-Based Issue Report Classification through Intent
  Extraction, Ensemble Learning, and Summarization".
- Abstract: three methods, "(1) Intent Extraction and Classification ...; (2) Ensemble Classification, which enhances the
  intent-based method through majority voting; and (3) Explained Few-Shot Learning, which implements the example-based strategy with
  transparent rationales"; models "GPT-4o, GPT-3.5-turbo, and Qwen 2.5-32B"; baselines "a RoBERTa-based model, SETFIT, and a
  previous LLM-based method"; "GPT-4 outperforms the state-of-the-art by 5–8%"; "Qwen-2.5 performs better than the larger
  GPT-3.5-turbo"; adaptive ensembles "under specific privacy constraints".

**Not verifiable (full text closed):** how examples are selected (random, fixed or retrieved) and how many; the labels and
datasets (the reference list cites the NLBSE'23 and NLBSE'24 competitions, so overlap with our five NLBSE'24 projects is plausible
but unconfirmed); whether they measure cost (the reference list cites Azure OpenAI and Cloudflare Workers AI pricing pages, which
suggests money cost; no sign of GPU memory); whether "a previous LLM-based method" is a fine-tuned LLM. No LoRA fine-tuning of the
prompted models is mentioned in the abstract.

**Recommendation (the user decides; they set this paper aside earlier): cite and contrast it.**
- Reasons: it is a 2026 TOSEM paper on few-shot LLM IRC, it uses Qwen2.5-32B (our largest model), its title says "few-shot",
  and its authors' published plan (Ital-IA 2024) is similarity-based example selection for IRC without fine-tuning. A reviewer who
  knows it will read an uncited "not systematically evaluated for IRC" as an omission. Citing it costs about 40 words and one
  reference.
- Where: paragraph 2 (LLMs for IRC), just before the narrowed novelty sentence (Option B above).
- Contrast sentence that stays true whatever their example selection turns out to be:
  > "De Vito et al. classify issues with GPT-4o, GPT-3.5-turbo and Qwen2.5-32B through intent extraction, ensembles and explained
  > few-shot examples, and compare them with RoBERTa, SetFit and an earlier LLM method~\cite{devito2026advancing}. We instead vary
  > the number of retrieved examples and compare retrieval-augmented few-shot prompting with LoRA fine-tuning of the same models on
  > the same labeled issues, including peak GPU memory."
- Optional: also cite the open-access plan (`devito2024italia`) to show the idea was proposed; this is honest but spends a
  reference on a paper without results. S2 recommends citing only the TOSEM paper.
- **Best fix if possible:** the user may have institutional ACM access. Reading §3 of the TOSEM paper (example selection, number
  of examples, dataset) would turn the cautious sentence into a precise one; if they retrieve examples by similarity on NLBSE'24,
  the contrast must say so, and the narrowed novelty sentence (Option A) still holds.
- If the user keeps it out: Option A alone is still required; the unqualified first half of the current sentence should not
  survive either way.

## Accuracy fixes for the current Related Work / Introduction (and other sections)

Ordered by severity. Section/line references are to the files as of 2026-09-25 afternoon (S1 was editing concurrently; find the
sentence by its text).

### A1. "State-of-the-art LLM-based methods ... fine-tune the LLM with LoRA ... to achieve their best reported performance" (02_related.tex, IRC paragraph). WRONG.

In all three cited papers the best result comes from **GPT models fine-tuned through OpenAI's fine-tuning API**. LoRA was used
for one open 8B model per paper, which scored lower; Aracena et al. NLBSE'24 used no LoRA at all.

- Heo and Lee, abstract p.136: "the project-agnostic classifier fine-tuned with GPT-4o yields the highest F1-score of 0.8680."
  LoRA only for Llama: "we loaded the model from Hugging Face and performed PEFT with 4-bit quantization and LoRA settings"
  (p.141). Llama-3.1-8B scored 0.8004 (project-specific) and 0.8319 (project-agnostic) (Tables IV, V).
- Aracena et al. SCP 2025: "the overall F1 score rose from 65.47% in the vanilla model to 85.67% in the fine-tuned model"
  (GPT-4o, p.8). LoRA only for DeepSeek: "fine-tuning DeepSeek-R1-Distill-Llama-8B ... using the UnsLoTH framework and Low-Rank
  Adaptation (LoRA) led to moderate performance, with an average F1 score of 59.33%" (p.3), on the 30K NLBSE'23 data, not on
  the five shared projects (Table 6, p.10).
- Aracena et al. NLBSE'24 (read in arXiv 2401.04637v1): "we fine-tuned the gpt-3.5-turbo base model provided in the OpenAI API"
  (p.1); 82.8% F1 (p.2). No open model, no LoRA.

**Accurate wording for S4** (short):
> "Their best reported results come from fine-tuning on labeled issues: GPT models fine-tuned through OpenAI's API
> \cite{aracena2024applyinglargelanguagemodels, aracena2025applying, heo2025study}, and open models such as Llama-3.1-8B and
> DeepSeek-R1-Distill-Llama-8B fine-tuned with LoRA~\cite{hu2022lora}, which scored lower \cite{heo2025study, aracena2025applying}."

Never cite `aracena2024applyinglargelanguagemodels` for LoRA. Never call LoRA fine-tuning "the state of the art"; say it is the
standard way prior IRC work fine-tunes open LLMs (BRIEF §6).

**Same problem elsewhere (for S3/S5, not S4):** 03_approach.tex, §II "LoRA Fine-Tuning Baseline" (≈l.189–194): "Recent IRC
approaches have achieved strong performance by supervised fine-tuning of LLMs [heo, aracena25, aracena24], using Low-Rank
Adaptation (LoRA)". Fix: "... by supervised fine-tuning of LLMs [heo, aracena25, aracena24]; open models were fine-tuned with
LoRA~\cite{hu2022lora} [heo, aracena25]." The live abstract (00_abstract.tex l.10, "Fine-tuned large language models (LLMs) give
the best reported results") and §I l.11 ("fine-tuning on labeled issues gives the best reported results ..., with LoRA for open
models") are accurate; §I l.11 cites aracena24 in the same list, which is fine because the LoRA clause is separate, but S3 may
move the LoRA clause's citation to `heo2025study, aracena2025applying` only. (Lines 19–27 of 00_abstract.tex repeat the wrong
claim but sit inside a `comment` block and do not render.)

### A2. `yu2023retrieval` does not support "similar examples can outperform random ones" (§I l.13, §II-C l.176, Related Work l.16).

Yu et al. fine-tune RoBERTa-large classifiers (plain fine-tuning or cloze-style prompt learning) and **train** a retriever with
two new losses, on 8- and 16-shot NLP benchmarks; there is no LLM in-context learning and no random-demonstration baseline
(Sec. 2–4, pp. 6722–6725, read in the Anthology PDF). Their abstract even says that in the few-shot setting "it is impossible to
retrieve semantically similar examples by using an off-the-shelf metric", and Sec. 2.2 (p. 6723): "static retrieval even
underperforms methods without retrieval in some few-shot tasks".

- Fix: cite only `liu2022makes` (KATE; abstract: "the retrieval-based prompt selection approach consistently outperforms the
  random selection baseline"), optionally with `rubin2022learning` or `milios2023context` (see Papers). Drop `yu2023retrieval`
  from those three sentences.
- Where Yu et al. fits (optional, paragraph 3): as the trained-retriever alternative. Contrast: "Yu et al. train the retriever
  and the classifier; we use an off-the-shelf sentence encoder and no training, and the neighbors' labels alone reach 59.5%
  macro F1 ($k$NN voting)." Do not reproduce their "impossible ... off-the-shelf" sentence: it is about trained encoders with
  16 examples per class, and our $k$NN voting result is the direct answer for our setting.
- §I and §II-C are S3's/S5's text; flag it to them.

### A3. First sentence of the few-shot paragraph (02_related.tex l.16) cites three papers that do not support it.

"Studies show that few-shot prompting can improve LLM performance in specialized tasks due to in-context learning
\cite{assi2026llm, le2023log, ma2023fairness..., logan2021..., brown2020language}."
- `le2023log` (LogPPT) **trains** RoBERTa by prompt tuning on K labeled logs: "we tune a pre-trained language model (e.g.,
  RoBERTa [22]) to predict a specific virtual label token" (p.2); "train the model for 200 steps" (p.7). Not in-context learning.
- `logan2021cuttingpromptsparameterssimple` argues the opposite: "we recommend finetuning LMs for few-shot learning as it is more
  accurate, robust to different prompts, and can be made nearly as efficient as using frozen LMs" (abstract; RoBERTa-large and
  ALBERT). It belongs in paragraph 4 (fine-tuning vs in-context learning).
- `ma2023fairness...` is about predictive bias of demonstrations; it belongs in paragraph 5 (label bias).
- `assi2026llm` (LLM-Cure) does use five fixed few-shot examples for classification (p.6); it can stay, as an SE example of
  few-shot prompting. `brown2020language` stays.

### A4. "transformer-based approaches ... rely on large amounts of training data, which limits their generalizability to settings where labeled data is scarce" (02_related.tex l.13). Partly contradicted.

- Supported by Heo and Lee (p.138–139): "creating issue classifiers using BERT-based models requires a large amount of issue
  data per project, and optimizing the accuracy of BERT models requires building project-specific classifiers, which is
  time-consuming and data-intensive."
- Contradicted by Colavito et al. IST 2025, abstract: "fine-tuning BERT-like encoder-only models enables achieving consistent,
  state-of-the-art performance across datasets even in presence of a small amount of labeled data available for training"; p.13:
  "a training set of hundreds of labeled issues". And by the NASA study (JSS 2026, p.7–8): "SetFit outperforms classifiers based
  on generative LLMs already with less than 20 labeled examples".
- Safe wording: "These classifiers are trained on labeled issues, often in large numbers~\cite{izadi2022catiss, siddiq2022bert}
  ..." and drop the "limits generalizability" clause, or cite only `heo2025study` for it as Heo and Lee's view. **Do not** add a
  claim that encoders need more data than RAG (encoder results are out of the paper; the NASA and IST results would contradict
  it). This also matters for the headline: "11× less labeled data" is a comparison with pooled LoRA fine-tuning only.
- `izadi2022predicting`, `izadi2022catiss`, `colavito2022issue` were not re-read (no local PDFs); CatIss trained on the NLBSE'22
  data of more than 800K issues according to the earlier verification (QUESTION_BUG_PRIOR_WORK.md), and Siddiq and Santos:
  "a dataset of more than 800,000 labeled issues" (abstract).

### A5. "Their extensive pretraining allows these models to be applied with little or no task-specific training [aracena25, aracena24, colavito24leveraging, colavito25, colavito24large]" (02_related.tex l.13). Supported, but by the wrong lead citations.

- Best support: Heo and Lee p.139: "Since GPT models are pre-trained with much larger datasets compared to BERT, they can perform
  well without additional fine-tuning or with only a small amount of data." Colavito et al. IST p.1–2: "without the need for
  fine-tuning, GPT-like LLMs can achieve a performance comparable to BERT-like LLMs."
- Aracena et al. state it only as an aim ("mitigating the necessity for extensive training data", abstract), and their own
  zero-shot GPT-4o scored 65.47% vs 85.67% fine-tuned (p.8).
- Suggested: cite `heo2025study, colavito2024leveraging, colavito2025benchmarking` for it; optionally add "although zero-shot
  performance varies across datasets~\cite{colavito2025benchmarking}" (IST abstract: "their performance varies significantly
  across datasets and they require substantial computational resources").

### A6. Other statements about prior work outside Related Work (for S5; verified against the papers only, not the code)

1. **03_approach.tex l.208**, "Training hyperparameters follow prior work [heo, aracena25]: LoRA rank and alpha 16, learning rate
   2×10⁻⁴, paged AdamW 8-bit, 3 epochs, batch 1, gradient accumulation 16, max length 2,048". The papers' text gives only: rank and
   alpha 16, max sequence length 2,048, and gradient accumulation without a value (Aracena SCP p.6); 3 epochs and per-device batch
   size 1 (Heo p.141). **Learning rate 2e-4, paged AdamW 8-bit and gradient accumulation 16 are in neither paper's text** (they may
   be in the authors' repositories, which S2 did not check). Safer: "follow prior work where reported: LoRA rank and alpha 16 and a
   2,048-token limit~\cite{aracena2025applying}, and 3 epochs with batch size 1~\cite{heo2025study}; we use a learning rate of
   $2\times10^{-4}$, the paged AdamW 8-bit optimizer, and 16 gradient-accumulation steps." (Dettmers et al.'s QLoRA,
   `dettmers2023qlora`, already in refs.bib, is the origin of the 4-bit + paged-optimizer recipe and could be cited here.)
2. **03_approach.tex l.199–200**, "We adopt the instruction-style prompt template used in these prior LoRA fine-tuning studies
   [heo, aracena25] verbatim". The two templates differ: Heo and Lee put "Classify, IN ONLY 1 WORD, the following GitHub issue as
   'feature', 'bug', or 'question' based on its title and body" inside the Llama-3 chat template (Fig. 5, p.141); Aracena et al.'s
   DeepSeek template is "### Instruction / ### Question / ### Response" with chain of thought (p.5). "Verbatim" can hold for at most
   one of them. S5 should cite only the one it matches (the paper's own text or appendix decides; S2 did not check the code).
3. **04_setup.tex l.7**, "the dataset introduced in Heo et al.": five projects are the NLBSE'24 competition data
   (`kallis2024nlbse`) and Heo and Lee added six (Table II, p.140: "The additional data of six projects we collected ..."). Also
   "Heo et al." should be "Heo and Lee" (BRIEF §6). Suggested: "the eleven-project dataset of Heo and Lee~\cite{heo2025study},
   which extends the five-project NLBSE'24 dataset~\cite{kallis2024nlbse} with six projects".
4. Heo and Lee's aggregation is a per-project mean ("Overall average" = mean of 11 project macro F1 values; checked arithmetic on
   Table IV for GPT-4o). Ours is pooled. Any sentence that puts our numbers next to theirs must not imply the same aggregation (and
   BRIEF §6 forbids favourable comparisons anyway).

## refs.bib corrections (existing keys; S4 owns refs.bib)

Verified against the sources named. Keep the keys. Full corrected BibTeX for the long ones is below the table.

| Key | Problem | Fix | Source |
|---|---|---|---|
| `hu2022lora` | `@article`, journal "Iclr", vol. 1, no. 2, pp. 3; author "Wang, Liang" (should be Lu Wang); "and others" | Replace with the ICLR 2022 entry below | openreview.net/forum?id=nZeVKeeFYf9 (DBLP conf/iclr/HuSWALWWC22) |
| `heo2025study` | `pages={1--11}` | `pages={136--146}` (printed pagination; Crossref/Xplore's 1–11 is a registration artifact: the preceding paper is 124–135) | local PDF footers; Crossref |
| `assi2026llm` | renders "Llm-cure: Llm-based ..." | `title={{LLM-Cure}: {LLM}-Based Competitor User Review Analysis for Feature Enhancement}` | api.crossref.org/works/10.1145/3744644 |
| `ma2023fairness...` | `@misc` arXiv | NeurIPS 2023, vol. 36, pp. 43136–43155 (entry below) | proceedings.neurips.cc |
| `logan2021...` | `@misc` arXiv | Findings of ACL 2022, pp. 2824–2835 (entry below) | aclanthology.org/2022.findings-acl.222 |
| `milios2023context` | arXiv | GenBench workshop 2023, pp. 173–184 (entry below) | aclanthology.org/2023.genbench-1.14 |
| `reimers2019sentence` | arXiv | EMNLP-IJCNLP 2019, pp. 3982–3992 (entry below) | aclanthology.org/D19-1410 |
| `sclar2023quantifying` | arXiv | ICLR 2024 (entry below) | openreview.net/forum?id=RIu5lyNXjT |
| `khandelwal2019generalization` | arXiv | ICLR 2020 (only if cited) | openreview.net/forum?id=HklBjCEKvH |
| `liu2022makes` | author "Dolan, William B" | "Dolan, Bill" (as in the Anthology); optional | aclanthology.org/2022.deelio-1.10 |
| `colavito2023few` | no DOI | add `doi={10.1109/NLBSE59153.2023.00011}` | Crossref |
| `colavito2024large` | missing series | add CEUR Workshop Proceedings vol. 3762 | ceur-ws.org/Vol-3762 |
| `cabot2015exploring` | second author parsed as surname "Izquierdo" | `C{\'a}novas Izquierdo, Javier Luis` (only if cited) | Crossref |
| `gomes2023bert`, `vargovich2023...` | title typos/case | not needed in the paper | — |

Unchanged and verified: `le2023log`, `dincc2025judge`, `kallis2024nlbse`, `colavito2024leveraging` (pp. 469–480),
`aracena2024applyinglargelanguagemodels` (pp. 57–60), `yu2023retrieval` (metadata only), `devlin2019bert`,
`panichella2023summary`.

Also: `SANER2027/refs.bib` has a stray line of `=` characters between `liu2019roberta` and `izadi2022catiss` (≈l.178). BibTeX
ignores text outside entries, so it is harmless, but S4 may delete it.

```bibtex
% Source: https://openreview.net/forum?id=nZeVKeeFYf9 (OpenReview BibTeX; same 8 authors as DBLP conf/iclr/HuSWALWWC22)
@inproceedings{hu2022lora,
  title={{LoRA}: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}

% Source: https://proceedings.neurips.cc/paper_files/paper/2023/file/8678da90126aa58326b2fc0254b33a8c-Bibtex-Conference.bib
@inproceedings{ma2023fairnessguidedfewshotpromptinglarge,
  title={Fairness-guided Few-shot Prompting for Large Language Models},
  author={Ma, Huan and Zhang, Changqing and Bian, Yatao and Liu, Lemao and Zhang, Zhirui and Zhao, Peilin and Zhang, Shu and Fu, Huazhu and Hu, Qinghua and Wu, Bingzhe},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  pages={43136--43155},
  year={2023},
  publisher={Curran Associates, Inc.}
}

% Source: https://aclanthology.org/2022.findings-acl.222.bib
@inproceedings{logan2021cuttingpromptsparameterssimple,
  title={Cutting Down on Prompts and Parameters: Simple Few-Shot Learning with Language Models},
  author={Logan IV, Robert and Balazevic, Ivana and Wallace, Eric and Petroni, Fabio and Singh, Sameer and Riedel, Sebastian},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2022},
  pages={2824--2835},
  year={2022},
  address={Dublin, Ireland},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/2022.findings-acl.222}
}

% Source: https://aclanthology.org/2023.genbench-1.14.bib
@inproceedings{milios2023context,
  title={In-Context Learning for Text Classification with Many Labels},
  author={Milios, Aristides and Reddy, Siva and Bahdanau, Dzmitry},
  booktitle={Proceedings of the 1st GenBench Workshop on (Benchmarking) Generalisation in NLP},
  pages={173--184},
  year={2023},
  address={Singapore},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/2023.genbench-1.14}
}

% Source: https://aclanthology.org/D19-1410.bib (pages confirmed against the PDF footer: 3982-3992)
@inproceedings{reimers2019sentence,
  title={Sentence-{BERT}: Sentence Embeddings using {S}iamese {BERT}-Networks},
  author={Reimers, Nils and Gurevych, Iryna},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={3982--3992},
  year={2019},
  address={Hong Kong, China},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/D19-1410}
}

% Source: https://openreview.net/forum?id=RIu5lyNXjT (OpenReview BibTeX; DBLP conf/iclr/Sclar0TS24)
@inproceedings{sclar2023quantifying,
  title={Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How {I} learned to start worrying about prompt formatting},
  author={Sclar, Melanie and Choi, Yejin and Tsvetkov, Yulia and Suhr, Alane},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## Papers

Ranked within each paragraph by importance (S4 cuts from the bottom). "In refs.bib" = key exists; "NEW" = entry in
`refs_candidates.bib`.

### Paragraph 1: supervised issue report classification

#### `kallis2019ticket`, `antoniol2008bug`, `fan2017road` (in refs.bib, cited)
- What: classic ML IRC (Ticket Tagger, a fastText-based GitHub app; Antoniol et al.'s bug vs non-bug text classifiers; Fan et al.'s
  text-mining study). Not re-read this session; the current citations are unchanged from the verified ESEM text.
- How we differ: they train a classifier per repository or dataset; we train nothing for RAG.
- Paragraph: 1.

#### `izadi2022catiss`, `trautsch2022predicting`, `colavito2022issue`, `bharadwaj2022github`, `siddiq2022bert` (in refs.bib; siddiq uncited)
- What: fine-tuned transformer encoders (RoBERTa, seBERT, BERT) for bug/enhancement/question, mostly NLBSE'22 tool-competition
  entries on more than 800K GitHub issues.
- Verified facts: Siddiq and Santos, "a dataset of more than 800,000 labeled issues" (abstract). CatIss and Bharadwaj and Kadam:
  question is the weakest class, attributed to it being the smallest class (QUESTION_BUG_PRIOR_WORK.md, C9).
- How we differ: they fine-tune an encoder on large labeled pools; this study holds the LLM fixed and varies only how the labeled
  issues are used (as prompt examples or for LoRA weight updates).
- Paragraph: 1. `siddiq2022bert` optional (add DOI 10.1145/3528588.3528660 if cited).

#### `colavito2023few` (in refs.bib, uncited)
- What: Colavito, Lanubile, Novielli, "Few-Shot Learning for Issue Report Classification", NLBSE 2023, pp. 16–19: their SetFit
  entry to the NLBSE'23 tool competition.
- Verified facts (abstract; the full text is paywalled, details from the authors' Ital-IA 2024 summary `colavito2024large`, read in
  full): SetFit = contrastive fine-tuning of a sentence transformer plus a classification head; four labels (bug, documentation,
  feature, question); trained on about 200 manually relabeled issues (184 usable after relabeling, Ital-IA Table 1); F1-micro 0.8321
  on the manually verified test subset and 0.7767 on the challenge test set (abstract).
- The NLBSE'24 dataset README (github.com/nlbse2024/issue-report-classification) asks users of the dataset to cite this paper,
  `kallis2024nlbse`, `kallis2019ticket` and Kallis et al. 2021 (`kallis2021predicting`, NEW, optional). Five of our eleven projects
  are that dataset.
- How we differ: SetFit fine-tunes a small sentence encoder on a few hundred issues; we keep the sentence encoder frozen and use
  it only to retrieve labeled issues for an LLM.
- Paragraph: 1 (one clause: "and SetFit-based few-shot fine-tuning~\cite{colavito2023few}"). Do not quote its numbers
  (different labels and data).
- Caveat: if the text says "NLBSE'23", cite `kallis2023nlbse` (NEW), not `kallis2024nlbse`.

#### `kallis2024nlbse` (in refs.bib, cited)
- NLBSE'24 tool competition: 3,000 issues, five projects, bug/feature/question, balanced, 50/50 split. Question weakest for almost
  all entries (QUESTION_BUG_PRIOR_WORK.md C2).
- Paragraph: 1 and 6 (already in 6).

### Paragraph 2: LLMs for issue report classification

#### `heo2025study` (in refs.bib) — Heo and Lee, ICPC 2025 (RENE track), pp. 136–146
- What: replicates Aracena et al. NLBSE'24 and extends the five NLBSE'24 projects with six more, then fine-tunes GPT-3.5 Turbo and
  GPT-4o through the OpenAI API and Llama-3.1-8B with 4-bit PEFT/LoRA. **Our benchmark is exactly their dataset** (Table II, p.140:
  eleven projects, "300:300 / bug: 100, feature: 100, question: 100").
- Verified facts:
  - No prompt examples; every classifier is fine-tuned: "For RQ1, we fine-tuned OpenAI's GPT-3.5 Turbo and GPT-4o using the
    Instruction tuning technique. For RQ2, we fine-tune the Llama 3.1 8B model." (p.141)
  - Settings: "RQ1 and RQ2 are focus on generating project-specific classifiers. For RQ3, we integrated all data from eleven projects
    to generate a project-agnostic classifier." (p.141)
  - Best: GPT-4o project-agnostic, F1 0.8680 (abstract, p.136); GPT-4o project-specific 0.8639; Llama-3.1-8B 0.8004 (PS) and
    0.8319 (PA). Their "Overall" is a mean of per-project F1 (not pooled).
  - Cost: training time only: "about 25-30 minutes on GPT-3.5 Turbo and GPT-4o ... about 40-45 minutes on Llama 3.1 8B" per
    project, and "275-280 minutes on GPT-3.5 and GPT-4o, and 470 minutes on Llama 3.1 8B" project-agnostic (p.145); API prices in
    Table III. No GPU memory, no labeled-data experiment.
  - Future work (p.145): "Future research could include optimizing prompts and applying N-shot techniques to improve
    classification accuracy across diverse project datasets."
- How we differ: they fine-tune (API or LoRA) and never put labeled issues in the prompt; we use the same eleven projects and
  both of their data settings, but compare a training-free use of the labeled issues (retrieved examples) with LoRA fine-tuning of
  the same open models, and measure peak GPU memory and labeled data per project.
- Paragraph: 2 (also the source of the benchmark in §III).

#### `aracena2025applying` (in refs.bib) — Aracena et al., SCP 246 (2025) 103333
- What: fine-tunes GPT-3.5-turbo, GPT-4o and GPT-4o-mini per repository through the OpenAI API on NLBSE'24 (five of our projects),
  and DeepSeek-R1-Distill-Llama-8B with Unsloth + LoRA on the 30K NLBSE'23 data.
- Verified facts:
  - Best: fine-tuned GPT-4o, 85.66–85.67% F1 (p.8), vs 65.47% for vanilla (zero-shot) GPT-4o.
  - DeepSeek LoRA: "average F1 score of 59.33%" (p.3); LoRA rank and alpha 16, max length 2,048, 4-bit via Unsloth (p.6).
  - Cost: money only ("GPT-4o costs $25.00 per 1M tokens, resulting in a total cost of just over $100", p.10); "Each model required
    approximately five hours to complete" (p.6). No GPU memory.
  - Labeled data: "Increasing the dataset size did not improve the F1 score" (abstract).
  - Future work (pp.13–14): RAG, "an LLM could retrieve relevant issues, discussions, and resolutions from a knowledge base before
    classifying a new issue. This would not only enhance precision but also reduce the need for extensive fine-tuning", "particularly
    for ambiguous labels such as ``question''". Also a "two-stage classification pipeline" for questions (p.14).
- How we differ: they fine-tune; we test their proposed retrieval augmentation directly, with retrieved labeled issues as prompt
  examples, and compare it with LoRA fine-tuning of the same open models on memory and labeled data.
- Paragraph: 2 (and 6, already).

#### `aracena2024applyinglargelanguagemodels` (in refs.bib) — Aracena et al., NLBSE 2024, pp. 57–60
- Read in arXiv 2401.04637v1 (preprint; camera-ready may differ). gpt-3.5-turbo fine-tuned through "OpenAI's fine-tuning API" per
  repository; 82.8% F1 (p.2); no examples in the prompt, no open model. Heo and Lee note it "ranked 5th in the issue report
  classification competition" (Heo p.137).
- Paragraph: 2 (API fine-tuning only; never for LoRA). Add DOI 10.1145/3643787.3648043 (Crossref-verified).

#### `colavito2025benchmarking` (in refs.bib) — Colavito et al., IST 184 (2025) 107758
- What: benchmarks 22 open LLMs (7B–72B, 4-bit) and GPT-4o zero-shot and 1-/2-shot on NLBSE'23 (manually verified, 4 labels) and
  NLBSE'24 (five of our projects), against RoBERTa and SetFit.
- Verified facts:
  - Random examples: "we experiment with 1- and 2-shot settings, including one and two examples per class ... we randomly select
    the examples to include in the prompt from the train set split" (p.4).
  - Result: "using generative LLMs with a few-shot learning does not improve the classification performance, compared to the one
    observed in the zero-shot learning condition, with the only notable exception of GPT, for which a slight improvement is observed
    and for large models requiring 4GPU for deployment when evaluated on the NLBSE24 dataset" (p.9).
  - No fine-tuning of generative LLMs (out of scope, p.14); they plan "the fine-tuning of generative LLMs and prompt engineering"
    as future work (p.14).
  - Cost: hardware class (2 vs 4 A100 64 GB), inference time over the test set ("9' for 7b models ... to 25 h for 70b models",
    p.13), share of parsable outputs, prompt truncation. No GPU memory in GB, no money.
  - No retrieval or similarity-based example selection anywhere in the paper.
- How we differ: they choose one or two random examples per class and do not fine-tune the LLMs; we retrieve the k most similar
  labeled issues per query, sweep k up to 15, and compare with LoRA fine-tuning of the same models.
- Paragraph: 2.

#### `colavito2026issue` (NEW) — Colavito, Lanubile, Novielli, Arreza, Shi, JSS 237 (2026) 112851 ("In Practice")
- What: bug vs non-bug classification on two NASA flight-software projects (cFS, 2,724 issues; F´, 751 issues) with six open LLMs
  (4-bit) and GPT-4o, zero-shot only, against fine-tuned RoBERTa and SetFit.
- Verified facts: "We use a zero-shot approach where the prompt only contains label descriptions without any labeled examples.
  This decision is based on findings from previous work (Colavito et al., 2024b), which indicated no performance gain in a few-shot
  learning setting" (p.5). SetFit is best (macro F1 .86 and .95, Table 4); "SetFit outperforms classifiers based on generative LLMs
  already with less than 20 labeled examples" (p.7–8).
- How we differ: binary labels, industrial data, zero-shot LLMs only; we study retrieved examples on three labels.
- Paragraph: 2, one clause: "... and a later study of NASA flight software therefore prompted LLMs only zero-shot
  \cite{colavito2026issue}." It strengthens the gap (random examples gave no gain, so this line of work stopped using examples;
  retrieved ones were never tried). **Risk:** its SetFit result (fewer than 20 labeled examples beat zero-shot LLMs) is a
  labeled-data comparison a reviewer could raise against our "11× less labeled data" framing. Mention only the zero-shot fact;
  do not discuss encoders (user decision). Rank: medium; drop if space is short.

#### `devito2026advancing` (NEW; needs the user's approval) — De Vito et al., TOSEM 2026
- See the section "De Vito et al." at the top. Paragraph 2, immediately before the narrowed novelty sentence. Rank: high if the
  user approves (it is the closest IRC paper).

#### `colavito2024leveraging`, `colavito2024large` (in refs.bib)
- MSR 2024 (GPT-like LLMs for issue labeling, pp. 469–480) and Ital-IA 2024 summary. Colavito IST cites the MSR paper for random
  example selection (IST p.4, ref [14]); S2 did not read the MSR paper itself. Keep as co-citations; do not state new facts.

### Paragraph 3: retrieval-selected in-context examples

#### `liu2022makes` (in refs.bib) — KATE, DeeLIO 2022
- Verified (abstract): for GPT-3, "we propose to retrieve examples that are semantically-similar to a test query sample to formulate
  its corresponding prompt"; "the retrieval-based prompt selection approach consistently outperforms the random selection baseline";
  "sentence encoders fine-tuned on task-related datasets yield even more helpful retrieval results". NLU and generation benchmarks.
- How we differ: NLP benchmarks with GPT-3; we apply the same idea to three-label IRC with open models and compare it with LoRA
  fine-tuning of the same models.
- Paragraph: 3 (lead citation). Keep.

#### `dincc2025judge` (in refs.bib) — Dinç and Tüzün, "Judge the Votes", AIware 2025, pp. 1–10. **Closest SE precedent.**
- What: binary bug-report validity (VALID/INVALID) on 10,000 Mozilla Firefox Bugzilla reports; LLMs prompted with retrieved labeled
  neighbors, plus a "judge" LLM that sees classifier votes.
- Verified facts:
  - Same retrieval stack as ours: "We appended top-5 semantically similar bug reports from the vector index of the training corpus we
    created using all-MiniLM-L6-v2 [12] as embedder ... with FAISS" (p.5).
  - Fixed k: "We fixed the retrieved neighbor count at k = 5 ... Future studies should sweep k more finely, compare dense (MiniLM)
    versus sparse (BM25) retrieval, and add automatic relevance filters to exclude mislabeled or off-topic neighbors." (p.7)
  - Single project: "Our dataset is drawn exclusively from Mozilla Firefox bug reports hosted on Bugzilla." (p.8)
  - Models: GPT-o3-mini, o4-mini, 4o-mini, Llama-3.1-70B (p.5). Baselines: fully fine-tuned BERT-family encoders and TF-IDF
    classifiers (p.4); fine-tuned RoBERTa is best, F1 0.909 (p.1); best few-shot LLM 0.815 vs 0.614 zero-shot (Table IV).
  - Cost in money and time: "Running nine models for 2,000 samples cost around $80, and took 15-18 hours" (p.7).
  - Bias: LLMs "tend to err on the side of classifying bug reports as valid" (p.8); the remedy is the extra judge call.
- How we differ: binary validity on one Bugzilla project with a fixed k = 5, compared with fine-tuned encoders; we classify three
  labels on eleven GitHub projects, sweep k up to where $k$NN voting peaks, compare with LoRA fine-tuning of the same LLMs on peak
  GPU memory and labeled data, and address label bias with a rule on the neighbors' labels instead of an extra LLM call. Their own
  future-work list (sweep k, filter neighbors) names two things this study does.
- Paragraph: 3 (and 5 for the extra-call contrast).

#### `nashid2023retrieval` (NEW) — CEDAR, ICSE 2023, pp. 2450–2462
- Verified: "automatically retrieves code demonstrations similar to the developer task, based on embedding or frequency analysis
  ... two different tasks, namely, test assertion generation and program repair" (abstract); Codex; fills the context with as many
  retrieved demonstrations as fit ("we select the maximum number of similar demonstrations that can fit within the limit", Sec. II);
  compared with separately published task-specific and fine-tuned models: "outperforms existing task-specific and fine-tuned models
  by 333% and 11%" for assertion generation and "competitive with recent fine-tuned models" for repair (abstract).
- How we differ: code generation tasks, a context-filling number of demonstrations, and fine-tuned baselines that are different
  models; we classify issues, sweep k, and fine-tune the same models with LoRA on the same labeled issues.
- Paragraph: 3. Rank high (the SE precedent at ICSE).

#### `gao2023what` (NEW) — Gao et al., ASE 2023, pp. 761–773
- Verified: studies "the selection, order, and number of demonstration examples" for "code summarization, bug fixing, and program
  synthesis" (abstract), with Codex; "BM-25 is a simple and effective method" for selection (Finding 2, Sec. IV-A); "More
  demonstration examples in the prompt will not always lead to better performance considering the truncation problem. To save
  costs, it is suggested that four examples are used" (Finding 5, Sec. IV-C; they varied 1–64).
- How we differ: code tasks with an API model and no fine-tuning baseline; we sweep k for classification, bound it by a
  retrieval-only vote, and compare with LoRA fine-tuning.
- Paragraph: 3 (the k-sweep contrast). Medium rank.

#### `milios2023context` (in refs.bib, uncited; venue update above)
- Verified (full text): retrieval with a frozen SBERT (`all-mpnet-base-v2`) for intent and emotion classification with 50–150
  labels (BANKING77, HWU64, CLINC150, GoEmotions); 20 examples by default, up to about 110 when filling the context; OPT and
  LLaMA models; compared with fine-tuned DeBERTa adapters and SetFit and a "Pre-trained SBERT 1-NN" baseline; retrieval + ICL is
  best on the intent sets in 5-/10-shot settings. Small models plateau as examples are added. Label-mix finding: the "neutral"
  class "only appears in the top 3 classes retrieved ... 9% of the time", and "the retriever may be limiting the performance".
- How we differ: many-label NLP classification; they use retrieval to fit the label space into the context. We study three labels in
  SE, sweep k, compare with LoRA fine-tuning of the same LLMs on memory, and act on the retrieved label mix with a filter.
- Paragraph: 3 (can replace `yu2023retrieval` as the second citation); optionally 5 (the retrieved label mix limits a class).

#### `johnston2026labelmate` (NEW, arXiv preprint 2026-09-03) — LabelMate
- Verified (abstract, re-checked by S2): derives project-specific labels and assigns them to new issues without pre-existing
  labeled training data; "16,500 issue reports across 30 GitHub repositories"; about 275 labels; "an average labeling accuracy of
  89.84%". Full text (sub-agent, §2–3): FAISS retrieves the k most similar same-repository issues with their original labels, k = 1
  to 19; gemma-2-9b, Llama-3.1-8B, Qwen2.5-7B; accuracy judged by an LLM "Label Evaluator"; no fine-tuning comparison; it calls
  bug/feature/question taxonomies "oversimplified".
- How we differ: fine-grained multi-label assignment judged by an LLM vs three-label IRC scored against gold labels; no
  fine-tuning comparison vs LoRA fine-tuning of the same models.
- Paragraph: 3 (or next to the novelty sentence, Option B). Medium rank. It is a preprint; the user may prefer to omit it, but a
  reviewer who knows it could raise it.

#### `rubin2022learning` (NEW) — Rubin, Herzig, Berant, NAACL 2022, pp. 2655–2671
- Verified: trains a dense retriever with LM-scored positives and negatives (EPR, abstract); semantic parsing (BREAK, MTOP,
  SMCalFlow); BM25 beat SBERT as the unsupervised retriever (Sec. 4); prompts fill a 2,048-token context; FAISS.
- How we differ: a trained retriever for semantic parsing; we use an off-the-shelf encoder for classification.
- Paragraph: 3, optional (only if the text contrasts trained vs off-the-shelf retrievers). Low rank.

#### `yu2023retrieval` (in refs.bib, cited) — see A2. Optional in paragraph 3 as the trained-retriever, trained-classifier
alternative; otherwise remove.

#### `assi2026llm` (in refs.bib) — LLM-Cure, TOSEM 35(4), 2026
- Verified: feature assignment of app reviews with "five few-shot examples demonstrating feature assignment" (p.6), hard-coded in
  the prompt (Fig. 3, p.7); retrieval appears only in the second phase, which pulls **unlabeled** positive competitor reviews by
  mistral-embed similarity as context for generating suggestions (pp.8–9). Mixtral-8x7B; no fine-tuning baseline; no cost
  measurement; average feature-assignment F1 85% (p.3).
- How we differ: fixed, hand-picked demonstrations; retrieval of unlabeled text for generation. Ours retrieves labeled issues per
  query as demonstrations.
- Paragraph: 3 at most as "few-shot prompting in SE with fixed examples"; do **not** describe it as retrieval-selected
  demonstrations. Low rank.

#### `le2023log` (in refs.bib) — LogPPT, ICSE 2023
- Verified: prompt-tuning of RoBERTa on K labeled logs chosen by "Adaptive Random Sampling" (p.1–2); K swept 4–128 (p.7–8);
  compared with full fine-tuning of the same RoBERTa (p.9–10). Not LLM in-context learning.
- Paragraph: remove from the in-context sentence (A3). If kept, only as "few-shot learning from a handful of labeled examples".
  Low rank.

#### `khandelwal2019generalization` (in refs.bib, uncited) — kNN-LM, ICLR 2020
- Verified: interpolates the LM's next-token distribution with a kNN distribution over a datastore, "with no additional training";
  domain adaptation "by simply varying the nearest neighbor datastore". No classification labels, no demonstrations.
- Use (optional): precedent for "new labeled data only has to be indexed". Not needed in Related Work; `cover1967nearest` and
  `dudani1976distance` already ground $k$NN voting in §II.

### Paragraph 4: fine-tuning versus in-context learning

#### `weyssow2025exploring` (NEW) — Weyssow, Zhou, Kim, Lo, Sahraoui, TOSEM 34(7), 2025. **Most important new reference.**
- What: PEFT (LoRA, QLoRA, others) vs in-context learning vs RAG for Python code generation (Conala, CodeAlpacaPy, APPS) with
  CodeLlama-7B and other models up to 34B.
- Verified facts (arXiv v3; the TOSEM text may differ slightly):
  - ICL uses random examples: "we use up to 16 examples for the Conala dataset and 8 examples for CodeAlpacaPy. These examples are
    randomly sampled from the corresponding training datasets" (Sec. 4.3).
  - RAG uses retrieved examples: "we leverage GTE-small ... we retrieve up to 16 examples for Conala and 4 examples for
    CodeAlpacaPy, selecting those with instructions most similar to the test input" (Sec. 4.3).
  - Result: "LoRA is superior to ICL and RAG on Conala and CodeAlpacaPy datasets across the three CodeLlama-7B variants" (Sec. 5.3).
  - Memory: peak GPU memory is measured for fine-tuning only (Fig. 1: "Peak GPU memory consumption during models fine-tuning using
    full fine-tuning (ft), LoRA, and QLoRA"); ICL/RAG cost appears only as a qualitative "low" (Table 1).
- How we differ: code generation, evaluated by exact match; we study three-label classification, measure peak GPU memory for both
  RAG inference and LoRA training plus inference, and count labeled data per project.
- **Framing (important).** Their finding agrees with ours for plain RAG: RAG trails pooled LoRA fine-tuning by 2.8 points and the
  CI excludes zero. Only filtered RAG matches (1.0 point behind, CI includes zero). Suggested contrast sentence: "For code
  generation, Weyssow et al. found LoRA ahead of both randomly chosen and retrieved examples~\cite{weyssow2025exploring}, and they
  measured memory only for fine-tuning. For IRC, plain retrieved examples also trail LoRA fine-tuning (\Cref{sec:rq3}); with the
  example filter, they match it while using a third less peak GPU memory." Do not write that we "overturn" or "contradict" them.
- Paragraph: 4. Rank 1.

#### `mosbach2023few` (NEW) — Mosbach et al., Findings of ACL 2023, pp. 12284–12314
- Verified: "we compare the generalization of few-shot fine-tuning and in-context learning to challenge datasets, while controlling
  for the models used, the number of examples, and the number of parameters, ranging from 125M to 30B" (abstract); "both approaches
  generalize similarly; they exhibit large variation and depend on properties such as model size and the number of examples"
  (abstract); OPT models; NLI and paraphrase identification; examples randomly sampled, 16 demonstrations for ICL (Sec. 3); cost
  compared only qualitatively (Table 3).
- How we differ: they give both methods the same few random examples on NLP benchmarks; we compare the setting practitioners face in
  IRC, where fine-tuning uses all labeled issues and prompting shows at most 15 retrieved ones, with the same models on both sides,
  and measure memory and labeled data.
- Paragraph: 4. Rank 2 (the canonical same-model comparison; it is the precedent for our same-model baseline).

#### `trad2025retrieval` (NEW) — Trad and Chehab, FLLM 2025, pp. 615–622
- Verified (arXiv abstract, re-checked by S2): compares "standard few-shot prompting, retrieval-augmented prompting with
  semantically similar examples, and retrieval-based labeling" with Gemini-1.5-Flash for code vulnerability detection;
  retrieval-augmented prompting reaches "an F1 score of 74.05%" at 20 shots and beats zero-shot prompting; "fine-tuned CodeBERT
  demonstrated superior performance" (91.22%) "though requiring additional training resources and maintenance effort". Full text
  (sub-agent): k varied 1–20; Gemini fine-tuned on Vertex AI scored 59.31%; four CWE labels, multi-label.
- How we differ: multi-label vulnerability detection with an API model, where the fine-tuned baselines are an API fine-tune and a
  different encoder; we compare with LoRA fine-tuning of the same open models on the same labeled issues and measure peak GPU
  memory. Its "retrieval-based labeling" baseline is the counterpart of our $k$NN voting.
- Paragraph: 4 (SE comparison of retrieved-example prompting and fine-tuning). Medium rank. Its title mirrors ours, so a reviewer
  may know it.

#### `logan2021cuttingpromptsparameterssimple` (in refs.bib, cited in the wrong place; see A3)
- Verified: "we recommend finetuning LMs for few-shot learning as it is more accurate, robust to different prompts, and can be made
  nearly as efficient as using frozen LMs" (abstract); RoBERTa-large and ALBERT-xxl-v2; "the main advantage of prompt-based
  finetuning over in-context learning is that it achieves higher accuracy, especially when the LM is relatively small".
- How we differ: sub-billion masked LMs on GLUE-style tasks; we use 3B–32B instruction LLMs on issue reports.
- Paragraph: 4, optional co-citation with Mosbach ("fine-tuning is often more accurate than in-context learning for small
  models~\cite{logan2021..., mosbach2023few}"). Low rank.

### Paragraph 5: label bias in demonstrations (the filter)

#### `zhao2021calibrate` (NEW) — Zhao et al., ICML 2021, PMLR 139, pp. 12697–12706
- Verified: LMs "are biased towards outputting answers that are (1) frequent in the prompt (majority label bias), (2) towards the
  end of the prompt (recency bias), and (3) common in the pre-training data (common token bias)" (Sec. 4); "when one class is more
  common, GPT-3 2.7B is heavily biased towards predicting that class" (Sec. 4). Fix: contextual calibration, which estimates the
  bias "by asking for its prediction when given a training prompt and a content-free test input such as 'N/A'" and re-weights the
  label probabilities (abstract, Sec. 5); it needs the label probabilities and extra queries (three content-free inputs). Their
  demonstrations are random ("choose different random sets of training examples", Sec. 3).
- How we differ: they calibrate the output using label probabilities and extra content-free queries; our filter changes the input,
  dropping bug-labeled examples by a count rule over the retrieved neighbors, with no extra LLM call and no access to probabilities.
- **Link to our hypothesis** (writing rule 10): our zero-shot models already label 46–59% of questions as bugs, and we hypothesize
  that bug-labeled examples retrieved for a question reinforce this; "majority label bias" is the named mechanism from prior work,
  and "common token bias" matches the zero-shot preference. State ours as a hypothesis.
- Paragraph: 5. Rank 1 in this paragraph.

#### `ma2023fairnessguidedfewshotpromptinglarge` (in refs.bib, cited in the wrong place; venue update above)
- Verified (arXiv v3 and NeurIPS camera-ready): scores whole prompts by the entropy of the LLM's label distribution on a
  content-free input and searches for a low-bias set and order of demonstrations (G-fair, T-fair); 4 demonstrations; one prompt for
  the whole test set; SST-2, AGNews, TREC, CoLA, RTE; no fine-tuning comparison. On the similarity-based baseline (KATE), they write
  that it "selects demonstrations with labels that are the same as the test samples, and ... LLMs tend to predict biased predictions
  toward the labels that always appear in the context."
- How we differ: they search for one fixed low-bias prompt with extra LLM calls; we keep per-query nearest-neighbor retrieval and
  act on the retrieved neighbors' label counts. Their KATE sentence is the closest prior statement of our motivation: retrieved
  examples carry a label mix, and the LLM follows it.
- Paragraph: 5. Rank 2.

#### `min2022rethinking` (NEW) — Min et al., EMNLP 2022, pp. 11048–11064
- Verified: "randomly replacing labels in the demonstrations barely hurts performance on a range of classification and multi-choce
  tasks, consistently over 12 different models including GPT-3" (abstract, "multi-choce" sic); what matters is "(1) the label space,
  (2) the distribution of the input text, and (3) the overall format" (abstract); k = 16 demonstrations "sampled at uniform"
  (Sec. 3).
- How we differ: random demonstrations; with retrieved ones, the neighbors' labels alone predict the query's label (59.5% macro F1,
  $k$NN voting) and the filter's gain depends on them.
- Paragraph: 5 or 3, optional; it needs care (a reviewer may read it as "labels in examples do not matter", which our filter result
  contradicts only for retrieved examples). Low-medium rank.

#### `dincc2025judge` — for the contrast "extra judge LLM call vs rule on the neighbors" (see paragraph 3).

### Paragraph 6: question-to-bug misclassification (keep; already verified)

All claims and quotes are in `docs/research/QUESTION_BUG_PRIOR_WORK.md` (C1–C14). New supporting evidence found this session:
- Aracena et al. SCP p.11 give three causes (interrogative markers, user mislabels, blurred boundary) and propose "a two-stage
  classification pipeline that first distinguishes between information-seeking (questions) and information-providing (bugs/features)
  issues" (p.14). Optional contrast: their proposed remedy is an extra classification stage; the filter is a rule on the retrieved
  examples.
- Milios et al.'s neutral-class finding (retrieved neighbors rarely include the hard class) is the NLP analogue of our observation
  that neighbors retrieved for questions are often bug-labeled. Only if space allows, and only if §II/§IV states that observation
  with a number (S2 did not check).
- Nothing found contradicts the existing paragraph. Update the method names (S1) and keep the rule "never claim RAG beats prior
  work on question".

### Not recommended

| Paper | Why not |
|---|---|
| `NoTrainingWheels_ICML2025.pdf` (Gupta, Sethi, Sethi, arXiv 2506.18598, ICML 2025 Actionable Interpretability workshop) | Subtracts a mean-activation "bias vector" inside fine-tuned ViT/BERT classifiers (Waterbirds, CelebA, UTKFace, MultiNLI). Not about demonstrations or retrieval; needs activation access. Only shared idea: training-free inference-time bias correction. |
| `CAA_rimsky_2024.pdf` (Contrastive Activation Addition, ACL 2024) | Activation steering of behaviors such as sycophancy in Llama 2 Chat. Irrelevant. |
| `tunstall2022efficient` (SetFit) | Non-archival workshop poster; encoders are out of the paper (user decision). In NEW bib only in case that changes. |
| `panichella2023summary`, `vargovich2023givemelabeledissues`, `gomes2023bert`, `wei2021finetuned`, `joulin2016bag`, `devlin2019bert` | Workshop summary; API-domain labels (not issue type); long-lived bug prediction; instruction tuning; fastText; BERT. None needed. |
| `sclar2023quantifying` | Not Related Work, but **useful for Threats (S5)**: open LLMs are "extremely sensitive to subtle changes in prompt formatting in few-shot settings", up to 76 accuracy points for LLaMA-2-13B (ICLR 2024; abstract/intro). We use one fixed prompt format. |
| `dettmers2023qlora` | Not Related Work, but S5 may cite it in §II for the 4-bit LoRA + paged optimizer recipe (see A6.1). |
| `cabot2015exploring` | Not Related Work; a SANER 2015 paper that fits §I motivation ("the label mechanism is scarcely used", abstract). Optional for S3/S5. |

## Proposed outline for Related Work

Six run-in paragraphs (`\myparagraph`), each ending its description with an explicit contrast. Word estimates are for live text.
The current live Related Work is **about 690 words** (counted without comments and commands at the tag). The outline totals about
**1,000 words** (+310), inside the +300 to +450 range. If S4's budget is smaller, cut in this order: (1) paragraph 6's second
half (the explanations) down to two sentences; (2) Rubin, Min, Logan, Gao; (3) the NASA clause; (4) merge paragraphs 4 and 5 into
one. Never cut the contrast sentences or the A1 fix.

Macros and labels assume S1/S3: `\knn`, `\rag`, `\frag`, `\Cref{sec:rq1,sec:rq2,sec:rq3}`. Numbers come from BRIEF §9 /
NUMBERS.md; S4 re-checks each against NUMBERS.md. Draft sentences below follow BRIEF §7 but are **proposals**; S4 owns the prose.

### 1. Supervised issue report classification (~120 words)

Cites: `antoniol2008bug, kallis2019ticket, fan2017road`; `izadi2022catiss, trautsch2022predicting, colavito2022issue,
bharadwaj2022github` (+`siddiq2022bert`); `kallis2024nlbse`; `colavito2023few`; `heo2025study`.

Describe (fixes A4):
> Early \irc\ work trained classic machine-learning classifiers on issue text~\cite{antoniol2008bug, kallis2019ticket,
> fan2017road}. Later work fine-tuned transformer encoders such as RoBERTa and seBERT~\cite{izadi2022catiss,
> trautsch2022predicting, colavito2022issue, bharadwaj2022github}, often on more than 800,000 labeled issues~\cite{siddiq2022bert},
> and SetFit-based few-shot fine-tuning was entered in the NLBSE tool competitions~\cite{colavito2023few, kallis2024nlbse}.

Contrast (proposed):
> All of these methods train a classifier on the labeled issues. We instead keep the LLM fixed and vary only how its labeled issues
> are used, as prompt examples or for LoRA weight updates, which is why our baseline is LoRA fine-tuning of the same models rather
> than a different classifier.

Do not add any claim about encoder accuracy or data needs (A4; encoders are out of the paper).

### 2. LLMs for issue report classification (~190 words)

Cites: `heo2025study, colavito2024leveraging, colavito2025benchmarking, colavito2026issue (NEW, optional),
aracena2024applyinglargelanguagemodels, aracena2025applying, hu2022lora`.

Describe (fixes A1, A5):
> LLMs can label issues without task-specific training~\cite{heo2025study, colavito2024leveraging, colavito2025benchmarking},
> although their zero-shot performance varies across datasets~\cite{colavito2025benchmarking}. Colavito et al. added one or two
> randomly chosen labeled examples per class to the prompt, which did not improve over zero-shot prompting for most
> models~\cite{colavito2025benchmarking}, and a later study of NASA flight software therefore prompted LLMs zero-shot
> only~\cite{colavito2026issue}. The best reported results come from fine-tuning on labeled issues: GPT models fine-tuned through
> OpenAI's API~\cite{aracena2024applyinglargelanguagemodels, aracena2025applying, heo2025study}, while open models such as
> Llama-3.1-8B and DeepSeek-R1-Distill-Llama-8B were fine-tuned with LoRA~\cite{hu2022lora} and scored
> lower~\cite{heo2025study, aracena2025applying}. For future work, Heo and Lee suggest few-shot prompting and Aracena et al.
> suggest retrieving relevant past issues to reduce the need for extensive fine-tuning~\cite{heo2025study, aracena2025applying}.

Contrast (proposed; every clause verified: none of the four papers retrieves examples, none varies k beyond 2 per class, none
measures GPU memory in GB, none compares fine-tuning with a training-free use of the same labeled issues):
> These studies either prompt with no or randomly chosen examples or fine-tune, and none measures GPU memory. We evaluate the two
> suggested directions together: we show the LLM the $k$ labeled issues most similar to the query, for $k$ from 0 to 15, and compare
> this with LoRA fine-tuning of the same four models on the same labeled issues, including peak GPU memory and the labeled data each
> target project needs.

If the user approves De Vito et al., insert before the contrast (+40 words) the sentence from the "De Vito et al." section, and
end the paragraph with the narrowed novelty sentence (Option A/B in NOVELTY RISKS). Then place the contrast's "These studies ..."
before the De Vito sentence, or restrict it to "Colavito et al., Heo and Lee, and Aracena et al.", so that it makes no claim about
De Vito et al.'s examples.

Careful: Heo and Lee's PS vs PA comparison and Aracena et al.'s dataset-size experiment are labeled-data experiments of a kind, so
do **not** write "none studies labeled-data needs". Heo and Lee report training time and API prices; Aracena et al. API cost;
Colavito et al. inference time and GPU count. "None measures GPU memory" is exact.

### 3. Retrieval-selected in-context examples (~170 words)

Cites: `liu2022makes`, `milios2023context`, `nashid2023retrieval (NEW)`, `gao2023what (NEW)`, `dincc2025judge`; optional
`rubin2022learning (NEW)`, `yu2023retrieval` (only as the trained-retriever alternative; see A2).

Describe:
> Choosing the labeled examples most similar to the query outperforms random choice for GPT-3 on NLP
> benchmarks~\cite{liu2022makes}, and retrieved examples have since been used for text classification with many
> labels~\cite{milios2023context} and, in software engineering, for code tasks such as assertion generation, program repair and
> summarization~\cite{nashid2023retrieval, gao2023what}. Closest to our work, Dinç and Tüzün show an LLM the five most similar
> labeled reports, retrieved with the same sentence encoder and FAISS as ours, to decide whether Firefox bug reports are
> valid~\cite{dincc2025judge}.

Contrast (proposed; verified: Judge the Votes fixes k = 5 and names a k sweep as future work, p.7; its fine-tuned baselines are
encoders; CEDAR compares with separately published fine-tuned models):
> They fix $k=5$ and compare with fine-tuned encoders on a single project. We apply retrieved examples to three-label \irc\ across
> eleven projects, sweep $k$ up to where a vote over the neighbors' labels alone peaks, separate the contributions of retrieval and
> of the LLM with \knn\ and zero-shot references (\Cref{sec:rq1}), and compare with LoRA fine-tuning of the same LLMs
> (\Cref{sec:rq3}).

Optional one-clause contrast with Gao et al. (only if S4 checks the k values in NUMBERS.md): Gao et al. suggest four examples for
code tasks (Finding 5); our best k per size is given in BRIEF §9 (RAG 3/6/12/12, filtered RAG 6/12/15/12). Do not claim a general
rule from it.

Do not write "prior work fixes the number of examples" in general: Gao et al. varied it from 1 to 64, Milios et al. from 20 to
context-filling, CEDAR and Rubin et al. fill the context.

### 4. Fine-tuning versus in-context learning (~120 words)

Cites: `mosbach2023few (NEW)`, `weyssow2025exploring (NEW)`; optional `logan2021cuttingpromptsparameterssimple`.

Describe:
> Mosbach et al. compared few-shot fine-tuning and in-context learning of the same models with the same randomly sampled examples
> and found that both generalize similarly on NLP benchmarks~\cite{mosbach2023few}. For code generation, Weyssow et al. found LoRA
> ahead of both randomly chosen and retrieved examples~\cite{weyssow2025exploring}, and they measured peak GPU memory only for
> fine-tuning.

Contrast (proposed; honesty guards 1 and 7):
> Our comparison follows how \irc\ is deployed: fine-tuning uses all labeled issues, while the prompt shows at most 15 retrieved
> ones, and we measure peak GPU memory for both. As for code generation, plain retrieved examples trail LoRA fine-tuning; with the
> example filter, they come within 1.0 point of it on average, a difference whose 95\% CI includes zero (\Cref{sec:rq3}).

Optional clause (+30 words, `trad2025retrieval`): "For code vulnerability detection, Trad and Chehab found retrieved examples ahead
of zero-shot prompting but behind a fine-tuned CodeBERT~\cite{trad2025retrieval}." It strengthens the point that the SE evidence so
far favours fine-tuning, which makes the IRC result the paper's contribution; but it adds a third comparison to a dense paragraph.

Never "overturn"/"contradict" Weyssow et al.; their task and metric differ.

### 5. Label bias in the examples (~130 words)

Cites: `zhao2021calibrate (NEW)`, `ma2023fairnessguidedfewshotpromptinglarge`; optional `min2022rethinking (NEW)`,
`dincc2025judge`.

Describe (writing rule 10: prior cause, then our matching observation with a pointer):
> LLMs prompted with examples favor labels that are frequent in the prompt or common in pretraining~\cite{zhao2021calibrate}, and
> examples selected by similarity, which tend to share a label, pull predictions toward that label~\cite{ma2023fairness...}. In our
> results, zero-shot prompting already labels 46--59\% of questions as bugs, and we hypothesize that bug-labeled neighbors
> retrieved for a question reinforce this error (\Cref{sec:rq2}). Zhao et al. correct the bias by calibrating the output
> probabilities with extra content-free queries~\cite{zhao2021calibrate}, and Ma et al. search for a low-bias set of examples with
> extra LLM calls~\cite{ma2023fairness...}.

Contrast (proposed):
> \Frag\ changes the input instead: a count rule over the retrieved neighbors' labels removes the bug-labeled examples for suspected
> questions, with no extra LLM call and no access to output probabilities.

Check before using "46--59%": it is the zero-shot share in BRIEF §9; S3 may have moved it. If paragraph 6 already states it, point
to it instead of repeating the number (number diet).

### 6. Question-to-bug misclassification (~250 words; keep, lightly trimmed)

Keep the two existing paragraphs (claims C1–C14 verified in QUESTION_BUG_PRIOR_WORK.md). Changes: new method names (S1); percent
format; "RAG-based methods" → "retrieved examples" / "\frag"; if space is short, shorten the explanations paragraph. Optional
contrast with Aracena et al.'s proposed remedy (a two-stage classifier that first separates information-seeking from
information-providing issues, SCP p.14): "Aracena et al. propose a separate first-stage classifier for questions; \frag\ instead
changes which retrieved examples the LLM sees." Honesty guard 6 applies throughout.

### Novelty sentence (end of paragraph 2 or 3; also §I)

See NOVELTY RISKS for the wording.

## Open questions for the user

1. **De Vito et al. (TOSEM 2026): cite it?** S2 recommends citing and contrasting it (paragraph 2, cautious sentence above). If you
   have ACM access, please check how it selects few-shot examples (§3 of the paper); that decides whether the contrast can be
   sharper. Until you decide, S4 follows your earlier decision (not cited) but must still narrow the novelty sentence (Option A).
2. **Novelty sentence:** approve Option A ("... has not been compared with LoRA fine-tuning of the same LLMs on the same labeled
   issues") for §I and the abstract? It drops "has not been systematically evaluated for IRC".
3. **LabelMate (arXiv, 2026-09-03):** cite it as retrieved-labeled-issue prompting for fine-grained labels, or leave it out as a
   very recent preprint?
4. **NASA study (`colavito2026issue`, JSS 2026):** S2 recommends one clause in paragraph 2 (LLMs prompted zero-shot only). Its
   SetFit result (fewer than 20 labeled examples beat zero-shot LLMs) is not mentioned, in line with keeping encoders out; be aware
   a reviewer may raise it against the "11× less labeled data" headline.
5. **Accuracy fixes outside Related Work** (A1 in the §II fine-tuning subsection, A2 in §I and §II, A6.1–A6.3 in §II and §III): S3/S5 own those sections; they are listed
   here so they are not lost.

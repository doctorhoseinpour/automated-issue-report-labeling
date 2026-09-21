# Investigating the bug–question boundary

Research memo, 17 September 2026. This is a proposal and evidence synthesis, not a claim of improved benchmark performance. The active SANER manuscript, implementation, recorded encoder results, local related-work PDFs, and primary online sources were reviewed. The manuscript and existing experiments are unchanged.

**Recommendation.** Start from project-specific SetFit, give it separate representations of the report's content and surface form, and teach the bug–question distinction through reliable, same-topic examples with different labels. Use an LLM to extract grounded evidence and supply an additional signal on difficult cases only if validation shows it helps. The central hypothesis is that recovering the requested action and learning the project's labeling convention will help more than simply suppressing bug predictions.

**1. What the evidence actually establishes**

The strongest recorded classifier in this repository is already an encoder. These are previously recorded results, not measurements rerun for this memo:

| Method | Scope | Macro F1 | Question F1 |
|---|---|---:|---:|
| SetFit, issue-adapted MPNet | Project-specific | .810 | .770 |
| SetFit, generic MPNet | Project-specific | .805 | .767 |
| LoRA Qwen-14B | Project-agnostic | .785 | .736 |
| BRAGTAG Qwen-32B | Project-specific | .781 | .729 |
| RAGTAG Qwen-32B | Project-specific | .767 | .699 |

Sources: [encoder results](../ENCODER_BASELINES_RESULTS.md), [BRAGTAG table](../../SANER2027/tables/bragtag_results.tex). The generative approaches retain research interest, but the performance target must include SetFit.

The manuscript reports that zero-shot Qwen sends 46–59% of true questions to bug; retrieval reduces that to 26–36%, and BRAGTAG to 19–27% at the selected configurations. This establishes a recurring directional error and a useful intervention. It does not isolate its cause: BRAGTAG simultaneously changes example labels, example content, prompt length, and sometimes how much query text survives truncation. See [evaluation](../../SANER2027/sections/05_evaluations.tex) and [implementation](../../llm_labeler.py).

Novielli's group provides unusually direct evidence about the target itself. Colavito et al.'s JSS study inspected 370 errors and described a bug-shaped initial report labeled question after a maintainer established that the behavior was expected. They explicitly discuss how the same original report could instead receive bug if the behavior were unintended. Generic filtering by popularity, age, or multiple labels did not generally solve classification. This points to a difference between what a reporter says and what a maintainer eventually concludes. [JSS 2024, publisher PDF pp. 7–8 and 12](https://doi.org/10.1016/j.jss.2023.111838).

Their next benchmark contains a balanced NLBSE'24 dataset, yet Table 7 reports question F1 of .48 for Llama3-8B and .79 for issue-adapted SetFit. Unequal class counts cannot explain that gap. Its comparison with curated data is suggestive, but changes multiple factors and removes unresolved human disagreements; it is not a controlled label-cleaning experiment. [IST 2025, pp. 3–4 and 8](https://doi.org/10.1016/j.infsof.2025.107758).

Their NASA study associates different labeling workflows with different difficulty: F-prime versus cFS has 91% versus 41% of labels assigned at creation, and SetFit macro F1 of .95 versus .86. This is a binary bug/non-bug task with several dataset differences, so the relationship is observational. It nevertheless strengthens the case for investigating when and by whom a label was assigned. [NASA/JSS 2026, pp. 6–7](https://ntrs.nasa.gov/citations/20260002137).

There is also a concrete information bottleneck. The retrieval code concatenates title and body and leaves MiniLM's sequence limit at its default. That model truncates beyond 256 word pieces. The LLM prompt builder also truncates descriptions from the right. A report's boilerplate and error messages can survive while a late request for explanation disappears. This is a testable hypothesis, not yet a measured explanation of the observed errors. [Index code](../../build_11k_index.py), [prompt truncation](../../llm_labeler.py), [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

**2. My interpretation: the labels combine different dimensions**

Consider the invented report: “The editor cannot find my Python interpreter. Here is the error. How should I configure it?” It contains a failure symptom and an information request. Whether it represents a defect in the editor depends on configuration, responsibility for the component, and sometimes later investigation. The presence of an error message, a question mark, or reproduction steps does not settle that distinction.

I would separate these variables:

| Variable | Information to preserve |
|---|---|
| Requested action | Explanation, configuration help, repair, new capability; allow multiple actions |
| Reported behavior | What happened; what was expected; regression claims |
| Supporting evidence | Quoted source spans; reproduction or documented-contract evidence when present |
| Ownership | Target project, dependency, user configuration, or unknown; distinguish claims from verified facts |
| Project convention | How the project's historical reports map those signals to native labels |
| Later disposition | Maintainer diagnosis and action, often unavailable at submission |

The benchmark is approximately asking for `P(native label | initial text, project)`. Some labels additionally depend on later evidence unavailable in that input. This is partly a prediction-of-triage-outcome problem. Project conditioning means learning how text maps to labels, not merely adding a class prior: each project here is class-balanced.

This interpretation predicts three different sources of error: recoverable evidence overlooked in the original text; recoverable conventions learned from historical examples; and ambiguity or missing evidence that the original text cannot resolve. Their proportions remain unknown. A stronger model may help the first two, but cannot guarantee reconstruction of absent information.

Do not turn this into a rule that a bug needs proof or a minimal reproducer. Legitimate bug reports can lack both. Nor should every request for help be a question: a reporter can ask for help with a genuine defect. Preserve unknown and overlapping evidence until the final label decision.

**3. The approach I would build**

First, keep the existing SetFit classifier as the reference. Give a candidate model two views: the original text and an evidence-preserving content view. The latter removes only recognizable empty scaffolding or instructions, retaining actual answers, code, error messages, expected/actual descriptions, and negation. Encode title, beginning, ending, and populated sections separately when the report exceeds the encoder horizon. Keep the total representation budget controlled against a head-and-tail baseline. This tests whether seeing the right evidence matters before adding complex reasoning.

Second, add a small set of structured evidence fields from the table above. For a pilot, a fixed LLM can extract them with exact quoted spans and `unknown` where unsupported. It must not see the target's gold label and must not infer a confirmed defect from a reporter's claim. Validate spans against the input and audit extraction errors. Initially treat these fields as extra features alongside the raw encoder representation; do not force all information through an imperfect summarizer. Later, useful extraction can be distilled into a smaller model.

Third, learn the difficult boundary using contrasts. Within each project's retained training data, retrieve topically similar bug and question reports, preferably with similar templates. The goal is to find examples where API names and formatting are uninformative and the difference lies in requested action or supporting evidence. Present both sides when using an LLM, or emphasize verified pairs during contrastive encoder training. Retain a feature example for a three-class decision. Never label synthetic reports solely from a generator's preference, and do not automatically treat every opposite native label as a reliable hard negative: ambiguous pairs need adjudication or reduced weight.

Fourth, fit a regularized decision layer using encoder scores, evidence features, and project interactions. Share parameters across projects and shrink project adjustments toward the shared mapping when data is thin. Compare against the existing separate per-project heads; the shared model is a hypothesis, not an assumed improvement. Calibration, routing, and weighting must be learned from out-of-fold training predictions.

An optional bug–question specialist can redistribute the baseline's combined bug/question probability while preserving feature probability. Route using observable disagreement between views, a calibrated margin, or bug/question being the two leading classes—not the true label or knowledge that the baseline is wrong. This limited specialist cannot correct all feature confusions; evaluate its complete three-class output, including that limitation. Use a learned combination rather than unconditional LLM override.

Finally, add controlled invariance training only if the first experiment supports it. Present the same content with a verified irrelevant wrapper removed or changed, encouraging the same prediction; pair it with genuine same-topic changes in requested action to require discrimination. Do not remove substantive statements such as a documented expected behavior under the name of cleanup. Counterfactual augmentation and supervised contrastive learning are established ingredients, not new inventions here. [Kaushik et al.](https://arxiv.org/abs/1909.12434), [Khosla et al.](https://arxiv.org/abs/2004.11362).

The reason this might work is specific: preserve evidence lost to truncation, reduce dependence on superficial cues, expose distinctions that topic retrieval misses, and learn local label conventions without discarding semantic information. It could fail if the residual errors mainly require unavailable future evidence, if extraction hallucinates, or if the specialist mostly repeats SetFit's mistakes. Each possibility has a corresponding test below.

**4. What is—and is not—new**

An intent-first two-stage classifier is already proposed by Aracena et al., along with RAG for ambiguous questions. A simple information-seeking gate therefore is not a sufficient novelty claim. [SCP 2025, publisher pp. 11 and 13–14](https://doi.org/10.1016/j.scico.2025.103333).

Similarly, an LLM judge is not inherently an improvement. On a different binary bug-validity task, *Judge the Votes* reports F1 .909 for its best RoBERTa, .906 for voting, and .871 for its retrieval-assisted LLM judge. This is a useful negative precedent for unconditional escalation to a larger model. [Dinç and Tüzün, AIware 2025](https://doi.org/10.1109/AIware69974.2025.00025).

A defensible contribution would be a demonstrated separation of report evidence, surface form, and project disposition; a controlled test of which causes bug–question errors; and a classifier exploiting the recoverable signals. Whether that combined method is novel needs a focused follow-up literature search before making a priority claim.

**5. A small, falsifiable experiment**

Use the historical 3,300-row training CSV. Reserve 10 examples per class per project: 330 validation reports. The other 2,970 alone supply training and retrieval examples. Keep normalized exact duplicates together or remove training-side duplicates of held-out reports; report near-duplicate limitations. Existing trained SetFit checkpoints saw these 330 reports, so retrain the encoder inside this split. Frozen pretrained base weights are acceptable, subject to their corpus provenance. All tuning and routing use inner folds of the retained pool. The paper's original test split is not used for method selection.

Run these comparisons in stages, keeping seeds and inputs paired:

| Comparison | What it resolves |
|---|---|
| Raw SetFit versus calibrated raw SetFit | Whether score correction alone is enough |
| Raw versus content view versus raw+content | Whether normalization helps, destroys useful signals, or provides complementary evidence |
| Prefix versus equal-budget head/tail or section views | Whether evidence visibility explains gains |
| Ordinary neighbors versus same-topic opposite-label neighbors | Whether boundary examples help beyond topic matching |
| Best encoder versus encoder+grounded evidence | Whether extraction adds predictive information |
| Encoder versus selectively combined specialist | Whether complementary errors justify extra inference |

Do not run the full Cartesian product initially. First screen representations, then add the specialist to the best validated representation. Include a BRAGTAG reference at a fixed setting if studying the generative mechanism. To interpret bug-example removal causally, compare with equal-count, matched-length deletion and fixed query visibility; otherwise shorter context remains an alternative explanation.

For a small human check, select 66 random training reports plus 66 reports enriched for model disagreement, spanning all projects and including correctly classified bugs and feature controls. Two readers first see only initial title/body, blinded to labels and predictions. They annotate the evidence dimensions, defensible classes, and whether classification needs missing information. Reveal native labels and later history in a separate pass to distinguish policy, new evidence, and clear annotation error. Report random and enriched samples separately, retain unresolved cases, and do not use audit examples as unseen evaluation after revising the method on them.

Primary outcome: pooled three-class macro F1 on all held-out reports. Also report question precision/recall/F1, bug recall, both question→bug and bug→question rates, feature F1, routing fraction, latency, and per-project changes. Count every report, including ambiguous and invalid cases. Use paired bootstrap intervals; stratification by project and label describes uncertainty conditional on these projects. Project-level variation and additional seeds must also be shown.

A practical screening target is +.02 macro F1 and +.03 question F1 over the retrained encoder, with no more than .02 absolute bug-recall loss. These are proposed engineering criteria, not expected results or significance thresholds. At 330 reports, small gains will be uncertain; promising results warrant a larger locked validation/final evaluation. Repeatedly selecting variants on this slice makes it development data, so it cannot also establish final superiority.

For a submission-time classifier, later comments and final labels are audit outcomes only. A separate updated-triage system may incorporate comments available by an explicitly defined cutoff, but its scores must be reported as a different information setting.

**6. Evidence cautions that materially affect this investigation**

The active discussion reports 77.33% template-format errors and two independent coders; the archived qualitative report instead records one LLM-assisted coder and different category percentages. These may reflect later recoding, but the available provenance does not reconcile them. They should not be combined as one verified estimate. In any event, percentages from selected errors do not estimate corpus-wide noise or establish an upper bound on F1. [Active discussion](../../SANER2027/sections/06_discussion.tex), [archived audit](../../scripts/audit/qual/QUAL_REPORT.md).

The audit sampler attaches unfiltered top-12 neighbors even for BRAGTAG; its approach argument does not change the attached context. It also analyzes separate residual-error samples. Consequently its retrieval-skew percentages cannot establish what the BRAGTAG model actually saw or isolate an intervention effect. [Sampler](../../scripts/audit/qual/sample_errors.py).

The issue-adapted MPNet checkpoint was pretrained using title/body pairs from NLBSE'22. Audit overlap with evaluation issue identities/text before relying on its small advantage over generic MPNet. This is a question of document exposure; overlap would not by itself prove supervised test-label leakage. Keep generic MPNet as a transparent reference. [Authors' model card](https://huggingface.co/Collab-uniba/github-issues-mpnet-st-e10).

The local corpus audit finds substantial label-dependent date ranges, so template versions can proxy issue age and sampling history. A future chronological evaluation should check this directly. Do not exploit timestamps as shortcuts to inflate this benchmark: deployment performance on new reports is the relevant goal.

**7. Artifacts from this investigation**

The [corpus audit](bug_question_probe/corpus_audit.json) contains descriptive counts, not classifier performance. Its regular-expression flags are surface-form proxies, not human judgments of template misuse. It finds bug-form markers in 705/2,200 bugs, 534/2,200 questions, and 317/2,200 features; four normalized exact-text duplicate groups include one conflicting-label group. The marker rule's coverage varies across project templates, so these are not comparable estimates of actual template usage.

A separate CPU feasibility probe and its outputs live in [bug_question_probe](bug_question_probe/). It uses the explicitly authorized historical training split and does not modify the manuscript or archival results. Its lexical model is a mechanism screen; its results must not be compared as an improved model on the paper's 3,300-report test set. See that directory's report for execution status and measurements.

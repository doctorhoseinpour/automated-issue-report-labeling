# RAG-next study: can anything beat SetFit-PS on the 11-project benchmark?

Living lab notebook, started 2026-09-24. Post-deadline research; nothing here goes into the SANER 2027 submission.

- **Code:** `scripts/experiments/rag_next/` (local clone and lab machine)
- **Outputs:** `results/issues11k/exploration/rag_next/` (lab machine only)

Existing `results/` data, the pipeline scripts, `SANER2027/` and `paper/` were not modified.

---

## 0. Protocol

| Item | Choice |
|---|---|
| Test set | The paper's 3,300-issue test split (`results/issues11k/agnostic/neighbors/test_split.csv`). Pooled concat-then-evaluate macro F1. |
| Dev split | Inside the paper's train split, for each (repo, label) group sorted by `created_at`: the **newest 30** issues are *dev* (990 issues) and the older 70 are *inner* (2,310). Built by `make_splits.py`, which checks that dev is strictly newer than inner in every group. The paper's train split is also strictly older than test in all 33 groups. |
| Tuning | Every design and hyperparameter choice is fit on *inner* and scored on *dev*. Test is used only for final candidates. Test evaluations are logged in §6. |
| Leakage | Retrieval indexes and demonstrations come from the train split only (dev phase: from *inner* only). Test labels are never read before a final evaluation. No external data, and nothing fetched about test issues. |
| Retrieval | The paper's recipe: `all-MiniLM-L6-v2` over `clean_text(title + " " + body)`, cosine. Reproduces the paper's `neighbors_k30.csv` rank-exact (100.00% title agreement on all 3,300 × 30 test slots). *Centering* (per-project mean subtraction) is an existing baseline component from the 2026-09-22 probe, not a contribution of this study. |
| Models | Qwen2.5-Instruct bnb-4bit via Unsloth (`FastLanguageModel.from_pretrained(..., load_in_4bit=True)`), the paper's checkpoints. |

**Dev tracks test.** VOTAG-PS@15 is 0.612 on dev vs 0.595 on test; centered VOTAG-PS@15 is 0.659 vs 0.638. Script: `dev_votag.py`.

## 1. Bars, verified from `results/`

Recomputed from the per-issue predictions by `collect_preds.py` + `headroom.py` (pooled, invalid = wrong):

| Method | Macro F1 | F1 bug / feat / question | Invalid |
|---|---|---|---|
| SetFit, issue-adapted MPNet, PS | **0.8100** | 0.810 / 0.850 / 0.770 | 0 |
| SetFit, generic MPNet, PS | 0.8053 | 0.801 / 0.848 / 0.767 | 0 |
| LoRA FT Qwen-14B, PA | 0.7853 | 0.792 / 0.828 / 0.736 | 0.03% |
| BRAGTAG-PS 3B (k6) / 7B (k12) / 14B (k15) / 32B (k12) | 0.7137 / 0.7376 / 0.7560 / 0.7811 | 32B: 0.775 / 0.840 / 0.729 | 32B: 4.0% |
| RAGTAG-PS 3B (k3) / 7B (k6) / 14B (k12) / 32B (k12) | 0.6970 / 0.7175 / 0.7318 / 0.7665 | | |
| RoBERTa-base, PA | 0.7713 | | 0 |
| VOTAG PS@15 / PA@16 | 0.5951 / 0.6039 | | 0 |

All numbers in the brief check out.

**Prior work on "this benchmark".** Heo & Lee (ICPC'25) report GPT-4o fine-tuning at macro F1 ≈ 0.864 (PS) and 0.868 (PA), and Llama-3.1-8B at 0.80–0.86 (Table IV/V). These numbers are **not comparable**. Five of the eleven projects use the NLBSE'24 split, which is not temporal: only 755 of its 1,500 test issues are in our test split (`docs/research/QUESTION_BUG_PRIOR_WORK.md`, R2). Heo & Lee also preprocess with CM1. On this repo's data, non-temporal block splits are easier: VOTAG scores 0.63–0.64 there vs 0.595–0.604 on our temporal split (2026-09-22 probe). A like-for-like comparison would require re-running their recipe on our split, which was not attempted.

## 2. Headroom analysis (existing test predictions; `headroom.py`)

Findings from the stored predictions of all 151 configurations. This is descriptive and involves no tuning.

1. **Retrieval coverage is not the bottleneck.** The true label is among the top-k PS neighbors for 80.6% of issues at k=3, 95.7% at k=9 and 98.4% at k=15, similar across classes. The problem is the decision made from the neighbors. Bug holds 41% of neighbor slots (33% of the index): the hubness bias.
2. **When the LLM overrides its neighbors, it is right more often than VOTAG.** Best-k RAGTAG/BRAGTAG disagree with the same-k VOTAG vote on 34–39% of issues. On those, the LLM is right 64–74% of the time and VOTAG 17–26%. The LLM adds real signal on top of the neighbor vote. Its failure is elsewhere: bias and missing project conventions (item 5).
3. **SetFit and the LLMs make different errors.** Correlations of per-issue correctness are 0.39–0.47 between SetFit-PS and each LLM method. Oracle accuracy of pairs:
   - SetFit + FT-14B: 0.888
   - SetFit + BRAGTAG-32B: 0.881
   - SetFit + BRAGTAG-7B: 0.881
   - SetFit + SetFit-mpnet: 0.865
   - Best single method (SetFit): 0.810

   The oracle over 10 methods reaches 0.962, and over all 99 non-VOTAG configurations 0.989.
4. **Noise floor.** 311 issues (9.4%) are wrong under ≥90% of the 99 configurations. Of these, 195 are questions (114 predicted as bug, 74 as feature), 88 are features and 28 are bugs. Only 38 issues (1.2%) are wrong under every configuration. This matches the paper's failure analysis (template misuse, hybrid intent, label noise). A realistic ceiling is roughly 0.88–0.90 accuracy, not 1.0.
5. **Per-project conventions are where SetFit wins.** On `ansible`, SetFit-PS scores 0.954 while BRAGTAG-32B scores 0.747 and FT-14B 0.798. There, *bug* and *feature* are mostly pull requests with a "Bugfix/Feature Pull Request" template, and *question* is plain issue text that often reads like a feature request. A convention like that is learnable from 300 labels but not from 12 in-context neighbors. The LLMs win on `bitcoin`, `dart-lang`, `react`, `roslyn` and `opencv`.
6. **Invalid outputs are a pipeline artifact.** 206 of 259 sampled BRAGTAG invalids (32B+7B, k12) have prompts above the 8,192-token context: after mid-log query truncation, the model continues the log. Scoring the three label tokens removes this by construction. It is worth about 0.02–0.03 macro F1 for BRAGTAG at k≥9, where 3–5% of outputs are invalid.
7. **SetFit is overconfident.** 95% of its test predictions have p>0.9, yet accuracy in that bin is 0.83. Any fusion must calibrate on held-out data; in-sample SetFit probabilities are useless for stacking.

**Implication.** The headroom lies in combining the LLM's semantic signal with a decision that is trained on *all* the project's labeled issues, not only 12. Better example selection alone cannot close a 0.81 vs 0.78 gap that is driven by label conventions.

## 3. Ideas considered (ranked before testing)

Mechanism, cost and a cheap kill test for each. Rank = prior belief of beating SetFit-PS.

| # | Idea | Hypothesis / mechanism | Cost | Kill test (dev) |
|---|---|---|---|---|
| 1 | **LLM decision-state read-out** ("probe"). Run the RAGTAG prompt, then read the hidden state at the answer position (after the `<label>` prefill) with a linear model trained on the labeled issues. | The LLM's answer-position state already encodes the label. Generation throws that away through the prior bug bias, invalid outputs and the 12-example horizon. A linear read-out trained on all 2,310/3,300 labels learns project conventions as SetFit does, on top of a far stronger representation. Known to beat ICL in general NLP (Cho et al. 2023; Abbas et al. 2024). | One prefill per issue, no generation. Training-light (a linear head). | Zero-shot-prompt states of 7B, PA probe on inner → dev: kill if it falls well below dev SetFit-PS. |
| 2 | **Retrieval-conditioned read-out.** Same, but the state comes from the k-shot RAGTAG prompt. | In-context neighbors add instance-level evidence (near-duplicates, same-component conventions) that a read-out of the zero-shot state lacks. | k× prompt tokens. Training queries need **causal** leave-one-out retrieval (§4). | Beats idea 1 on dev at the same model? |
| 3 | **Calibrated label scoring.** Score the three label tokens (constrained decoding), then calibrate on dev. | Removes invalid outputs and the bug prior. Training-free apart from 3–12 calibration parameters. | Same pass as idea 1. | Beats BRAGTAG on dev at the same size? |
| 4 | **Fusion with SetFit and kNN.** Log-linear stacking of components, fit on dev by cross-fitting. | Error correlation of about 0.44 and pair oracles of about 0.88 promise a gain if each component's probabilities are honest. | Needs SetFit trained on inner (GPU, about 40 min). | Cross-fitted stacker on dev vs SetFit alone. |
| 5 | **Class-balanced contrastive demonstrations** (top-m per class). | Removes hubness-driven bug over-representation and shows the model same-topic bug-vs-question contrasts. | LLM runs. | Beats BRAGTAG on dev? |
| 6 | **Task-conditioned retrieval.** Retrieve demonstrations by distance between LLM decision states instead of MiniLM. | A training-free embedding conditioned on the classification task should be more label-homophilous. | Reuses idea 1 states. | kNN vote in state space vs MiniLM on dev. |
| 7 | Retrieval-augmented LoRA fine-tuning. | FT plus in-context evidence. | Heavy. | Only if 1–4 fail. |

**Outcomes** (details in §4–§6):

| # | Idea | Outcome |
|---|---|---|
| 1 | Decision-state read-out | **Kept, the headline.** Dev 0.827–0.852 (3B–32B); test 0.813–**0.845**. R-32B beats SetFit-PS by +0.035, robust to project resampling. |
| 2 | Read-out of k-shot RAG states | **Dropped before running.** Training states need leave-one-out or causal retrieval, both biased on this benchmark (§4.1). RAG signals add ≤ 0.002 on top of idea 1 (§4.5). |
| 3 | Calibrated label scoring | **Superseded.** Calibration lifts the model's own zero-shot label from 0.688 to 0.743 (7B dev), and constrained decoding removes invalid outputs, but the read-out of the same pass is +0.08 better. |
| 4 | Fusion (stacker) | **Rejected on test.** +0.004 dev on top of the 14B read-out (TF-IDF + kNN), but −0.009 on test: the dev-fit stacker over-predicts bug on the shifted test period (§6.4). |
| 5 | Class-balanced demonstrations | **Not run.** Superseded by idea 6, which removes hubness better (decision-state neighbors: homophily 0.72 vs 0.52), and by the finding that the generation step, not demo selection, is the bottleneck (§4.7). |
| 6 | Decision-state retrieval | **Kept as the training-free result.** A kNN vote in decision-state space beats RAGTAG/BRAGTAG at every size on test. K-32B (0.799) beats BRAGTAG-32B and ties SetFit. As RAGTAG demos it helps the LLM (+0.017 dev) but less than voting directly. |
| 7 | Retrieval-augmented LoRA | **Not run.** Ideas 1 and 6 already met the bar at a fraction of the cost. |

## 4. Dev experiments

### 4.1 Retrieval leave-one-out shift (`loo_shift.py`)

A read-out trained on k-shot prompts needs k-shot prompts for its *training* issues, whose neighbors must come from the other training issues. Plain leave-one-out retrieval is **easier** than held-out retrieval:

| Queries | Neighbor pool | VOTAG@9 (raw) | Median time gap to neighbors |
|---|---|---|---|
| inner, leave-one-out | inner | 0.656 | 77 days |
| dev | inner | 0.615 | 144 days |
| train, leave-one-out | train | 0.659 | 92 days |
| test | train | 0.590 | 208 days |

Neighbors filed close in time are more often the same kind of issue. A read-out trained on leave-one-out prompts would therefore learn to over-trust demonstrations.

**Causal retrieval does not fix this: the benchmark samples each label from a different time window.** The first idea was *causal* retrieval: training queries see only older issues, as in deployment. But within each project, the three labels come from different periods. In the train split:

| Project | Question | Feature | Bug |
|---|---|---|---|
| ansible | 2020-01 – 2020-07 | 2023-04 – 2023-06 | 2023-07 – 2023-08 |
| react | 2017-10 – 2019-08 | 2016-03 – 2018-03 | 2021-09 – 2022-05 |
| flutter | 2019-11 – 2020-01 | 2022-12 – 2023-04 | 2020-09 – 2021-11 |

Each (repo, label) group appears to be "the latest 200 issues with that label", so frequent labels cover short recent windows. Under causal retrieval, an old training question can only see older issues, and those are almost all questions: 73% of inner queries lack 3 eligible examples of some label, against 15% for dev queries. That bias depends on the label and would be even worse than the leave-one-out shift.

**Design decision:**
- No read-out is trained on k-shot prompt states.
- The supervised read-out uses the **zero-shot** prompt state, which involves no retrieval and therefore no shift.
- Retrieval evidence (the LLM's label distribution under the RAGTAG/BRAGTAG prompt, kNN votes) enters only through a low-dimensional stacker fit on **dev** queries. Their retrieval conditions (older index, about 144-day gap) match test's (about 208 days).
- `created_at` is never used as a feature. It is label-predictive on this benchmark, so using it would be a shortcut.

### 4.2 Reference points on dev and test

**Lexical reference (`tfidf_baseline.py`, CPU).** TF-IDF (word 1–2-grams + char 3–5-grams) with multinomial LR, fit on inner and scored on dev. PA plateaus at **0.781** (C=16: bug 0.808, feat 0.775, q 0.758; C=256: 0.782). PS is lower at every C (0.748–0.760). Its distance from the SetFit dev reference (0.833) is reported in §4.9. Surface form such as templates and keywords carries most of the supervised signal, and pooling data across projects helps even for a lexical model.

**Centering, validated with the LLM (the other session, 2026-09-24).** This is the existing baseline component; this study did not run it. Qwen-7B RAGTAG-PS at k=9 with centered neighbors scores 0.714 on test, vs 0.710 for the paper's k=9: +0.004, CI [−0.008, +0.015], a tie. It also ties RAGTAG at its best k (0.718), and BRAGTAG-7B beats it (0.738; −0.024, CI [−0.036, −0.011]). The VOTAG gain (0.590 → 0.639) does not reach the LLM. When centering fixes the neighbor vote, the LLM was usually already right (72%), and the LLM agrees with its examples' vote only 57–59% of the time. **Lesson: kNN/homophily proxies overstate LLM gains.** Nothing in this study is claimed from a proxy alone. Predictions: `bgsulab:~/center_probe_20260924/preds_centered/`.

### 4.3 Implementation pitfall: Unsloth's bare base model is not causal without padding

The extractor must read hidden states. The obvious call, `model.model(input_ids, ...)` (Unsloth's patched base model), **applies no causal mask when the batch has no padding**: batch size 1, or equal-length rows. Unsloth's causal-LM wrapper passes xformers' `LowerTriangularMask` down; the bare call passes nothing. Another session flagged this, and `check_features.py` confirms it on Qwen-3B:

- The bare call's label log-probs differ from `model(...).logits` by **1.3–10.2 nats**.
- The wrapper with `logits_to_keep=1` matches to 0.0000.
- The final state reconstructed from a forward hook (`lm_head(norm(h_last))`) matches to 0.0000.
- Left-padded batches drift by 0.12–0.88 nats relative to single sequences.

`llm_features.py` therefore calls the wrapper at **batch size 1**, as the paper's runs do, so features are exact and independent of batch composition. It captures layer states with forward hooks and re-checks the reconstruction on its first batch.

### 4.4 Idea 1: supervised read-out of the zero-shot decision state (Qwen-7B)

**Setup.** One forward pass per issue with the paper's zero-shot RAGTAG prompt (system prompt, `Title:/Body:`, assistant prefill `<label>`), with no generation. `llm_features.py` records two things:
- The log-probabilities of the three label tokens.
- The hidden states at the answer position (the last prompt token) for layers {14, 18, 21, 24, 28} of 28. It also records mean-pooled issue-token states.

The pass took 612 s for 6,600 issues at a 6.5 GB peak. A multinomial LR on standardized states is fit on *inner* (2,310) and scored on *dev* (990). Script: `dev_study.py probe q7_k0`; log: `dev_logs/probe_q7_k0.txt`.

| Read-out (7B, dev) | Macro F1 | F1 bug / feat / question | q→bug |
|---|---|---|---|
| The LLM's own label, constrained decoding (argmax of 3 label tokens) | 0.688 | 0.706 / 0.811 / 0.547 | 0.491 |
| Same, recalibrated by LR on the 3 log-probs (PA / PS) | 0.733 / 0.743 | 0.738 / 0.812 / 0.680 (PS) | 0.264 |
| Probe, PA, last-token state, L14 / L18 / L21 / L24 / L28 (best C) | 0.800 / **0.822** / 0.815 / 0.815 / 0.820 | L18: 0.822 / 0.847 / 0.796 | 0.103 |
| Probe, PA, mean-pooled state, best layer | 0.795 | | |
| Probe, PS (per-project LR), L18 / L28 | 0.806 / 0.821 | | |
| Probe, **AUG** (Daumé feature augmentation: shared + project-specific weights), L18 / L28 | **0.829** / 0.826 | L18: 0.830 / 0.846 / 0.810 | 0.100 |
| Probe, AUG, average of L18/21/24/28 (C = 0.003) | 0.827 | 0.828 / 0.852 / 0.800 | 0.097 |
| TF-IDF + LR, PA (lexical reference) | 0.781 | 0.808 / 0.775 / 0.758 | 0.085 |

- **The read-out is the big lever.** The model's *own* decision from the same forward pass scores 0.688, and a linear read-out of its state scores about 0.82–0.83 (+0.14). Questions sent to bug drop from 49% to 10%.
- The LLM "knows" much more than its generation expresses. Most of the bias and convention problems of RAGTAG are decision problems, not representation problems.
- Last-token states beat mean pooling by about 0.03. Layers from 18/28 onward are on a plateau.
- Pooling projects with project-specific deviations (AUG) is the best scope, and PS alone is worst, the same ordering TF-IDF shows.
- The differences between the top configurations (±0.01) are within dev noise (SE ≈ 0.012 at n = 990). To limit selection bias, the default read-out is the **layer-averaged AUG probe**, not the single best cell.
- **Per project** (dev), the probe beats TF-IDF on 9/11 projects, including ansible (0.955 vs 0.889), react, vscode and TypeScript. It loses on roslyn and opencv.

### 4.5 Idea 4: fusion (stacker cross-fitted inside dev)

`dev_study.py fusion` (5-fold × 5 repeats inside dev, stratified by project × label):

| Components (7B) | Dev macro F1 |
|---|---|
| Probe (layer-averaged AUG) alone | 0.827 |
| + TF-IDF | 0.834 |
| + TF-IDF + kNN vote (MiniLM PS@15) | 0.838 |
| + TF-IDF + kNN + RAG label scores (RAGTAG-7B, k=12, constrained) | 0.840 |
| + zero-shot label scores | no gain |

RAGTAG-7B at k=12 with constrained decoding scores 0.742 on dev on its own; the paper's test figure with invalids is 0.706. It adds only about +0.002 on top of the read-out. **The supervised read-out already captures most of what 12 in-context demonstrations provide.**

### 4.6 Idea 6: task-conditioned retrieval in decision-state space (training-free)

Nearest neighbors in the LLM's zero-shot decision-state space: states standardized with the index's unlabeled mean/std, cosine, PS index as in the paper. No labels go into the representation. Dev kNN vote (similarity-weighted, `vtag.py` semantics):

| Retriever (dev queries, *inner* index) | Homophily@9 | Vote @3 / @9 / @15 |
|---|---|---|
| MiniLM (paper) | 0.52 | 0.596 / 0.615 / 0.612 |
| MiniLM centered | 0.54 | 0.600 / 0.642 / 0.659 |
| Qwen-7B state L18, PS | 0.68 | 0.761 / 0.751 / 0.754 |
| **Qwen-7B state L28, PS** | **0.72** | 0.769 / **0.784** / 0.765 |
| Qwen-7B state L28, PA | 0.72 | 0.779 / 0.787 / 0.784 |

A training-free vote over decision-state neighbors (0.78–0.79 dev) beats every 7B RAGTAG/BRAGTAG configuration. Per §4.2, kNN proxies can overstate what an LLM gains, so this was validated with an actual LLM run (§4.7).

The decision-state vote is strongly correlated with the read-out: 0.68 correlation of per-issue correctness, since both come from the same representation. Adding it to the read-out stack changes nothing (0.830–0.838).

### 4.7 Does the LLM use better demonstrations? (actual LLM runs, 7B, k=12, dev)

Constrained decoding (argmax over the three label tokens) of the paper's RAGTAG prompt, with three demonstration sources (`dev_rag_scores.py`):

| Prompt (Qwen-7B, k=12, dev) | Macro F1 | F1 bug / feat / question | q→bug | Predicted bug |
|---|---|---|---|---|
| RAGTAG, MiniLM demos (paper) | 0.742 | 0.752 / 0.827 / 0.646 | 0.312 | 0.405 |
| BRAGTAG, margin 3 | 0.748 | 0.739 / 0.820 / 0.684 | 0.230 | 0.344 |
| RAGTAG, **decision-state demos** (L28, PS) | **0.759** | 0.768 / 0.825 / 0.686 | 0.270 | 0.393 |
| *No LLM:* similarity vote over the same decision-state demos | 0.783 | | | |
| Stack of the two (fit on dev, cross-fitted) | 0.795 | | | |

- **Retrieval quality does transfer, partially.** Decision-state demos give +0.017 over RAGTAG and +0.011 over BRAGTAG at the same k and model, without BRAGTAG's heuristic.
- **The generation step is still the bottleneck.** With highly homophilous demos, the LLM's own answer (0.759) is *worse than a plain vote over those demos* (0.783). It agrees with the vote on only 81% of issues and keeps a bug bias (39% predicted bug vs 33% true). This matches the other session's centering finding, and the read-out result in §4.4: the information is in the state, but the decoded label under-uses it.
- **Practical conclusion.** Among training-free methods, the decision-state kNN vote dominates the prompted LLM: one prefill pass per issue, no generation, no invalid outputs, no fitted parameters. Among all methods, the supervised read-out dominates. Generated in-context answers are not needed for either.

### 4.8 Sample efficiency (7B, dev; `sample_efficiency.py`)

Each method sees only a random subset of *inner* with n labeled issues per (project, label), 3 draws each, and is scored on the full dev set:

| n per (project, label) | Total labels | Read-out (AUG, layer avg) | Read-out (PA, final layer) | Decision-state kNN@9 (training-free) | TF-IDF + LR |
|---|---|---|---|---|---|
| 5 | 165 | **0.784** ± 0.011 | 0.771 ± 0.004 | 0.722 ± 0.011 | 0.618 ± 0.022 |
| 10 | 330 | 0.802 ± 0.002 | 0.790 ± 0.004 | 0.742 ± 0.004 | 0.670 ± 0.008 |
| 20 | 660 | 0.816 ± 0.007 | 0.801 ± 0.004 | 0.753 ± 0.007 | 0.718 ± 0.002 |
| 40 | 1,320 | 0.820 ± 0.004 | 0.810 ± 0.006 | 0.762 ± 0.006 | 0.752 ± 0.005 |
| 70 (all inner) | 2,310 | 0.827 | 0.820 | 0.784 | 0.781 |

Same analysis at **14B** (`sample_efficiency.py q14_k0`). Read-out (AUG) at n = 5 / 10 / 20 / 40 / 70: 0.803 / 0.831 / 0.833 / 0.843 / 0.852. Decision-state kNN: 0.727 / 0.759 / 0.777 / 0.804 / 0.813. Test check of n = 10: §6.3.

- **The read-out is extremely sample-efficient.** With 5 labeled issues per class per project (165 labels), it scores 0.784 on dev, about the level of the paper's *best LLM on test* (LoRA FT-14B, 0.785 with 3,300 labels). Dev and test are not directly comparable, but dev tracks test closely (§0). It also beats RAGTAG/BRAGTAG-7B on dev (0.742/0.748), which read the full 210-issue *inner* index of each project.
- The curve saturates after about 20 per class. Most of the value is the LLM representation, not the amount of labels.
- The low-label regime is where the paper argued RAG's niche lies (cold start, no training). There, a linear read-out over a handful of labels beats in-context demonstrations at every budget tested. It is "training-light": a logistic regression fit in seconds on CPU.

### 4.9 Model size (dev; `dev_size.py`)

The read-out default was fixed on 7B *before* scoring the other sizes. It uses:
- AUG scope;
- C = 0.003;
- the average of the per-layer probabilities at depth fractions {0.625, 0.75, 0.875, 1.0}, which is layers 18/21/24/28 for 7B, 22/27/32/36 for 3B, 30/36/42/48 for 14B and 40/48/56/64 for 32B.

| Size | Extraction (6,600 issues, batch 1) | Own label (constrained) | Read-out (default) | Decision-state kNN@9 (final layer, PS; training-free) |
|---|---|---|---|---|
| 3B | 290 s, 2.8 GB | 0.618 | 0.836 | 0.781 |
| 7B | 612 s, 6.5 GB | 0.688 | 0.827 | 0.784 |
| 14B | 1,201 s, 10.8 GB | 0.676 | **0.852** | **0.813** |
| 32B | 903 s on an **OSC H100** (two chunks), 20.3 GB | 0.716 | 0.8517 | **0.819** |

- The read-out nearly removes the dependence on model size on dev: 3B and 7B are within dev noise of each other, and 14B and 32B are tied (0.8518 vs 0.8517). On test, 32B turned out clearly better (§6.1).
- 32B was extracted on OSC Cardinal (1× H100-94GB) with a venv pinned to the lab stack (`osc/`). A cross-hardware check on the 90 ansible dev issues at 14B found hidden-state cosine ≥ 0.9987 (median 0.9999) and 100% agreement of both the own label and the read-out prediction (`xcheck_hw.py`). H100 and 4090 features are interchangeable here.
- At 14B, the neighboring C values give 0.852/0.850, and the best single PA layer gives 0.844.
- The model's own zero-shot decision barely improves with size (0.618 → 0.688 → 0.676 → 0.716). The label bias is not mainly a capacity problem.

**Fusion across sizes and components** (`dev_fusion2.py`, cross-fitted stacker on dev):

| Components | Dev macro F1 |
|---|---|
| 14B read-out alone | 0.852 |
| 14B read-out + TF-IDF | 0.853 |
| 14B read-out + kNN (MiniLM PS@15) | 0.850 |
| 14B read-out + TF-IDF + kNN | **0.856** |
| 3B + 7B + 14B read-outs | 0.843 |
| all five | 0.853 |

Once the 14B read-out is in, retrieval and lexical components add at most about +0.004, which is within dev noise. Combining read-outs across sizes does not help.

**Prior-work reproduction check.** The archival Llama-3.1-8B LoRA run in `results/` (PS) scores 0.505 pooled. It is a legacy run (CLAUDE.md keeps Llama only "for the historical record") and is not used as a bar. The paper's Qwen LoRA baseline follows Heo & Lee's recipe verbatim (§II-E of the paper) and serves as the prior-work reproduction on our split.

**SetFit-PS on dev (the reference bar; `run_setfit_dev.sh`, unmodified `run_setfit.py`, trained on inner).**

| Dev | Macro F1 | F1 bug / feat / question | q→bug |
|---|---|---|---|
| SetFit-PS (issue-adapted MPNet) | **0.833** | 0.826 / 0.861 / 0.811 | 0.094 |
| Read-out 3B / 7B / **14B** | 0.836 / 0.827 / **0.852** | 14B: 0.856 / 0.872 / 0.828 | 14B: 0.076 |
| TF-IDF + LR | 0.781 | | |

On dev, the 14B read-out beats SetFit by +0.019, and 3B/7B are tied with it. The test results (§6.1) reproduce this ordering almost exactly (+0.018, +0.012, +0.003). The dev protocol predicted test well. SetFit took 47 min on the GPU for the 11 per-project dev runs.

**Fusion rule applied (`fusion_rule.py`, preview before 32B).** S* = 14B.
- [R-14B + TF-IDF + kNN] cross-fitted dev stacker: 0.856.
- Adding SetFit-PS: 0.859 (+0.002, below the pre-registered +0.005 threshold), so **SetFit is not included**.
- R-14B + SetFit alone: 0.849.

**Final application, with 32B:** the dev read-outs are 14B 0.85178 and 32B 0.85166, so the rule picks **S* = 14B** by 0.0001. SetFit's gain in the stack is +0.0024, below the threshold, so SetFit is not included. The frozen config is `configs/fusion.json`: [R-14B, TF-IDF, MiniLM kNN PS@15], stacker C = 1.

**What the 14B read-out still gets wrong (dev; `dev_errors.py`).**

| True \ predicted | Bug | Feature | Question |
|---|---|---|---|
| Bug | 276 | 16 | 38 |
| Feature | 14 | 286 | 30 |
| Question | 25 | 24 | 281 |

- A random sample of the remaining question→bug errors is almost entirely bug-shaped reports filed with the bug template: "Steps to Reproduce", `Type: Bug`, segfault logs, "Can't sync signet using 0.21-rc3". This is the template-misuse / maintainer-disposition category from the paper's failure analysis, and likely not recoverable from the initial text.
- **The read-out is well calibrated.** Issues with confidence ≥ 0.9 make up 68% of dev and are 94.5% accurate. Confidence 0.7–0.9 is 72% accurate, below 0.7 about 53%. SetFit assigns p > 0.9 to 95% of test issues at 83% accuracy (§2). The read-out therefore supports selective automation, e.g. auto-labeling the confident two-thirds and routing the rest to a maintainer.

**Cold start for a new project (dev only; `lopo.py`).** Leave-one-project-out: the read-out head (PA, layer average, C = 0.003) is fit on the other 10 projects' inner issues and scored on the held-out project's dev issues, pooled over all 11 held-out projects.

| Dev | 7B | 14B |
|---|---|---|
| LLM's own zero-shot label (constrained) | 0.688 | 0.676 |
| **Read-out, no labels from the target project** | **0.780** | **0.779** |
| Read-out with in-project labels (AUG) | 0.827 | 0.852 |
| *Reference:* RAGTAG-7B k=12 with the target project's labeled index (constrained) | 0.742 | |

- A head trained on *other* projects transfers: +0.09–0.10 over zero-shot, and above RAGTAG even though RAGTAG uses the target project's own labels.
- In-project labels add another 0.05–0.07. That is the per-project label-convention effect, and it varies widely by held-out project: 0.64–0.90 at 7B (roslyn 0.64, TypeScript 0.70, tensorflow 0.67 at 7B; vscode 0.68 at 14B).
- This is dev only; no test evaluation was spent on it.

**Dropped: read-out of k-shot (RAG) prompt states (idea 2).** Training such a read-out needs k-shot states for the training issues. Leave-one-out retrieval is easier than test-time retrieval, and causal retrieval is label-biased (§4.1). The one unbiased option is time-split cross-fitting within each (repo, label) group, which halves the index. Evidence against the effort:
- RAG label scores add only +0.002 on top of the zero-shot read-out (§4.5).
- Decision-state neighbors are redundant with the read-out (§4.6).
- The cost would be about 12× the prefill tokens (≈3–4 h at 14B).

## 5. Pre-registration of test candidates

Written 2026-09-24, ~16:40 lab time, before any test evaluation of a new method. Every choice below was made on dev. The hyperparameters were fixed on 7B dev and carried to other sizes via depth fractions.

**Test protocol.**
- Components are refit on the full train split (3,300) and applied to test (3,300). The stacker, where present, is fit on dev outputs of inner-fit components (§0).
- Predictions go through `final_predict.py`; scoring through `final_eval.py`, which uses `evaluate.py`'s `evaluate_predictions` and pooled macro F1.
- Paired issue-level bootstrap, 2,000 resamples, against: SetFit-PS (issues), SetFit-PS (mpnet), FT-PA-14B, BRAGTAG-PS-32B, and BRAGTAG-PS/RAGTAG-PS at matched size (best k per the paper). For 7B also the centered RAGTAG run (other session, k=9).
- Also reported: per-class F1, q→bug, per-project macro F1 against SetFit-PS, invalid rate, cost.

| ID | Method | Config (`configs/*.json`) | Training |
|---|---|---|---|
| R-3B / R-7B / R-14B / R-32B | **Decision-state read-out**: zero-shot RAGTAG prompt, one prefill, answer-position states at depth fractions {0.625, 0.75, 0.875, 1.0}, per-layer AUG logistic regression (C = 0.003), probabilities averaged | `readout_{size}.json` | Linear head only (seconds, CPU). Same 3,300 labels as SetFit/FT. |
| K-3B / K-7B / K-14B / K-32B | **Decision-state kNN vote**: final-layer state, standardized with the index's unlabeled statistics, cosine, PS index (the paper's), k = 9, similarity-weighted vote | `stateknn_{size}.json` | **Training-free**, no fitted parameters. Same label index as VOTAG/RAGTAG. |
| F | **Fusion**: LR stacker over [R at size S*, TF-IDF LR, MiniLM kNN PS@15]. SetFit-PS probabilities are added **only if** the dev cross-fitted stacker gains ≥ +0.005 with them. S* = the size with the best dev read-out. | written when S* is known | Stacker fit on dev (990). |

- **Planned test evaluations:** 9. Each run is appended to `results/issues11k/exploration/rag_next/test_eval_log.csv`.
- **Primary hypothesis:** R-14B, the best read-out on dev among 3B–14B, beats SetFit-PS (0.810) and FT-PA-14B (0.785) on test. All R and K beat BRAGTAG at matched size.
- **Secondary:** K-* (training-free) beats RAGTAG/BRAGTAG at matched size.

**Addendum (declared ~17:25 lab time, after the six R/K test evaluations of §6 and before running it).** Low-label analysis **R-14B-n10**:
- The R-14B default read-out, but its head is fit on only **10 train issues per (project, label)**: 330 labels, 10% of train. Draws use seeds 0, 1 and 2, giving 3 test evaluations.
- The budget was fixed at "10 per class per project" a priori, as the natural 10% point, before looking at the 14B dev curve. That curve (§4.8, 14B paragraph) gives 0.803 / **0.831** / 0.833 / 0.843 / 0.852 at 5 / 10 / 20 / 40 / 70 per class per project; SetFit trained on all inner labels scores 0.833.
- Question: does the read-out stay competitive with baselines that use 10× the labels?

This takes the planned total to **12** test evaluations.

## 6. Test results

Test evaluations: **12 of 12** planned, all run, no unplanned ones: 8 in §6.1–6.2, 3 low-label draws in §6.3 and the fusion in §6.4 (log: `results/issues11k/exploration/rag_next/test_eval_log.csv`). All are pooled over the 3,300 test issues, scored by `evaluate.py`. CIs are paired issue-level bootstrap, 2,000 resamples. Differences are candidate minus baseline.

### 6.1 Decision-state read-out (R-*, training-light)

| | R-3B | R-7B | R-14B | **R-32B** |
|---|---|---|---|---|
| Macro F1 | 0.8127 | 0.8223 | 0.8278 | **0.8445** |
| F1 bug / feat / question | 0.812 / 0.856 / 0.770 | 0.824 / 0.856 / 0.787 | 0.823 / 0.865 / 0.795 | 0.846 / 0.875 / **0.812** |
| q→bug | 0.197 | 0.182 | 0.179 | **0.152** |
| Invalid outputs | 0 | 0 | 0 | 0 |
| vs SetFit-PS issues (0.8100) | +0.003 [−0.009, +0.015] | +0.012 [−0.001, +0.025] | +0.018 [+0.005, +0.031] | **+0.035 [+0.022, +0.046]** |
| vs SetFit-PS mpnet (0.8053) | +0.007 [−0.005, +0.021] | +0.017 [+0.004, +0.030] | +0.023 [+0.010, +0.036] | +0.039 [+0.026, +0.051] |
| vs FT-PA-14B (0.7853) | +0.027 [+0.014, +0.041] | +0.037 [+0.024, +0.051] | +0.043 [+0.029, +0.056] | +0.059 [+0.046, +0.072] |
| vs BRAGTAG-PS-32B (0.7811) | +0.032 [+0.019, +0.044] | +0.041 [+0.029, +0.054] | +0.047 [+0.034, +0.059] | +0.063 [+0.052, +0.075] |
| vs BRAGTAG-PS, same size | +0.099 [+0.083, +0.115] | +0.085 [+0.070, +0.100] | +0.072 [+0.059, +0.085] | +0.063 [+0.052, +0.075] |
| vs RAGTAG-PS, same size | +0.116 [+0.101, +0.131] | +0.105 [+0.090, +0.120] | +0.096 [+0.082, +0.110] | +0.078 [+0.066, +0.090] |
| vs centered RAGTAG-PS-7B (0.7140) | +0.099 | +0.108 [+0.093, +0.123] | +0.114 | +0.131 [+0.115, +0.145] |
| Projects ≥ SetFit-PS | 6/11 | 9/11 | 9/11 | **10/11** |

**Per project vs SetFit-PS (macro F1 over each project's 300 test issues):**

| Project | R-14B | R-32B | SetFit-PS | R-32B − SetFit |
|---|---|---|---|---|
| ansible | 0.880 | 0.916 | 0.954 | −0.038 |
| bitcoin | 0.788 | 0.800 | 0.764 | +0.036 |
| dart-lang | 0.867 | 0.900 | 0.834 | +0.066 |
| roslyn | 0.735 | 0.736 | 0.720 | +0.016 |
| react | 0.886 | 0.885 | 0.830 | +0.055 |
| flutter | 0.852 | 0.862 | 0.855 | +0.007 |
| kubernetes | 0.913 | 0.930 | 0.898 | +0.033 |
| TypeScript | 0.766 | 0.780 | 0.693 | +0.087 |
| vscode | 0.830 | 0.857 | 0.807 | +0.049 |
| opencv | 0.742 | 0.782 | 0.716 | +0.065 |
| tensorflow | 0.836 | 0.834 | 0.821 | +0.013 |

- **The primary hypothesis holds.** R-14B beats SetFit-PS, the strongest configuration in the study, with an issue-level CI that excludes zero. It beats the best LLM (FT-PA-14B) by 0.043, and BRAGTAG-14B, a same-model comparison, by 0.072. Under project-cluster resampling its lead over SetFit is no longer significant (§6.5).
- **R-32B is the best result of the study: 0.8445.** It beats SetFit-PS by +0.035, and the CI excludes zero under issue-level *and* project-cluster resampling (§6.5). It beats BRAGTAG-32B, the paper's best method at the same size, by +0.063, and is ≥ SetFit on 10/11 projects. Dev had 32B and 14B tied (0.8517 vs 0.8518), so this gain was not predicted by dev. It is reported, not selected: R-32B was a pre-registered candidate.
- R-7B is significantly better than everything except SetFit-PS (issues), with which it is tied. R-3B is tied with SetFit and still significantly beats every LLM configuration in the paper, including FT-14B and BRAGTAG-32B.
- **Question class.** R-32B reaches question F1 0.812 and R-14B 0.795, vs 0.770 for SetFit and 0.729 for BRAGTAG-32B. R-32B sends 15.2% of questions to bug, the lowest of any method on this split; R-14B 17.9%, SetFit 18.3%, BRAGTAG-32B 19.6%, FT-14B 23.4%.
- **Dev → test shift.** On dev the read-outs sent only 7.6–10% of questions to bug and predicted bug for 32–33% of issues. On test these are 18–20% and 38%. The test questions (newer) look more bug-like to every method; SetFit's q→bug is 18.3% on test too. Macro F1 transferred better (dev 0.852 → test 0.828 at 14B) than the error profile did.
- **The one clear loss is ansible** (R-14B −0.074, R-32B −0.038). Post hoc (§6.5), almost all R-14B ansible errors are *questions* predicted as bug (25 of 100). Many are backport-PR-like titles labeled question in this dataset (`[stable-2.10] Update pip tests …`, `Fixed Ansible API Example`). SetFit's per-project contrastive fine-tune learns this convention (99/100 ansible questions right).

### 6.2 Decision-state kNN vote (K-*, training-free)

| | K-3B | K-7B | K-14B | **K-32B** |
|---|---|---|---|---|
| Macro F1 | 0.7730 | 0.7793 | 0.7916 | **0.7993** |
| F1 bug / feat / question | 0.768 / 0.833 / 0.718 | 0.775 / 0.835 / 0.728 | 0.779 / 0.851 / 0.744 | 0.791 / 0.855 / 0.752 |
| vs BRAGTAG-PS, same size | **+0.059 [+0.044, +0.075]** | **+0.042 [+0.027, +0.056]** | **+0.036 [+0.022, +0.048]** | **+0.018 [+0.006, +0.030]** |
| vs RAGTAG-PS, same size | +0.076 [+0.061, +0.092] | +0.062 [+0.047, +0.076] | +0.060 [+0.046, +0.073] | +0.033 [+0.020, +0.045] |
| vs centered RAGTAG-PS-7B | +0.059 | +0.065 [+0.051, +0.080] | +0.078 | +0.085 [+0.071, +0.100] |
| vs FT-PA-14B | −0.012 [−0.027, +0.003] | −0.006 [−0.021, +0.009] | +0.006 [−0.008, +0.021] | +0.014 [−0.000, +0.028] |
| vs BRAGTAG-PS-32B | −0.008 [−0.021, +0.005] | −0.002 [−0.014, +0.011] | +0.011 [−0.002, +0.023] | +0.018 [+0.006, +0.030] |
| vs SetFit-PS issues | −0.037 [−0.050, −0.024] | −0.031 [−0.045, −0.017] | −0.018 [−0.032, −0.005] | −0.011 [−0.024, +0.002] |

- **The secondary hypothesis holds.** With no training and no generation, the decision-state vote beats RAGTAG and BRAGTAG at every size. K-3B (0.773) beats BRAGTAG-14B (0.756).
- K-14B is statistically tied with the best fine-tuned LLM (FT-PA-14B) and with BRAGTAG-32B. **K-32B beats BRAGTAG-32B**, the paper's best configuration, and is **statistically tied with SetFit-PS** (−0.011, CI [−0.024, +0.002]). It does this with no training, no generation, and no fitted parameters.
- Among training-free methods, only adding a trained head (§6.1) moves clearly past SetFit.

### 6.3 Low-label read-out (R-14B-n10, declared addendum; 3 test evaluations)

The head is fit on 10 train issues per (project, label): 330 labels, 10% of the 3,300 every baseline uses. Three random draws:

| Draw | Macro F1 | vs FT-PA-14B (3,300 labels) | vs BRAGTAG-PS-14B (3,300-issue index) | vs BRAGTAG-PS-32B | vs SetFit-PS (3,300 labels) |
|---|---|---|---|---|---|
| seed 0 | 0.7866 | +0.001 [−0.015, +0.016] | +0.031 [+0.017, +0.045] | +0.006 [−0.007, +0.018] | −0.023 [−0.037, −0.010] |
| seed 1 | 0.7970 | +0.012 [−0.002, +0.026] | +0.041 [+0.028, +0.054] | +0.016 [+0.004, +0.028] | −0.013 [−0.027, +0.001] |
| seed 2 | 0.7921 | +0.007 [−0.008, +0.021] | +0.036 [+0.023, +0.049] | +0.011 [−0.002, +0.023] | −0.018 [−0.033, −0.004] |
| **mean ± sd** | **0.792 ± 0.004** | | | | |

- With a tenth of the labels, the read-out **ties the best fine-tuned LLM** trained on all labels (FT-PA-14B).
- It **significantly beats BRAGTAG at the same model size** in every draw, even though BRAGTAG's retrieval index holds all 3,300 labeled issues.
- It does **not** reach SetFit-PS trained on all 3,300 labels.
- **The dev curve was optimistic at low budgets.** Dev scored 0.831 at n=10 vs 0.792 on test, a drop of 0.039, against a drop of 0.024 at full data. Low-label claims should rest on these test numbers, not on §4.8.

### 6.4 Fusion (F, pre-registered rule; 1 test evaluation)

F = LR stacker over [R-14B, TF-IDF LR, MiniLM kNN PS@15], fit on dev (§4.9); S* = 14B by 0.0001; no SetFit.

| | Macro F1 | F1 bug / feat / question | q→bug | Predicted bug | vs SetFit-PS | vs R-14B alone |
|---|---|---|---|---|---|---|
| F | 0.8186 | 0.822 / 0.865 / 0.770 | 0.229 | **0.414** | +0.009 [−0.005, +0.021] | −0.009 |

**Negative result: the stacker hurts on test.** On dev it added +0.004 over R-14B, via cross-fitting inside dev. On test it loses 0.009 to R-14B alone and over-predicts bug (41.4% vs 33% true). The stacker's large bug intercept (+0.58), fit on dev outputs of *inner*-trained components, does not transfer to *train*-refit components on the shifted test period. Fitting a second-level model on 990 dev issues is fragile here. The read-out alone is the better method, and no post-hoc fusion was tried on test.

### 6.5 Post-hoc robustness and complementarity (descriptive; `posthoc_test.py`)

Run after all 12 evaluations. It changes no method.

| Comparison | Diff | Issue-level bootstrap | Exact McNemar (n01 / n10) | **Project-cluster bootstrap** (resample the 11 projects) |
|---|---|---|---|---|
| R-32B vs SetFit-PS | +0.035 | [+0.022, +0.046] | 269 / 155, p < 1e-4 | **[+0.015, +0.054]** |
| R-32B vs FT-PA-14B | +0.059 | [+0.046, +0.072] | 346 / 149, p < 1e-4 | [+0.033, +0.084] |
| R-14B vs SetFit-PS | +0.018 | [+0.005, +0.031] | 253 / 194, p = 0.006 | **[−0.005, +0.036]** |
| R-14B vs FT-PA-14B | +0.043 | [+0.029, +0.056] | 319 / 177, p < 1e-4 | [+0.018, +0.065] |
| F vs SetFit-PS | +0.009 | [−0.005, +0.021] | 242 / 209, p = 0.13 | [−0.019, +0.035] |

- **R-32B beats SetFit even when projects, not issues, are the unit of resampling.** R-14B's lead holds at the issue level but not across projects: it is driven by 9–10 of 11 projects, and ansible's −0.074 is large. Claims about generalizing to *other* projects should rest on R-32B.
- **Complementarity** (per-issue correctness correlation with R-14B; oracle accuracy of the pair):

  | Paired with R-14B | Correlation | Oracle accuracy |
  |---|---|---|
  | SetFit-PS | 0.54 | 0.887 |
  | FT-PA-14B | 0.53 | 0.882 |
  | BRAGTAG-32B | 0.56 | 0.871 |
  | R-32B | 0.78 | 0.867 |

  SetFit and the read-out still make fairly different errors, but a dev-fit combiner did not capture it (§6.4).
- **Temporal shift is shared.** q→bug on test: SetFit 0.183, FT-14B 0.234, BRAGTAG-32B 0.196, R-14B 0.179, **R-32B 0.152**. It is roughly twice the dev level for every method.

### 6.6 Cost

Measured on the lab RTX 4090 at batch size 1, excluding model load, as the paper does. Extraction covers all 6,600 issues: train issues to fit the head or build the state index, and test issues to predict. Each half costs about half.

| Method | GPU work | Peak GPU memory | CPU fit | Invalid |
|---|---|---|---|---|
| R-3B / K-3B | 290 s (0.08 h) | 2.8 GB | LR 20 s / kNN 0.1 s | 0 |
| R-7B / K-7B | 612 s (0.17 h) | 6.5 GB | LR 38 s / kNN 0.2 s | 0 |
| R-14B / K-14B | 1,201 s (0.33 h) | 10.8 GB | LR 70 s / kNN 0.1 s | 0 |
| R-32B / K-32B | 903 s (0.25 h) on an **OSC H100** (two 451-s chunks) | 20.3 GB | LR 55 s / kNN 0.1 s | 0 |
| *Paper:* BRAGTAG-PS 14B / 32B | 1.35 h / 4.15 h (inference, 3,300 test) | 12.6 / 22.3 GB | — | 4.7% / 4.0% |
| *Paper:* LoRA FT-PA 14B | 1.49 h train + 0.22 h infer | 16.7 GB | — | 0.03% |
| *Paper:* SetFit-PS (issues) | 59.3 min train + 0.2 min infer | 21.0 GB | — | 0 |

The 14B read-out needs about a quarter of BRAGTAG-14B's GPU time for the same model, with less memory and no generation. It costs about a fifth of LoRA-14B and less GPU memory than any baseline except BRAGTAG at the same size. The head is a logistic regression fit in about a minute on CPU.

Caveat: the paper's BRAGTAG/FT timings were measured partly on the L40/A6000 (see `CLAUDE.md`), and the 32B extraction ran on an H100. Ratios are indicative, not a controlled benchmark. On the 4090, 32B extraction would take roughly 2.5× the 14B time (≈ 50 min), still under a third of BRAGTAG-32B's 4.15 h. The OSC run cost two quarter-node H100 jobs of about 10 min each on project PCS0289.


## 7. Verdict

**Did anything beat SetFit-PS (0.810), the best LLM (0.785), and BRAGTAG/centering at matched size? Yes.**

| Candidate | Test macro F1 | vs SetFit-PS | vs FT-PA-14B | vs BRAGTAG, same size | Training |
|---|---|---|---|---|---|
| **R-32B, decision-state read-out** | **0.8445** | **+0.035 [+0.022, +0.046]**; project-cluster CI [+0.015, +0.054] | +0.059 | +0.063 | Linear head on the 3,300 train labels |
| R-14B, decision-state read-out | 0.8278 | +0.018 [+0.005, +0.031]; project-cluster CI [−0.005, +0.036] | +0.043 | +0.072 | Linear head |
| R-7B / R-3B | 0.8223 / 0.8127 | tie / tie | +0.037 / +0.027 | +0.085 / +0.099 | Linear head |
| **K-32B, decision-state kNN vote** | **0.7993** | −0.011 [−0.024, +0.002], tie | +0.014, tie | **+0.018 [+0.006, +0.030]** | **None** (training-free) |
| K-14B / K-7B / K-3B | 0.792 / 0.779 / 0.773 | below | tie | +0.036 / +0.042 / +0.059 | None |
| R-14B with 10% of the labels (330) | 0.792 ± 0.004 | below | tie | +0.036 | Linear head |
| F, fusion (pre-registered) | 0.8186 | tie | +0.033 | +0.063 | Linear head + dev stacker |

- **Best result: R-32B at 0.8445.** It is the best number on this split, above SetFit-PS by +0.035 (robust to resampling projects), above the paper's best LLM by +0.059, and above BRAGTAG-32B by +0.063.
  - Question F1 0.812 (SetFit 0.770); q→bug 15.2%, the lowest of any method.
  - ≥ SetFit on 10/11 projects; zero invalid outputs.
  - Cost: 15 GPU-minutes on one H100 for all 6,600 issues (train + test), 20.3 GB, plus a one-minute CPU logistic regression.
  - It is **training-light, not training-free:** a linear head is fit on the same 3,300 labels SetFit and LoRA use. The LLM is never fine-tuned and no gradient flows through it.
- **Matched size:** every read-out beats BRAGTAG and RAGTAG at its size by +0.063 to +0.116. The centered RAGTAG-7B baseline (0.714) is beaten by +0.10 to +0.13.
- **Best training-free result: K-32B at 0.799.** The decision-state kNN vote beats BRAGTAG-32B and ties SetFit-PS and LoRA FT-14B, with no training, no generation and no fitted parameter. At every size it beats the paper's training-free methods (RAGTAG/BRAGTAG) by +0.018 to +0.076.
- **Honest negatives:**
  - The pre-registered fusion (F) *lost* to its own base read-out on test (§6.4).
  - The low-label read-out ties LoRA-14B but does not reach SetFit (§6.3).
  - The dev split under-predicted 32B (tied with 14B on dev, +0.017 over it on test) and over-predicted low-label performance.

**What made the difference.** The paper's generative pipeline discards information the LLM already has:
- From one forward pass, the model's own constrained label scores 0.62–0.72 on dev (3B–32B), while a linear read-out of the same state scores 0.83–0.85. On test the read-out reaches 0.81–0.84.
- Better retrieval helps a prompted LLM only a little (+0.017), and the LLM's answer stays *worse than a plain vote over its own demonstrations* (§4.7).
- For this task, the effective use of retrieval-style resources is:
  1. a supervised read-out over all labeled issues, which learns project label conventions as SetFit does;
  2. training-free, retrieval in the LLM's own task-conditioned state space with a vote.

  Generation is not needed in either.

**What limits it (evidence):**
1. **A noise floor from the labels.** About 9% of test issues are misclassified by ≥90% of all 99 configurations. The remaining dev question→bug errors are bug-template reports labeled question (§4.9). The paper and Colavito et al. attribute these to template misuse and maintainer disposition, which the initial text cannot reveal.
2. **Temporal shift.** On test, every method sends roughly twice as many questions to bug as on dev (SetFit 18.3%, R-14B 17.9%, R-32B 15.2%, vs 7–10% on dev). The newer test questions look more bug-like to all methods, and a dev-fit combiner amplified the shift (§6.4).
3. **Project conventions that a pooled head misses.** On ansible, the read-out (R-14B 0.880, R-32B 0.916) stays behind SetFit's per-project contrastive fine-tune (0.954). Its errors are backport-PR-like issues that this dataset labels question.
4. **Retrieval adds little once a read-out exists.** TF-IDF, kNN and RAG scores add at most +0.004 on dev. The headroom left for RAG-style evidence on top of a trained read-out looks small on this benchmark.

**Is it still "RAG"?** Only partly, and the notebook says so. The best method is not retrieval-augmented. The retrieval contribution that survives testing is the training-free decision-state vote: retrieval in the LLM's own task-conditioned representation instead of a generic sentence embedder, with no generation. It is the strongest training-free method on this benchmark by a wide margin.

**Prior art (what is not new).**
- Linear probes on LLM hidden states beat in-context learning on general text classification: Cho et al. 2023, cited in *Logistic Regression makes small LLMs strong and explainable "tens-of-shot" classifiers* (arXiv:2408.03414, 2024); Abbas et al., *Enhancing In-context Learning via Linear Probe Calibration* (AISTATS 2024).
- kNN over an LLM's output distribution is *kNN Prompting* (Xu et al., ICLR 2023).
- Feature augmentation for domain adaptation is Daumé III (2007).

What this study adds for issue report classification (IRC):
- The evidence, on the paper's own benchmark and models, that the generation step, not the representation or the retrieval, limits RAGTAG/BRAGTAG.
- That a read-out of the *zero-shot RAGTAG prompt* state beats SetFit and every LLM configuration, including LoRA.
- That retrieving demonstrations in that state space beats sentence-embedding retrieval for both voting and prompting.
- That a head trained on other projects transfers to a new project (dev, §4.9).
- The benchmark trap: label-dependent time windows make causal retrieval biased and leave-one-out retrieval optimistic.

A focused literature search would be needed before any priority claim.

**Caveats.**
- SetFit-PS is a single archival seed (42).
- R-32B was extracted on an H100 (OSC), the other sizes on the lab 4090. The 14B cross-hardware check (§4.9) shows the features are interchangeable to within bf16 rounding.
- Test evaluations: 12 planned and pre-registered (§5), plus none unplanned; all are reported, including the weaker ones.
- Hyperparameters were chosen on 7B dev and carried to other sizes; no test-time selection.
- Heo & Lee's GPT-4o/Llama numbers are on a different, easier split and remain uncompared (§1).
- The read-out uses the full 8k-token prompt, while SetFit truncates at 512 tokens. Both see the same issue text, but not the same amount of it.
- The dev→test gap is larger at low label budgets (§6.3).

## 8. Reproduction

Run on the lab machine: `ssh bgsulab`, `cd ~/llm-labler`. The Python scripts import their sibling modules, so run them from `scripts/experiments/rag_next/` with `../../../venv/bin/python`. All outputs go to `results/issues11k/exploration/rag_next/`.

```bash
cd ~/llm-labler/scripts/experiments/rag_next
PY=../../../venv/bin/python

# 1. Headroom analysis of the existing predictions (read-only over results/issues11k)
$PY collect_preds.py && $PY headroom.py

# 2. Dev split + retrieval (the paper's MiniLM recipe; verifies rank-exact agreement with the paper)
$PY make_splits.py && $PY neighbors.py && $PY make_ps_csvs.py && $PY loo_shift.py && $PY dev_votag.py

# 3. GPU jobs, via the shared-GPU queue (waits for 120 s of GPU idleness before each job)
#    queue lines "<name>|<command>" in results/issues11k/exploration/rag_next/queue.txt, e.g.
#    check3b|venv/bin/python scripts/experiments/rag_next/check_features.py unsloth/Qwen2.5-3B-Instruct-bnb-4bit
#    q14_k0|venv/bin/python scripts/experiments/rag_next/llm_features.py --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit \
#           --k 0 --roles inner,dev,test --max_seq_length 8192 --out results/issues11k/exploration/rag_next/features/q14_k0
#    (same for 3B/7B/32B; RAG dev prompts: --k 12 --neighbors features/nb_PS_raw_dev.npz --roles dev [--debias_margin 3])
#    setfit_dev_issues|bash scripts/experiments/rag_next/run_setfit_dev.sh Collab-uniba/github-issues-mpnet-st-e10
tmux new-session -d -s ragnext "bash ~/llm-labler/scripts/experiments/rag_next/gpu_queue.sh"

# 4. Decision-state retrieval indexes (label-free)
$PY state_neighbors.py q3_k0 36 q3L36; $PY state_neighbors.py q7_k0 28 q7L28
$PY state_neighbors.py q14_k0 48 q14L48; $PY state_neighbors.py q32_k0 64 q32L64

# 5. Dev studies (fit on inner, score on dev)
$PY tfidf_baseline.py
$PY dev_study.py probe q7_k0                 # read-out grid (layers, pooling, C, PA/PS/AUG)
$PY dev_size.py q14_k0                       # per-size default read-out + decision-state kNN
$PY dev_fusion2.py q3_k0,q7_k0,q14_k0        # stacking across sizes/components
$PY dev_rag_scores.py                        # constrained RAGTAG/BRAGTAG/decision-state-demo prompts
$PY sample_efficiency.py q7_k0; $PY sample_efficiency.py q14_k0
$PY dev_errors.py q14_k0 30,36,42,48; $PY lopo.py q14_k0
$PY fusion_rule.py                            # applies the pre-registered fusion rule, writes configs/fusion.json

# 5b. 32B on OSC Cardinal (H100), from the LOCAL PC (bgsulab cannot reach OSC):
#     cd <local clone>; rsync bgsulab:~/llm-labler/results/issues11k/exploration/rag_next/splits/pool.csv /tmp/pool.csv
#     bash scripts/experiments/rag_next/osc/sync_to_osc.sh /tmp/pool.csv        # code + split -> /fs/ess/PCS0289/rag_next
#     ssh alirezzzhp1378@cardinal.osc.edu 'cd /fs/ess/PCS0289/rag_next && sbatch repo/scripts/experiments/rag_next/osc/setup_env.sbatch'
#       (CPU: venv pinned to the lab stack + Qwen2.5-32B/14B bnb-4bit prefetch, ~5 min)
#     ssh ... 'cd /fs/ess/PCS0289/rag_next && F=$PWD/repo/results/issues11k/exploration/rag_next/features && \
#       XCHECK_14B=1 sbatch --partition=debug --time=01:00:00 --job-name=ragnext-q32a repo/scripts/experiments/rag_next/osc/extract.sbatch \
#         --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit --k 0 --roles inner,dev --max_seq_length 8192 --out $F/q32_k0_innerdev && \
#       XCHECK_14B=0 sbatch --partition=debug --time=01:00:00 --job-name=ragnext-q32b repo/scripts/experiments/rag_next/osc/extract.sbatch \
#         --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit --k 0 --roles test --max_seq_length 8192 --out $F/q32_k0_test'
#       (the regular gpu partition quoted a 2-day queue; the 1-h debug partition started within minutes)
#     copy features/q32_k0_* + xcheck_q14_ansible_dev.* OSC -> local -> bgsulab, then on bgsulab:
$PY merge_features.py q32_k0 q32_k0_innerdev q32_k0_test && $PY xcheck_hw.py && $PY dev_size.py q32_k0

# 6. Test (pre-registered configs in configs/; every final_eval.py call is logged)
$PY final_predict.py configs/readout_14B.json ../../../results/issues11k/exploration/rag_next/test_preds/readout_14B.csv
$PY final_eval.py ../../../results/issues11k/exploration/rag_next/test_preds/readout_14B.csv readout_14B --size 14B
# same pattern for readout_{3B,7B,32B}, stateknn_{3B,7B,14B,32B}, readout_14B_n10_s{0,1,2}, fusion
$PY posthoc_test.py                          # descriptive, after all evaluations
```

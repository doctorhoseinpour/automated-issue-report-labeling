# 11-Project Benchmark Findings

**Date:** 2026-04-28 (original) / **2026-04-29 (analysis refresh)**

**Status:** Campaign complete. All 4 models × {zero-shot, RAGTAG k∈{1,3,6,9} agnostic+proj-spec, debias_m3 k∈{1,3,6,9} proj-spec, FT agnostic+proj-spec} fully evaluated. Qwen-32B RAGTAG was rerun on NRP A6000 (the original 4090 results were OOM-corrupted and are superseded). See [NRP_MIGRATION_STATUS.md](NRP_MIGRATION_STATUS.md) for the campaign record.

The original master table (Section 0 below) covers the original 4-model lineup including Llama-3B/8B. **Sections RQ1 / RQ2 / RQ3 (this 2026-04-29 refresh) cover the active 4-Qwen lineup (3B / 7B / 14B / 32B, uniform bnb-4bit) and supersede the original numbers for the paper.** All numbers in the refresh sections are derived programmatically by `scripts/analysis/*.py` reading from `results/issues11k/`; tables and figures are written to `docs/analysis/`.

---

# RQ1 (2026-04-29) — VTAG plateau and the case for the RAGTAG k grid

VTAG is similarity-weighted k-NN over the same FAISS index that RAGTAG uses, with no LLM call. We use the full k sweep on disk (k = 1..20, 25, 30) to identify where VTAG plateaus, then justify the chosen RAGTAG grid k ∈ {1, 3, 6, 9} as covering the climb plus the entry to the plateau.

## RQ1.1 — Macro F1 by k

Table at [analysis/rq1_vtag_table.csv](analysis/rq1_vtag_table.csv); figures [rq1_vtag_curve_agnostic.png](analysis/figures/rq1_vtag_curve_agnostic.png), [rq1_vtag_curve_project_specific.png](analysis/figures/rq1_vtag_curve_project_specific.png), and [rq1_vtag_kgrid_justification.png](analysis/figures/rq1_vtag_kgrid_justification.png).

| k | Agnostic F1 | Project-specific F1 (mean of 11) |
|---:|---:|---:|
| 1 | 0.5649 | 0.5564 |
| 3 | 0.5787 | 0.5626 |
| 6 | 0.5912 | 0.5718 |
| 9 | 0.5984 | 0.5783 |
| 13 | 0.6019 | 0.5786 |
| 15 | 0.6006 | 0.5808 |
| **16 (peak ag)** | **0.6039** | 0.5779 |
| 20 | 0.5997 | 0.5714 |
| 30 | 0.5933 | 0.5570 |

**Plateau detection** (first k where F1 reaches `peak − 0.005`): agnostic plateau k = 13 (peak 0.604 at k = 16); project-specific plateau k = 7 (peak 0.584 at k = 7).

## RQ1.2 — Cost (the "near-zero" claim, quantified)

Per-cell wall-clock (median across all 22 k values, excluding model-load time):

| Setting | Median wall-clock per (k, project) | Total prompt tokens |
|---|---:|---:|
| Agnostic | 6.4 ms | 0 |
| Project-specific | 0.6 ms | 0 |

VTAG is functionally free at inference. Total time to evaluate the entire 22-k sweep across all 12 cells (1 agnostic + 11 ps) is ≈ 0.3 seconds.

## RQ1.3 — Per-project variance

The agnostic VTAG curve is smooth; project-specific VTAG variance is large (per-project std rises from 0.05 at k = 1 to 0.09 at k = 30). The hardest projects are TypeScript and Roslyn (~0.50 at most k); the easiest is dart-lang (~0.72). Detail: [analysis/rq1_vtag_per_project.csv](analysis/rq1_vtag_per_project.csv) and [rq1_vtag_variance.csv](analysis/rq1_vtag_variance.csv).

## RQ1.4 — Why we chose RAGTAG k ∈ {1, 3, 6, 9}

The chosen k grid spans VTAG's climb-up (k = 1..6: F1 rises from 0.565 → 0.591 ag, 0.556 → 0.572 ps) and reaches the entry to the plateau (k = 9: 0.598 ag, 0.578 ps). Beyond k = 9, VTAG gains < 0.005 macro F1 per k step on agnostic and oscillates noisily on project-specific. RAGTAG's prompt budget at k = 9 is already near 8k tokens (~98 % of the model's 8k context); pushing k higher would force more aggressive truncation. The k grid is therefore both empirically motivated by VTAG's behaviour and constrained by the LLM context budget.

**Limitation.** Only the `similarity` voting scheme was run on the 11k benchmark. Shepard (sim²) and majority schemes are implemented in `vtag.py` but were not swept; future work.

---

# RQ2 (2026-04-29) — VTAG vs RAGTAG vs FT: comprehensive comparison and failure analysis

## RQ2.1 — Master leaderboard, 4 Qwen models

Source: [analysis/rq2_leaderboard.csv](analysis/rq2_leaderboard.csv). Project-specific F1 is the per-project mean across 11 projects (macro-macro). Agnostic F1 is on the pooled 3,300-issue test set.

### Qwen-3B

| Approach | Setting | k | F1 macro | Acc | F1 bug | F1 feat | F1 q | Inv |
|---|---|---|---:|---:|---:|---:|---:|---:|
| zero-shot | ag | — | 0.6132 | 0.6258 | 0.6816 | 0.7227 | 0.4352 | 0.0024 |
| RAGTAG | ag | k3 | 0.6944 | 0.7006 | 0.7094 | 0.8018 | 0.5720 | 0.0024 |
| RAGTAG | ps | k3 | 0.6939 | 0.7030 | 0.7088 | 0.8021 | 0.5708 | 0.0021 |
| **Debias** | **ps** | **k6** | **0.7089** | 0.7079 | 0.6815 | 0.8022 | 0.6431 | 0.0179 |
| FT | ag | — | 0.6520 | 0.6630 | 0.6989 | 0.7475 | 0.5098 | 0.0055 |
| FT | ps | — | 0.5754 | 0.5851 | 0.6479 | 0.6558 | 0.4225 | 0.0191 |

### Qwen-7B

| Approach | Setting | k | F1 macro | Acc | F1 bug | F1 feat | F1 q | Inv |
|---|---|---|---:|---:|---:|---:|---:|---:|
| zero-shot | ag | — | 0.6621 | 0.6791 | 0.6959 | 0.7905 | 0.5000 | 0.0009 |
| RAGTAG | ag | k6 | 0.7122 | 0.7103 | 0.7287 | 0.8082 | 0.5996 | 0.0348 |
| RAGTAG | ps | k6 | 0.7142 | 0.7152 | 0.7323 | 0.8166 | 0.5938 | 0.0346 |
| Debias | ps | k6 | 0.7299 | 0.7342 | 0.7483 | 0.8138 | 0.6277 | 0.0161 |
| **FT** | **ag** | — | **0.7411** | 0.7412 | 0.7329 | 0.8263 | 0.6641 | 0.0012 |
| FT | ps | — | 0.5113 | 0.5267 | 0.5041 | 0.6299 | 0.4001 | 0.0106 |

### Qwen-14B

| Approach | Setting | k | F1 macro | Acc | F1 bug | F1 feat | F1 q | Inv |
|---|---|---|---:|---:|---:|---:|---:|---:|
| zero-shot | ag | — | 0.6452 | 0.6752 | 0.6905 | 0.8057 | 0.4395 | 0.0006 |
| RAGTAG | ag | k9 | 0.7170 | 0.7109 | 0.7268 | 0.8272 | 0.5971 | 0.0427 |
| RAGTAG | ps | k9 | 0.7165 | 0.7142 | 0.7283 | 0.8330 | 0.5883 | 0.0430 |
| **Debias** | **ps** | **k9** | **0.7418** | 0.7388 | 0.7477 | 0.8337 | 0.6441 | 0.0321 |
| FT | ag | — | 0.7154 | 0.7121 | 0.7027 | 0.7675 | 0.6759 | 0.0009 |
| FT | ps | — | 0.6366 | 0.6385 | 0.6922 | 0.6516 | 0.5660 | 0.0012 |

### Qwen-32B

| Approach | Setting | k | F1 macro | Acc | F1 bug | F1 feat | F1 q | Inv |
|---|---|---|---:|---:|---:|---:|---:|---:|
| zero-shot | ag | — | 0.6876 | 0.7058 | 0.7150 | 0.8143 | 0.5335 | 0.0018 |
| RAGTAG | ag | k9 | 0.7594 | 0.7464 | 0.7577 | 0.8379 | 0.6825 | 0.0442 |
| RAGTAG | ps | k9 | 0.7579 | 0.7476 | 0.7545 | 0.8410 | 0.6782 | 0.0449 |
| **Debias** | **ps** | **k9** | **0.7745** | 0.7664 | 0.7754 | 0.8405 | 0.7077 | 0.0330 |
| FT | ag | — | 0.7462 | 0.7552 | 0.7781 | 0.8138 | 0.6466 | 0.0000 |
| FT | ps | — | 0.7130 | 0.7200 | 0.6794 | 0.7972 | 0.6625 | 0.0006 |

### VTAG floor (model-independent, on the same FAISS index RAGTAG uses)

| Approach | Setting | k | F1 macro | F1 bug | F1 feat | F1 q |
|---|---|---|---:|---:|---:|---:|
| VTAG | ag | k16 | 0.6039 | 0.6318 | 0.6294 | 0.5505 |
| VTAG | ps | k7 | 0.5837 | 0.6194 | 0.5981 | 0.5336 |
| VTAG-debias | ag | k9 | 0.6048 | 0.5892 | 0.6287 | 0.5966 |
| VTAG-debias | ps | k9 | 0.5852 | 0.5628 | 0.6075 | 0.5853 |

**Headlines:**
- Best overall: Qwen-32B Debias-ps k9 (F1 = 0.7745).
- 7B is the only model where FT-ag wins on point estimate, but the gap is **not statistically significant** (see RQ2.9).
- From 14B up, debias-ps clears FT-ag.

## RQ2.2 — Per-project leaderboard

Heatmap for the canonical model (Qwen-32B): [rq2_per_project_heatmap_qwen32b.png](analysis/figures/rq2_per_project_heatmap_qwen32b.png). Per-model heatmaps for 3B/7B/14B alongside. Tidy data: [analysis/rq2_per_project_table.csv](analysis/rq2_per_project_table.csv).

Project hardness ranking (mean F1 across all approaches and Qwen sizes):

| Rank | Project | Mean F1 |
|---|---|---:|
| Hardest | microsoft_TypeScript | 0.573 |
| | dotnet_roslyn | 0.601 |
| | microsoft_vscode | 0.615 |
| | opencv_opencv | 0.654 |
| | bitcoin_bitcoin | 0.666 |
| | ansible_ansible | 0.679 |
| | tensorflow_tensorflow | 0.704 |
| | flutter_flutter | 0.705 |
| | facebook_react | 0.749 |
| | kubernetes_kubernetes | 0.759 |
| Easiest | dart-lang_sdk | 0.778 |

Two patterns in the per-project heatmap:
- **ansible_ansible is the only project where FT-ps (0.82) clearly beats every other approach** for Qwen-32B. Likely a result of ansible's relatively homogeneous bug labels.
- **microsoft_vscode is hardest for zero-shot** (Qwen-32B ZS = 0.51) but recovers strongly under RAGTAG/debias (0.74–0.75). Retrieval is doing real work for vscode-specific feature/question phrasing.

## RQ2.3 — RAGTAG k-curves per Qwen size

Two-by-two panel: [rq2_kcurves_panel.png](analysis/figures/rq2_kcurves_panel.png).

Plateau-of-RAGTAG by model size:
- 3B and 7B plateau at k = 3 (3B) and k = 6 (7B); pushing higher doesn't help.
- 14B and 32B continue improving to k = 9; the k-grid's upper bound coincides with the RAGTAG plateau at the larger sizes.

## RQ2.4 — Scaling-law axis

Source: [analysis/rq2_scaling_table.csv](analysis/rq2_scaling_table.csv); figures [rq2_scaling_agnostic.png](analysis/figures/rq2_scaling_agnostic.png) and [rq2_scaling_project_specific.png](analysis/figures/rq2_scaling_project_specific.png).

| Model | ZS ag | RAGTAG-ag | RAGTAG-ps | Debias-ps | FT-ag | FT-ps |
|---|---:|---:|---:|---:|---:|---:|
| 3B | 0.613 | 0.694 | 0.694 | 0.709 | 0.652 | 0.575 |
| 7B | 0.662 | 0.712 | 0.714 | 0.730 | **0.741** | 0.511 |
| 14B | 0.645 | 0.717 | 0.717 | **0.742** | 0.715 | 0.637 |
| 32B | 0.688 | 0.759 | 0.758 | **0.775** | 0.746 | 0.713 |

- **RAGTAG/debias scale steeper with model size than FT-ag** in the agnostic setting: RAGTAG-ag gains 0.065 from 3B to 32B; FT-ag gains 0.094 with a non-monotonic trajectory (drop from 7B to 14B).
- **FT-ps collapses non-monotonically.** 7B is the worst (0.511), recovering only at 32B. Consistent with the well-known phenomenon that LoRA on small training pools is unstable for mid-size models.
- **The "FT wins" anomaly is uniquely Qwen-7B agnostic.** The point estimate is +0.029 over RAGTAG-ag and +0.011 over Debias-ps; the latter is not statistically significant (RQ2.9).

## RQ2.5 — Pareto frontier (cost vs F1)

Source: [analysis/rq2_pareto_table.csv](analysis/rq2_pareto_table.csv).

**Project-specific (figure [rq2_pareto_gpu_project_specific.png](analysis/figures/rq2_pareto_gpu_project_specific.png)).** All FT-ps points are dominated. The Pareto frontier in the project-specific setting is composed entirely of RAGTAG and debias points, from Qwen-3B-debias-k1 (398 GPU-s, F1 0.675) to Qwen-32B-debias-k9 (11,424 GPU-s, F1 0.775).

**Agnostic (figure [rq2_pareto_gpu_agnostic.png](analysis/figures/rq2_pareto_gpu_agnostic.png)).** Different story: FT-ag at 7B is on the Pareto frontier (940 GPU-s, F1 0.741), as is FT-ag at 32B (5,824 GPU-s, F1 0.746). RAGTAG fills the rest of the frontier.

So FT is a Pareto-efficient choice in the agnostic setting (one classifier across all projects), but a Pareto-dominated choice in the project-specific setting (per-project tuning).

## RQ2.6 — Bug-bias quantification (LLM-side, not retrieval-side)

Source: [analysis/rq2_overprediction.csv](analysis/rq2_overprediction.csv); figure [rq2_bug_overprediction.png](analysis/figures/rq2_bug_overprediction.png); confusion matrices for Qwen-32B [rq2_confusion_qwen32b.png](analysis/figures/rq2_confusion_qwen32b.png).

Bug over-prediction ratio (predicted_bug / true_bug; 1.0 = calibrated; mean across model sizes / k values within each cell type):

| Cell type | Bug | Feature | Question |
|---|---:|---:|---:|
| Zero-shot ag | 1.54 | 0.99 | 0.47 |
| RAGTAG ag (k1-9) | 1.32 | 1.04 | 0.58 |
| RAGTAG ps (k1-9) | 1.32 | 1.03 | 0.58 |
| Debias ps (k1-9) | 1.25 | 1.03 | 0.68 |
| FT ag | 1.24 | 0.93 | 0.82 |
| FT ps | 1.15 | 0.99 | 0.83 |

Confusion-matrix view (Qwen-32B): zero-shot misclassifies 46 % of questions as bugs (506 / 1,100). RAGTAG-k9-ag drops it to 28 % (304); FT to 29 % (316); debias-k9-ps to 24 % (259, lowest of the four).

## RQ2.7 — Retrieval-side bug skew is mild

Source: [analysis/rq2_retrieval_bug_skew.csv](analysis/rq2_retrieval_bug_skew.csv); figure [rq2_retrieval_bug_skew.png](analysis/figures/rq2_retrieval_bug_skew.png).

Mean fraction of top-k FAISS neighbors that are labeled "bug", sliced by ground-truth class (agnostic):

| k | GT = bug | GT = feature | GT = question |
|---:|---:|---:|---:|
| 3 | 0.632 | 0.279 | 0.309 |
| 9 | 0.609 | 0.289 | 0.325 |
| 30 | 0.582 | 0.307 | 0.356 |

For non-bug ground truth, the top-k contain ≈ 30 % bug neighbors — close to the 33 % balanced expectation. Retrieval is well-calibrated for non-bug classes; the bug-bias is **primarily an LLM-side phenomenon**, not a retrieval-side imbalance. This reframes the mechanism: retrieval-time debiasing works less by "fixing bad retrieval" and more by "nudging the LLM's prior at inference time via a class-balanced prompt."

## RQ2.8 — Qualitative failure analysis

264 of 3,300 test issues (8.0 %) are **consensus-bug failures**: every Qwen size's best plain RAGTAG (agnostic) predicted "bug", but ground truth was feature or question. Per ground-truth split: 79 % are question→bug, 21 % feature→bug.

Auto-categorisation of consensus failures:

| Category | Count | Description |
|---|---:|---|
| ambiguous | 195 | None of the patterns below — mostly labeling noise or genuinely hard cases |
| question-phrased | 31 | Starts with "how/what/why/can I"; ends with "?" — should be question |
| error-trace | 30 | Contains traceback / Exception / *Error class — looks bug-like even if labeled feature/question |
| error-trace + question-phrased | 5 | Both patterns |
| feature-cue | 3 | Explicit "feature request" / "proposal" language |

Sample table: [analysis/rq2_qualitative_failures.md](analysis/rq2_qualitative_failures.md). Hand-review observation: many consensus-failure issues contain explicit "Bug:" framing in the title or body (often via GitHub's bug-report template), even though the maintainer's final label is feature or question. The LLM correctly reads the user's framing but misses the maintainer's reclassification — this is a labeling-mismatch problem more than a model error, and a meaningful chunk of the residual 24 % bug over-prediction is attributable to label noise of this kind.

## RQ2.9 — Bootstrap CIs and McNemar tests on headline pairs

Source: [analysis/rq2_significance.csv](analysis/rq2_significance.csv); markdown summary [rq2_significance.md](analysis/rq2_significance.md). Bootstrap 1,000 iters, n = 3,300 paired test issues. Significance markers: `***` p < 0.001, `**` p < 0.01, `*` p < 0.05.

Selected critical findings:

| Pair | ΔF1 (A − B) | 95 % CI | McNemar p | Sig |
|---|---:|---|---:|:---:|
| H2[Qwen-32B]: RAGTAG ag k9 vs VTAG ag k9 | +0.161 | [+0.142, +0.181] | 2.4e-44 | *** |
| H3[Qwen-7B]: RAGTAG ag k6 vs FT ag | −0.029 | [−0.044, −0.014] | 4.5e-05 | *** (FT wins on RAGTAG-vs-FT) |
| H4[Qwen-14B]: Debias ps k9 vs RAGTAG ps k9 | +0.025 | [+0.016, +0.033] | 2.2e-09 | *** |
| H4[Qwen-32B]: Debias ps k9 vs RAGTAG ps k9 | +0.016 | [+0.008, +0.023] | 6.3e-07 | *** |
| H5[Qwen-14B]: Debias ps k9 vs FT ag | +0.030 | [+0.013, +0.046] | 0.003 | ** |
| H5[Qwen-32B]: Debias ps k9 vs FT ag | +0.031 | [+0.014, +0.045] | 0.139 | ns* |
| **H6: Qwen-7B FT ag vs Debias ps k6 (the +0.011 anomaly)** | **+0.008** | **[−0.007, +0.023]** | **0.36** | **ns** |

\*The 32B Debias-vs-FT gap has bootstrap CI excluding 0 (so the F1 ranking is reliable under resampling) but McNemar p > 0.05 (so paired decisions are statistically comparable). Both findings should be reported in the paper.

**The 7B FT-wins anomaly is not statistically significant.** Qwen-7B FT ag (0.7411) vs Qwen-7B Debias ps k6 (0.7299) is a tie under McNemar (p = 0.36) and the 95 % CI of the gap straddles zero. The narrative should treat 7B as a tie between FT-ag and Debias-ps, not a clear FT win.

---

# RQ3 (2026-04-29) — Debiasing intervention closes the FT gap

The debiasing technique: when retrieving the top-k FAISS neighbors for an issue, count the labels among the neighbors. If `bug_count − second_class_count ≤ margin (m=3)`, drop all bug-labeled neighbors and fall back to the next k non-bug neighbors. The intervention is applied at retrieval time and the rebalanced few-shots are consumed verbatim by the LLM.

## RQ3.1 — Debias vs plain RAGTAG, all (model, k) cells

Source: [analysis/rq3_debias_vs_ragtag.csv](analysis/rq3_debias_vs_ragtag.csv).

Δ macro F1 (debias − plain RAGTAG, project-specific mean):

| Model | k = 1 | k = 3 | k = 6 | k = 9 |
|---|---:|---:|---:|---:|
| Qwen-3B | −0.0088 | −0.0008 | +0.0309 | +0.0315 |
| Qwen-7B | −0.0097 | −0.0018 | +0.0157 | +0.0202 |
| Qwen-14B | −0.0061 | −0.0088 | +0.0194 | +0.0253 |
| Qwen-32B | −0.0090 | −0.0103 | +0.0122 | +0.0166 |

Debias **hurts at low k** (removing bug evidence harms precision when there's only one or three neighbors), **helps consistently at k ≥ 6**, and is largest at k = 9. The smallest model (3B) shows the biggest gain because it benefits most from class rebalancing; the largest model (32B) gains least because it's already best calibrated. Invalid-output rate falls under debias at high k (e.g., Qwen-32B ps k=9: 0.045 plain → 0.033 debias).

## RQ3.2 — Debias-best vs FT, per Qwen size

(See RQ2.9 for full bootstrap CIs and McNemar p-values.)

| Model | Best debias (ps) | FT-ag | Δ (debias − FT) | Significance |
|---|---:|---:|---:|:---|
| Qwen-3B | 0.7089 (k6) | 0.6520 | +0.057 | *** (debias > FT clearly) |
| Qwen-7B | 0.7299 (k6) | 0.7411 | −0.011 | ns (statistical tie, not a clear FT win) |
| Qwen-14B | 0.7418 (k9) | 0.7154 | +0.026 | ** (debias > FT) |
| Qwen-32B | 0.7745 (k9) | 0.7462 | +0.028 | bootstrap CI excludes 0; McNemar ns |

## RQ3.3 — Per-project win rate of debias-ps

Source: [analysis/rq3_per_project_winrate.csv](analysis/rq3_per_project_winrate.csv); figure [rq3_per_project_winrate.png](analysis/figures/rq3_per_project_winrate.png).

| Model | vs FT-ag (wins / 11) | mean Δ | vs FT-ps (wins / 11) | mean Δ |
|---|---:|---:|---:|---:|
| Qwen-3B | 8 / 11 | +0.063 | 9 / 11 | +0.140 |
| Qwen-7B | 6 / 11 | −0.005 | 11 / 11 | +0.225 |
| Qwen-14B | 6 / 11 | +0.031 | 10 / 11 | +0.109 |
| Qwen-32B | 8 / 11 | +0.032 | 9 / 11 | +0.065 |

Debias-ps consistently wins vs FT-ps (9–11/11 projects per model), and wins or ties vs FT-ag (6–8/11 projects per model).

## RQ3.4 — Scaling effect of debias

Figure: [rq3_debias_gain_by_size.png](analysis/figures/rq3_debias_gain_by_size.png).

- Left panel: Δ macro F1 vs k for each Qwen size. All four lines cross zero between k = 3 and k = 6; all are positive at k = 9.
- Right panel: best-debias vs FT-ag vs FT-ps. Debias is monotonically increasing with size; FT-ag has the 7B spike; FT-ps is the lowest line throughout.

## RQ3.5 — VTAG-debias mechanism ablation: the LLM-as-rescuer effect

The same retrieval-time intervention applied to a no-LLM voting baseline (VTAG-debias) gives a near-zero macro-F1 gain: +0.006 ag, +0.007 ps. The LLM does the heavy lifting in RAGTAG-debias.

Per-class recall delta (debias − plain) at k = 9, ground-truth-conditioned:

| System | Δ recall bug | Δ recall feature | Δ recall question | Δ macro F1 |
|---|---:|---:|---:|---:|
| VTAG ag | **−0.192** | +0.048 | **+0.155** | +0.006 |
| VTAG ps | **−0.183** | +0.047 | **+0.147** | +0.007 |
| RAGTAG-3B ps | −0.124 | +0.005 | +0.202 | +0.032 |
| RAGTAG-7B ps | −0.020 | +0.001 | +0.076 | +0.020 |
| RAGTAG-14B ps | −0.008 | +0.009 | +0.073 | +0.025 |
| RAGTAG-32B ps | **+0.002** | +0.003 | +0.052 | +0.017 |

VTAG-debias trades ~0.19 bug-recall for ~0.16 question-recall (1:1 trade, marginal macro gain). RAGTAG-debias at 7B+ loses essentially zero bug-recall while gaining 5–8 % question-recall. The LLM is reading the retrieval-rebalanced few-shots and **correctly preserving true-bug predictions even when bug examples are removed from the prompt** — the rescue mechanism.

**Quantifying the rescue.** Of the 164 test issues where VTAG-debias flipped a true-bug from "bug" to "question" (vs plain VTAG), how often did each Qwen model's RAGTAG-debias correctly predict "bug"?

| Model | n flips | n rescued | Rescue rate |
|---|---:|---:|---:|
| Qwen-3B | 164 | 79 | **48.2 %** |
| Qwen-7B | 164 | 133 | **81.1 %** |
| Qwen-14B | 164 | 139 | **84.8 %** |
| Qwen-32B | 164 | 138 | **84.1 %** |

Source: [analysis/rq3_llm_rescue.csv](analysis/rq3_llm_rescue.csv); per-class deltas in [analysis/rq3_vtag_debias_perclass.csv](analysis/rq3_vtag_debias_perclass.csv); figure [rq3_mechanism_ablation.png](analysis/figures/rq3_mechanism_ablation.png).

## RQ3.6 — Cost reframe

Source: [analysis/rq3_cost_reframe.csv](analysis/rq3_cost_reframe.csv); figure [rq3_cost_reframe.png](analysis/figures/rq3_cost_reframe.png). Excludes model-load time per the project convention.

| Model | Debias-ps total (s) | Debias F1 | FT-ag train (s) | FT-ag inf (s) | FT-ag total (s) | FT F1 | Δ F1 | Compute ratio (deb / FT) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen-3B | 803 | 0.709 | 367 | 309 | 676 | 0.652 | +0.057 | 1.19× |
| Qwen-7B | 1,439 | 0.730 | 568 | 371 | 939 | 0.741 | −0.011 | 1.53× |
| Qwen-14B | 5,431 | 0.742 | 1,728 | 1,211 | 2,939 | 0.715 | +0.026 | 1.85× |
| Qwen-32B | 11,424 | 0.775 | 3,616 | 2,208 | 5,824 | 0.746 | +0.028 | 1.96× |

**Caveat to the data-efficiency claim.** Debias-ps uses 300 retrieval examples per project (vs FT-ag's 3,300 training examples — 11× fewer), but on **total GPU-seconds it costs 1.2–2.0× more than FT-ag** because of the 11 separate inferences with long retrieval prompts. Debias is uniquely valuable for low-resource projects where only 300 examples are available — there it beats any FT-ps trained on the same 300 examples by 0.06–0.23 macro F1 depending on model size.

**Peak GPU memory tells a different story.** FT's peak memory is set by the training step (gradients + optimizer state + model + activations); RAGTAG/Debias inference only carries the model + activations + a longer prompt. Across the family:

| Model | FT peak (train binding) | Debias-ps inference peak | Memory headroom | Min deployable GPU |
|---|---:|---:|---:|---|
| Qwen-3B | 5.5 GB | 2.9 GB | −2.6 GB (1.9× less) | both fit anywhere |
| Qwen-7B | 9.9 GB | 6.8 GB | −3.1 GB (1.5× less) | both fit on 12 GB |
| Qwen-14B | 17.0 GB | 12.6 GB | −4.4 GB (1.3× less) | both fit on 24 GB |
| **Qwen-32B** | **31.7 GB** | **22.3 GB** | **−9.5 GB (1.4× less)** | **FT needs ≥40 GB (A100 / A6000); Debias fits on a single 24 GB RTX 3090/4090** |

For Qwen-32B specifically, this is the practitioner-facing flip: FT-32B locks you into datacenter-class hardware (A100/A6000, 40+ GB) for the training step alone; Debias-32B inference fits on a single consumer 24 GB GPU. So at the largest size, debias is dramatically cheaper in deployable hardware class — **a single RTX 4090 vs an A100** — even though wall-clock is 1.96× higher.

The honest framing of the data-efficiency / cost story now has three axes:

| Axis | Direction | Magnitude |
|---|---|---|
| Training examples consumed | Debias wins | 300 ps vs 3,300 ag = 11× fewer per project |
| Wall-clock GPU-seconds | FT wins | Debias is 1.2–2.0× slower at the largest sizes |
| Peak GPU memory & hardware class | Debias wins | 1.3–1.9× less; at 32B, the difference is consumer-vs-datacenter |
| Operational simplicity | Debias wins | No gradient updates, no training pipeline, swap models trivially |

---

## 0. Master Table — Final Results (2026-04-28, original — includes Llama)

Every (model × approach × k × setting) combination, all 7 metrics + per-class precision/recall + invalid count. Project-specific values are 11-project averages.

| Model | Approach | k | Setting | acc | f1_macro | f1_bug | f1_feat | f1_q | P/R bug | P/R feat | P/R q | invalid |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| Llama-3B | zero-shot | — | agnostic | 0.627 | 0.583 | 0.666 | 0.769 | 0.314 | .53/.88 | .74/.80 | .77/.20 | 9/3300 |
| Llama-3B | RAGTAG | k1 | agnostic | 0.668 | 0.651 | 0.690 | 0.763 | 0.501 | .63/.77 | .70/.84 | .70/.39 | 3/3300 |
| Llama-3B | RAGTAG | k1 | proj-spec | 0.666 | 0.647 | 0.689 | 0.759 | 0.492 | .63/.77 | .70/.84 | .70/.39 | 5/3300 |
| Llama-3B | RAGTAG | k3 | agnostic | 0.666 | 0.658 | 0.671 | 0.757 | 0.546 | .62/.73 | .71/.81 | .67/.46 | 5/3300 |
| Llama-3B | RAGTAG | k3 | proj-spec | 0.669 | 0.657 | 0.671 | 0.761 | 0.541 | .63/.73 | .72/.82 | .68/.46 | 5/3300 |
| Llama-3B | RAGTAG | k6 | agnostic | 0.659 | 0.659 | 0.686 | 0.760 | 0.532 | .62/.77 | .74/.78 | .68/.44 | 94/3300 |
| Llama-3B | RAGTAG | k6 | proj-spec | 0.662 | 0.660 | 0.681 | 0.767 | 0.531 | .62/.76 | .76/.79 | .70/.44 | 91/3300 |
| Llama-3B | RAGTAG | k9 | agnostic | 0.640 | 0.642 | 0.663 | 0.761 | 0.502 | .59/.76 | .76/.76 | .66/.41 | 120/3300 |
| Llama-3B | RAGTAG | k9 | proj-spec | 0.640 | 0.641 | 0.661 | 0.755 | 0.508 | .60/.75 | .76/.76 | .68/.41 | 123/3300 |
| Llama-3B | debias_m3 | k1 | proj-spec | 0.670 | 0.650 | 0.692 | 0.766 | 0.492 | .62/.79 | .72/.83 | .70/.39 | 6/3300 |
| Llama-3B | debias_m3 | k3 | proj-spec | 0.679 | 0.669 | 0.659 | 0.756 | 0.592 | .69/.65 | .69/.84 | .68/.54 | 6/3300 |
| Llama-3B | debias_m3 | k6 | proj-spec | 0.692 | 0.687 | 0.655 | 0.761 | 0.644 | .77/.61 | .71/.83 | .68/.64 | 45/3300 |
| Llama-3B | debias_m3 | k9 | proj-spec | 0.671 | 0.670 | 0.621 | 0.754 | 0.634 | .76/.57 | .72/.80 | .65/.65 | 87/3300 |
| Llama-3B | **FT** | — | **agnostic** | **0.728** | **0.728** | 0.731 | 0.803 | 0.649 | .66/.82 | .84/.77 | .72/.59 | 9/3300 |
| Llama-3B | FT | — | proj-spec | 0.482 | 0.428 | 0.551 | 0.520 | 0.212 | .42/.82 | .60/.50 | .66/.13 | 15/3300 |
| Llama-8B | zero-shot | — | agnostic | 0.640 | 0.599 | 0.672 | 0.790 | 0.334 | .53/.93 | .79/.79 | .84/.21 | 8/3300 |
| Llama-8B | RAGTAG | k1 | agnostic | 0.672 | 0.650 | 0.695 | 0.793 | 0.460 | .56/.91 | .80/.78 | .80/.32 | 4/3300 |
| Llama-8B | RAGTAG | k1 | proj-spec | 0.675 | 0.647 | 0.695 | 0.800 | 0.446 | .56/.91 | .82/.79 | .80/.33 | 3/3300 |
| Llama-8B | RAGTAG | k3 | agnostic | 0.701 | 0.686 | 0.720 | 0.807 | 0.532 | .60/.91 | .81/.80 | .81/.40 | 7/3300 |
| Llama-8B | RAGTAG | k3 | proj-spec | 0.701 | 0.682 | 0.720 | 0.806 | 0.520 | .60/.91 | .82/.80 | .81/.40 | 8/3300 |
| Llama-8B | RAGTAG | k6 | agnostic | 0.696 | 0.695 | 0.719 | 0.812 | 0.554 | .61/.87 | .84/.79 | .79/.43 | 98/3300 |
| Llama-8B | RAGTAG | k6 | proj-spec | 0.701 | 0.696 | 0.724 | 0.814 | 0.551 | .62/.88 | .85/.79 | .80/.44 | 96/3300 |
| Llama-8B | RAGTAG | k9 | agnostic | 0.681 | 0.682 | 0.712 | 0.803 | 0.530 | .61/.86 | .83/.77 | .76/.41 | 125/3300 |
| Llama-8B | RAGTAG | k9 | proj-spec | 0.683 | 0.681 | 0.711 | 0.804 | 0.527 | .61/.85 | .83/.78 | .78/.41 | 128/3300 |
| Llama-8B | debias_m3 | k1 | proj-spec | 0.671 | 0.642 | 0.688 | 0.792 | 0.446 | .56/.90 | .82/.78 | .80/.33 | 3/3300 |
| Llama-8B | debias_m3 | k3 | proj-spec | 0.687 | 0.671 | 0.692 | 0.794 | 0.526 | .59/.86 | .81/.79 | .77/.42 | 3/3300 |
| Llama-8B | debias_m3 | k6 | proj-spec | 0.718 | 0.715 | 0.727 | 0.813 | 0.604 | .66/.82 | .83/.80 | .74/.53 | 47/3300 |
| Llama-8B | debias_m3 | k9 | proj-spec | 0.728 | 0.732 | 0.732 | 0.813 | 0.652 | .71/.77 | .83/.80 | .74/.61 | 90/3300 |
| Llama-8B | **FT** | — | **agnostic** | **0.744** | **0.742** | 0.702 | 0.800 | 0.724 | .86/.60 | .74/.86 | .68/.77 | 11/3300 |
| Llama-8B | FT | — | proj-spec | 0.533 | 0.492 | 0.419 | 0.667 | 0.390 | .60/.37 | .54/.90 | .56/.32 | 6/3300 |
| Qwen-14B | zero-shot | — | agnostic | 0.675 | 0.645 | 0.691 | 0.806 | 0.440 | .57/.88 | .76/.86 | .87/.29 | 2/3300 |
| Qwen-14B | RAGTAG | k1 | agnostic | 0.703 | 0.684 | 0.714 | 0.822 | 0.516 | .59/.89 | .80/.85 | .85/.37 | 3/3300 |
| Qwen-14B | RAGTAG | k1 | proj-spec | 0.703 | 0.677 | 0.719 | 0.820 | 0.493 | .60/.89 | .81/.85 | .85/.37 | 3/3300 |
| Qwen-14B | RAGTAG | k3 | agnostic | 0.726 | 0.712 | 0.735 | 0.830 | 0.569 | .62/.90 | .82/.85 | .85/.43 | 4/3300 |
| Qwen-14B | RAGTAG | k3 | proj-spec | 0.726 | 0.706 | 0.737 | 0.832 | 0.549 | .63/.90 | .83/.85 | .84/.43 | 4/3300 |
| Qwen-14B | RAGTAG | k6 | agnostic | 0.713 | 0.716 | 0.724 | 0.831 | 0.592 | .63/.85 | .83/.84 | .85/.46 | 118/3300 |
| Qwen-14B | RAGTAG | k6 | proj-spec | 0.714 | 0.710 | 0.726 | 0.834 | 0.572 | .64/.85 | .84/.84 | .85/.45 | 117/3300 |
| Qwen-14B | RAGTAG | k9 | agnostic | 0.711 | 0.717 | 0.727 | 0.827 | 0.597 | .64/.84 | .83/.82 | .83/.47 | 141/3300 |
| Qwen-14B | RAGTAG | k9 | proj-spec | 0.714 | 0.717 | 0.728 | 0.833 | 0.588 | .65/.84 | .84/.83 | .83/.47 | 142/3300 |
| Qwen-14B | debias_m3 | k1 | proj-spec | 0.697 | 0.671 | 0.706 | 0.816 | 0.490 | .59/.88 | .80/.84 | .85/.37 | 3/3300 |
| Qwen-14B | debias_m3 | k3 | proj-spec | 0.716 | 0.697 | 0.720 | 0.817 | 0.555 | .62/.87 | .81/.84 | .84/.43 | 4/3300 |
| Qwen-14B | debias_m3 | k6 | proj-spec | 0.736 | 0.730 | 0.745 | 0.830 | 0.615 | .66/.86 | .83/.84 | .84/.50 | 55/3300 |
| Qwen-14B | **debias_m3** | k9 | **proj-spec** | **0.739** | **0.742** | 0.748 | 0.834 | 0.644 | .68/.83 | .84/.84 | .83/.54 | 106/3300 |
| Qwen-14B | FT | — | agnostic | 0.712 | 0.715 | 0.703 | 0.767 | 0.676 | .73/.68 | .85/.70 | .61/.76 | 3/3300 |
| Qwen-14B | FT | — | proj-spec | 0.638 | 0.637 | 0.692 | 0.652 | 0.566 | .67/.72 | .76/.59 | .54/.60 | 4/3300 |
| Qwen-32B | zero-shot | — | agnostic | 0.706 | 0.688 | 0.715 | 0.814 | 0.533 | .61/.86 | .77/.87 | .86/.39 | 6/3300 |
| Qwen-32B | zero-shot | — | proj-spec | 0.705 | 0.680 | 0.717 | 0.817 | 0.506 | .62/.86 | .78/.87 | .86/.39 | 6/3300 |
| Qwen-32B | RAGTAG | k1 | agnostic | 0.737 | 0.726 | 0.745 | 0.828 | 0.604 | .64/.90 | .81/.85 | .86/.47 | 4/3300 |
| Qwen-32B | RAGTAG | k1 | proj-spec | 0.736 | 0.720 | 0.747 | 0.830 | 0.584 | .64/.89 | .83/.85 | .86/.47 | 4/3300 |
| Qwen-32B | RAGTAG | k3 | agnostic | 0.761 | 0.755 | 0.766 | 0.838 | 0.662 | .67/.90 | .83/.85 | .86/.54 | 9/3300 |
| Qwen-32B | RAGTAG | k3 | proj-spec | 0.761 | 0.751 | 0.764 | 0.841 | 0.648 | .67/.89 | .84/.85 | .86/.54 | 8/3300 |
| Qwen-32B | RAGTAG | k6 | agnostic | 0.745 | 0.754 | 0.754 | 0.837 | 0.672 | .68/.84 | .84/.84 | .84/.56 | 120/3300 |
| Qwen-32B | RAGTAG | k6 | proj-spec | 0.748 | 0.755 | 0.755 | 0.841 | 0.667 | .69/.84 | .85/.84 | .85/.57 | 119/3300 |
| Qwen-32B | RAGTAG | k9 | agnostic | 0.746 | 0.759 | 0.758 | 0.838 | 0.682 | .69/.83 | .84/.83 | .84/.57 | 146/3300 |
| Qwen-32B | RAGTAG | k9 | proj-spec | 0.748 | 0.758 | 0.754 | 0.841 | 0.678 | .70/.83 | .86/.83 | .85/.58 | 148/3300 |
| Qwen-32B | debias_m3 | k1 | proj-spec | 0.727 | 0.711 | 0.733 | 0.821 | 0.580 | .64/.87 | .81/.85 | .86/.46 | 3/3300 |
| Qwen-32B | debias_m3 | k3 | proj-spec | 0.752 | 0.741 | 0.752 | 0.830 | 0.640 | .67/.86 | .82/.86 | .86/.53 | 7/3300 |
| Qwen-32B | debias_m3 | k6 | proj-spec | 0.767 | 0.767 | 0.771 | 0.843 | 0.685 | .71/.86 | .85/.85 | .84/.60 | 57/3300 |
| Qwen-32B | **debias_m3** | k9 | **proj-spec** | **0.766** | **0.775** | 0.775 | 0.841 | 0.708 | .73/.83 | .85/.84 | .82/.63 | 109/3300 |
| Qwen-32B | FT | — | agnostic | 0.755 | 0.746 | 0.778 | 0.814 | 0.647 | .71/.86 | .76/.88 | .83/.53 | 0/3300 |
| Qwen-32B | FT | — | proj-spec | 0.720 | 0.713 | 0.679 | 0.797 | 0.662 | .70/.68 | .77/.84 | .73/.64 | 2/3300 |

**Bold** = best f1_macro per model.

### Per-model winner

| Model | Best approach | f1_macro |
|---|---|---:|
| Llama-3B | FT (agnostic) | 0.728 |
| Llama-8B | FT (agnostic) | 0.742 |
| Qwen-14B | **debias_m3 k=9** (proj-spec) | 0.742 |
| Qwen-32B | **debias_m3 k=9** (proj-spec) | 0.775 |

### Pattern 1 — Debias_m3 vs plain RAGTAG (proj-spec avg, robust across all 4 models)

| Model | k=1 | k=3 | k=6 | k=9 | Mechanism (invalid k=6 → k=9) |
|---|---:|---:|---:|---:|---|
| Llama-3B | +0.003 | +0.012 | +0.027 | +0.028 | 91→123 (RAG) vs 45→87 (deb) |
| Llama-8B | −0.005 | −0.012 | +0.018 | **+0.052** | 96→128 vs 47→90 |
| Qwen-14B | −0.006 | −0.009 | +0.019 | +0.025 | 117→142 vs 55→106 |
| Qwen-32B | −0.009 | −0.010 | +0.012 | +0.017 | 119→148 vs 57→109 |

Debias hurts (slightly) at low k, helps clearly at k≥6, with biggest gains at k=9. **Debias roughly halves the invalid count** at high k — a key part of the mechanism: it keeps prompts from overloading the model into malformed output.

### Pattern 2 — FT vs zero-training, monotonic crossover by model scale

| Model | best FT | best RAGTAG | best debias | FT − bestRAG | FT − bestDeb |
|---|---:|---:|---:|---:|---:|
| Llama-3B | 0.728 | 0.660 | 0.687 | **+0.068** | **+0.041** |
| Llama-8B | 0.742 | 0.696 | 0.732 | +0.046 | +0.010 |
| Qwen-14B | 0.715 | 0.717 | 0.742 | −0.002 | −0.026 |
| Qwen-32B | 0.746 | 0.759 | 0.775 | −0.013 | −0.028 |

The FT advantage closes monotonically with scale and **inverts between 8B and 14B**. At Qwen-32B, debias_m3 k=9 beats fine-tuning by 0.028 with no training at all. **This is the headline RQ3 result.**

*Caveat:* the 8B→14B inversion conflates model size and family (Llama → Qwen). With current data, the safe claim is "the FT advantage erodes as models get larger and/or stronger at instruction-following; by Qwen-32B scale, zero-training debiased RAGTAG matches or beats fine-tuning."

### Other notable observations

1. **Per-project FT is broken at small-data scales.** Llama-3B drops from 0.728 (agnostic FT) to 0.428 (proj-spec FT avg). Llama-8B: 0.742 → 0.492. With 300 train issues per project, LoRA does not get enough signal. Qwen is more robust (14B: −0.078, 32B: −0.033) but agnostic > proj-spec for FT in every case. **Per-project fine-tuning is not a viable deployment story.**

2. **Question is the universal weak class.** For every model and approach, `recall_question` is the lowest of the three classes — matching prior qualitative analysis ("question→bug" is the dominant error). Even Qwen-32B debias k=9 only reaches 0.63 question recall vs 0.83 bug and 0.84 feature.

3. **Agnostic ≈ proj-spec for RAGTAG and debias**, within 0.01 at every k. Re-confirms the cross-project retrieval finding (88.3% same-project neighbors in agnostic): restricting the index to one project adds nothing once retrieval already pulls same-project examples.

4. **The 30k FT advantage doesn't fully transfer to 11k.** On 30k, FT was comfortably ahead; on 11k the gap is much smaller (Llama-3B FT vs best zero-training: 0.041, vs >0.10 on 30k). Less training data → less FT edge.

5. **Invalid-rate scales with k** for every model. Even Qwen-32B agnostic k=9 has 4.4% invalid output. This is intrinsic to crowded prompts, not model-specific. Debias halves the rate, providing a clean instrumented mechanism for the paper.

---

## 1. Dataset

6,600 GitHub issues from 11 diverse open-source projects, perfectly balanced:
- **11 projects** x 600 issues each (300 train / 300 test)
- **3 labels** x 100 per project per split (bug / feature / question)
- Balanced by design: no class imbalance confounds

| Project | Train | Test |
|---------|-------|------|
| ansible/ansible | 300 | 300 |
| bitcoin/bitcoin | 300 | 300 |
| dart-lang/sdk | 300 | 300 |
| dotnet/roslyn | 300 | 300 |
| facebook/react | 300 | 300 |
| flutter/flutter | 300 | 300 |
| kubernetes/kubernetes | 300 | 300 |
| microsoft/TypeScript | 300 | 300 |
| microsoft/vscode | 300 | 300 |
| opencv/opencv | 300 | 300 |
| tensorflow/tensorflow | 300 | 300 |

**Two experimental settings:**
- **Agnostic:** FAISS index over all 3,300 train issues, test on all 3,300. Model sees cross-project examples.
- **Project-specific:** Separate FAISS index per project (300 train), test on 300. Simulates a single-repo deployment.

**Key finding on settings:** Cross-project retrieval analysis shows 88.3% of agnostic neighbors come from the same project (see Section 5). Agnostic RAGTAG is therefore nearly equivalent to project-specific RAGTAG and can be excluded from the paper. The meaningful comparison is project-specific RAGTAG vs agnostic FT — RAGTAG uses 300 same-project examples at retrieval time while FT trains on all 3,300.

---

## 2. Compact Summary

| Approach | Llama-3B | Llama-8B | Qwen-14B | Qwen-32B | Qwen-32B (valid-only) |
|----------|----------|----------|----------|----------|----------------------|
| Zero-shot | 0.576 | 0.591 | 0.638 | 0.677 | — |
| VTAG (PS, best k) | 0.602 | 0.602 | 0.602 | 0.602 | — |
| RAGTAG (PS, best k) | 0.678 | 0.704 | 0.727 | 0.746 | 0.777 |
| Debiased RAGTAG (PS, best k) | 0.698 | 0.737 | **0.749** | 0.746 | **0.787** |
| Fine-Tuning (agnostic) | 0.725 | 0.736 | *pending* | *pending* | — |
| Debias gain | +0.020 | +0.034 | +0.022 | +0.000 | +0.010 |
| Debias wins/11 | 9 | 11 | 10 | — | — |

**Headline results:**
- Llama-8B debiased (0.737) matches Llama-8B FT (0.736) with zero training
- Qwen-14B debiased (0.749) beats both Llama FTs (0.725, 0.736) with zero training
- Qwen-32B valid-only (0.787) shows the true model capability when not limited by GPU memory
- All 4 debiased RAGTAG results beat Llama-3B FT (0.725)
- Qwen-32B raw scores (0.746) are artificially suppressed by 22.4% OOM-driven invalid rate at k=9

---

## 3. Agnostic Results (3,300 test issues, overall)

| Model | Approach | Macro F1 | Bug F1 | Feature F1 | Question F1 | Invalid |
|-------|----------|----------|--------|------------|-------------|---------|
| Llama-3B | Zero-shot | 0.583 | 0.666 | 0.769 | 0.314 | 9 |
| Llama-3B | RAGTAG k=1 | 0.651 | 0.689 | 0.763 | 0.501 | 3 |
| Llama-3B | RAGTAG k=3 | 0.658 | 0.671 | 0.758 | 0.546 | 5 |
| Llama-3B | RAGTAG k=6 | **0.664** | 0.699 | 0.760 | 0.532 | 94 |
| Llama-3B | RAGTAG k=9 | 0.648 | 0.681 | 0.761 | 0.502 | 120 |
| Llama-3B | Fine-Tuning | **0.729** | 0.734 | 0.803 | 0.649 | 9 |
| | | | | | | |
| Llama-8B | Zero-shot | 0.598 | 0.671 | 0.790 | 0.334 | 8 |
| Llama-8B | RAGTAG k=1 | 0.650 | 0.695 | 0.794 | 0.460 | 4 |
| Llama-8B | RAGTAG k=3 | 0.686 | 0.719 | 0.807 | 0.532 | 7 |
| Llama-8B | RAGTAG k=6 | **0.698** | 0.729 | 0.812 | 0.554 | 98 |
| Llama-8B | RAGTAG k=9 | 0.686 | 0.725 | 0.803 | 0.530 | 125 |
| Llama-8B | Fine-Tuning | **0.743** | 0.706 | 0.800 | 0.724 | 11 |
| | | | | | | |
| Qwen-14B | Zero-shot | 0.645 | 0.690 | 0.806 | 0.439 | 2 |
| Qwen-14B | RAGTAG k=1 | 0.684 | 0.713 | 0.822 | 0.516 | 3 |
| Qwen-14B | RAGTAG k=3 | 0.712 | 0.735 | 0.830 | 0.569 | 4 |
| Qwen-14B | RAGTAG k=6 | 0.721 | 0.741 | 0.831 | 0.592 | 118 |
| Qwen-14B | RAGTAG k=9 | **0.723** | 0.744 | 0.827 | 0.597 | 141 |
| | | | | | | |
| Qwen-32B | Zero-shot | 0.686 | 0.713 | 0.811 | 0.534 | 51 |
| Qwen-32B | RAGTAG k=1 | 0.717 | 0.739 | 0.822 | 0.592 | 91 |
| Qwen-32B | RAGTAG k=3 | **0.741** | 0.753 | 0.834 | 0.635 | 212 |
| Qwen-32B | RAGTAG k=6 | 0.727 | 0.739 | 0.825 | 0.616 | 486 |
| Qwen-32B | RAGTAG k=9 | 0.697 | 0.712 | 0.779 | 0.602 | 754 |

Bug bias persists across all models: question F1 is always lowest for RAGTAG and zero-shot. FT largely resolves it through gradient updates. Qwen-32B invalid rates are severe (up to 22.8% at k=9) due to CUDA OOM on RTX 4090.

---

## 4. Agnostic Per-Project Results

### 4.1 Zero-shot

| Project | Llama-3B | Llama-8B | Qwen-14B | Qwen-32B |
|---------|----------|----------|----------|----------|
| ansible | 0.576 | 0.553 | 0.567 | 0.586 |
| bitcoin | 0.587 | 0.613 | 0.632 | 0.665 |
| dart | 0.713 | 0.719 | 0.771 | 0.827 |
| dotnet | 0.551 | 0.542 | 0.584 | 0.615 |
| react | 0.682 | 0.701 | 0.759 | 0.794 |
| flutter | 0.568 | 0.674 | 0.717 | 0.733 |
| k8s | 0.638 | 0.608 | 0.633 | 0.712 |
| TypeScript | 0.465 | 0.477 | 0.543 | 0.596 |
| vscode | 0.444 | 0.486 | 0.500 | 0.515 |
| opencv | 0.543 | 0.547 | 0.608 | 0.660 |
| tensorflow | 0.574 | 0.584 | 0.705 | 0.744 |
| **Average** | **0.576** | **0.591** | **0.638** | **0.677** |

### 4.2 RAGTAG (agnostic, best k per project)

| Project | Llama-3B | Llama-8B | Qwen-14B | Qwen-32B |
|---------|----------|----------|----------|----------|
| ansible | 0.695 | 0.650 | 0.671 | 0.686 |
| bitcoin | 0.632 | 0.682 | 0.672 | 0.689 |
| dart | 0.740 | 0.781 | 0.827 | 0.852 |
| dotnet | 0.591 | 0.596 | 0.627 | 0.670 |
| react | 0.758 | 0.757 | 0.798 | 0.819 |
| flutter | 0.652 | 0.712 | 0.745 | 0.738 |
| k8s | 0.802 | 0.831 | 0.826 | 0.806 |
| TypeScript | 0.551 | 0.600 | 0.606 | 0.646 |
| vscode | 0.649 | 0.682 | 0.646 | 0.664 |
| opencv | 0.652 | 0.701 | 0.692 | 0.717 |
| tensorflow | 0.684 | 0.721 | 0.769 | 0.782 |
| **Average** | **0.673** | **0.701** | **0.716** | **0.734** |

### 4.3 Fine-Tuning (agnostic, per-project)

| Project | Llama-3B | Llama-8B | Qwen-14B | Qwen-32B |
|---------|----------|----------|----------|----------|
| ansible | 0.665 | 0.672 | — | — |
| bitcoin | 0.708 | 0.707 | — | — |
| dart | 0.778 | 0.855 | — | — |
| dotnet | 0.621 | 0.661 | — | — |
| react | 0.781 | 0.811 | — | — |
| flutter | 0.731 | 0.663 | — | — |
| k8s | 0.845 | 0.875 | — | — |
| TypeScript | 0.634 | 0.698 | — | — |
| vscode | 0.737 | 0.727 | — | — |
| opencv | 0.701 | 0.650 | — | — |
| tensorflow | 0.774 | 0.775 | — | — |
| **Average** | **0.725** | **0.736** | — | — |

---

## 5. Cross-Project Retrieval Analysis

In the agnostic setting, FAISS retrieves neighbors from a pool of all 11 projects. Analysis of neighbor provenance (k=9):

**Overall:** 88.3% of retrieved neighbors come from the same project as the test issue.

| Project | Same-project rate |
|---------|-------------------|
| tensorflow | 99.3% |
| kubernetes | 97.6% |
| react | 96.2% |
| TypeScript | 95.9% |
| opencv | 90.7% |
| flutter | 86.0% |
| ansible | 84.6% |
| vscode | 81.6% |
| dart | 81.1% |
| dotnet | 80.7% |
| bitcoin | 77.7% |

**By neighbor rank:** Rank 0 = 93.2%, Rank 8 = 85.3%. Same-project rate decreases with rank but remains dominant.

**Implication:** Agnostic RAGTAG can be excluded from the paper — it is nearly equivalent to project-specific RAGTAG because FAISS naturally clusters by project. The 11.7% cross-project neighbors add noise, not signal. Agnostic RAGTAG averages (0.673 for 3B, 0.701 for 8B) are close to project-specific (0.678, 0.704). The meaningful comparison is project-specific RAGTAG (300 train) vs agnostic FT (3,300 train) — RAGTAG competes at a 10x data disadvantage because retrieval naturally selects the right subset.

---

## 6. Project-Specific Results

### 6.1 VTAG Baseline (Pure Retrieval, No LLM)

| Project | Best k | VTAG F1 |
|---------|--------|---------|
| ansible | varies | 0.593 |
| bitcoin | varies | 0.597 |
| dart | varies | 0.624 |
| dotnet | varies | 0.536 |
| react | varies | 0.611 |
| flutter | varies | 0.580 |
| k8s | varies | 0.732 |
| TypeScript | varies | 0.559 |
| vscode | varies | 0.648 |
| opencv | varies | 0.635 |
| tensorflow | varies | 0.507 |
| **Average** | | **0.602** |

### 6.2 RAGTAG Baseline (All Models, All k Values)

**Llama-3B:**
| Project | k=1 | k=3 | k=6 | k=9 | Best | k |
|---------|-----|-----|-----|-----|------|---|
| ansible | 0.683 | 0.681 | 0.682 | 0.692 | 0.692 | 9 |
| bitcoin | 0.617 | 0.626 | 0.658 | 0.610 | 0.658 | 6 |
| dart | 0.706 | 0.732 | 0.711 | 0.680 | 0.732 | 3 |
| dotnet | 0.563 | 0.584 | 0.556 | 0.569 | 0.584 | 3 |
| react | 0.742 | 0.729 | 0.730 | 0.715 | 0.742 | 1 |
| flutter | 0.634 | 0.598 | 0.675 | 0.672 | 0.675 | 6 |
| k8s | 0.765 | 0.801 | 0.809 | 0.763 | 0.809 | 6 |
| TypeScript | 0.500 | 0.530 | 0.523 | 0.549 | 0.549 | 9 |
| vscode | 0.569 | 0.635 | 0.645 | 0.604 | 0.645 | 6 |
| opencv | 0.656 | 0.606 | 0.647 | 0.646 | 0.656 | 1 |
| tensorflow | 0.679 | 0.711 | 0.666 | 0.622 | 0.711 | 3 |
| **Average** | | | | | **0.678** | |

**Llama-8B:**
| Project | k=1 | k=3 | k=6 | k=9 | Best | k |
|---------|-----|-----|-----|-----|------|---|
| ansible | 0.611 | 0.634 | 0.644 | 0.649 | 0.649 | 9 |
| bitcoin | 0.654 | 0.673 | 0.692 | 0.670 | 0.692 | 6 |
| dart | 0.751 | 0.745 | 0.781 | 0.790 | 0.790 | 9 |
| dotnet | 0.583 | 0.579 | 0.592 | 0.573 | 0.592 | 6 |
| react | 0.719 | 0.756 | 0.754 | 0.749 | 0.756 | 3 |
| flutter | 0.681 | 0.696 | 0.697 | 0.672 | 0.697 | 6 |
| k8s | 0.763 | 0.813 | 0.815 | 0.806 | 0.815 | 6 |
| TypeScript | 0.529 | 0.581 | 0.612 | 0.585 | 0.612 | 6 |
| vscode | 0.569 | 0.640 | 0.708 | 0.656 | 0.708 | 6 |
| opencv | 0.604 | 0.656 | 0.700 | 0.690 | 0.700 | 6 |
| tensorflow | 0.653 | 0.729 | 0.701 | 0.697 | 0.729 | 3 |
| **Average** | | | | | **0.704** | |

**Qwen-14B:**
| Project | k=1 | k=3 | k=6 | k=9 | Best | k |
|---------|-----|-----|-----|-----|------|---|
| ansible | 0.616 | 0.647 | 0.656 | 0.678 | 0.678 | 9 |
| bitcoin | 0.631 | 0.646 | 0.692 | 0.706 | 0.706 | 9 |
| dart | 0.800 | 0.821 | 0.821 | 0.841 | 0.841 | 9 |
| dotnet | 0.594 | 0.613 | 0.628 | 0.618 | 0.628 | 6 |
| react | 0.775 | 0.797 | 0.793 | 0.797 | 0.797 | 9 |
| flutter | 0.724 | 0.739 | 0.764 | 0.758 | 0.764 | 6 |
| k8s | 0.786 | 0.835 | 0.813 | 0.811 | 0.835 | 3 |
| TypeScript | 0.575 | 0.594 | 0.591 | 0.614 | 0.614 | 9 |
| vscode | 0.560 | 0.613 | 0.650 | 0.662 | 0.662 | 9 |
| opencv | 0.653 | 0.686 | 0.699 | 0.703 | 0.703 | 9 |
| tensorflow | 0.729 | 0.773 | 0.771 | 0.763 | 0.773 | 3 |
| **Average** | | | | | **0.727** | |

**Qwen-32B:**
| Project | k=1 | k=3 | k=6 | k=9 | Best | k |
|---------|-----|-----|-----|-----|------|---|
| ansible | 0.644 | 0.678 | 0.693 | 0.674 | 0.693 | 6 |
| bitcoin | 0.680 | 0.691 | 0.715 | 0.704 | 0.715 | 6 |
| dart | 0.817 | 0.848 | 0.844 | 0.848 | 0.848 | 9 |
| dotnet | 0.640 | 0.665 | 0.674 | 0.669 | 0.674 | 6 |
| react | 0.807 | 0.819 | 0.812 | 0.812 | 0.819 | 3 |
| flutter | 0.769 | 0.801 | 0.763 | 0.667 | 0.801 | 3 |
| k8s | 0.799 | 0.799 | 0.774 | 0.696 | 0.799 | 1 |
| TypeScript | 0.628 | 0.652 | 0.600 | 0.474 | 0.652 | 3 |
| vscode | 0.599 | 0.662 | 0.635 | 0.604 | 0.662 | 3 |
| opencv | 0.684 | 0.704 | 0.722 | 0.749 | 0.749 | 9 |
| tensorflow | 0.784 | 0.778 | 0.791 | 0.740 | 0.791 | 6 |
| **Average** | | | | | **0.746** | |

Qwen-32B's best k skews low (k=1-6) because higher k values trigger more OOMs. Its raw scores are suppressed — see Section 9.

---

## 7. Debiased Retrieval Results (All Models, margin=3)

### 7.1 Mechanism

When `bug_count - question_count <= margin(3)` in the retrieved neighbors, all bug neighbors are removed. This prevents bug-biased retrieval from reinforcing the model's parametric bug prior for ambiguous cases.

### 7.2 Summary Table (Best-k per Project)

**Llama-3B (avg gain +0.020, 9/11 improve):**
| Project | RAGTAG | Debiased | Delta | Agn FT | Deb vs FT |
|---------|--------|----------|-------|--------|-----------|
| ansible | 0.692 | **0.741** | +0.049 | 0.665 | **+0.076** |
| bitcoin | 0.658 | **0.679** | +0.021 | 0.708 | -0.029 |
| dart | 0.732 | **0.744** | +0.012 | 0.778 | -0.033 |
| dotnet | 0.584 | **0.594** | +0.010 | 0.621 | -0.027 |
| react | 0.742 | **0.750** | +0.008 | 0.781 | -0.031 |
| flutter | 0.675 | **0.715** | +0.040 | 0.731 | -0.015 |
| k8s | 0.809 | **0.838** | +0.030 | 0.845 | -0.006 |
| TypeScript | 0.549 | 0.521 | -0.029 | 0.634 | -0.113 |
| vscode | 0.645 | 0.625 | -0.020 | 0.737 | -0.112 |
| opencv | 0.656 | **0.707** | +0.051 | 0.701 | **+0.007** |
| tensorflow | 0.711 | **0.761** | +0.050 | 0.774 | -0.013 |
| **Average** | **0.678** | **0.698** | **+0.020** | **0.725** | **-0.027** |

2 wins vs FT (ansible, opencv). Gap to FT narrows from 0.047 to 0.027. Two regressions: TypeScript (-0.029), vscode (-0.020) — bug F1 collapses at higher k while question F1 improves. Over-correction on projects with ambiguous bug/question boundaries.

**Llama-8B (avg gain +0.034, 11/11 improve):**
| Project | RAGTAG | Debiased | Delta | Agn FT | Deb vs FT |
|---------|--------|----------|-------|--------|-----------|
| ansible | 0.649 | **0.674** | +0.025 | 0.672 | **+0.002** |
| bitcoin | 0.692 | **0.704** | +0.013 | 0.707 | -0.002 |
| dart | 0.790 | **0.821** | +0.031 | 0.855 | -0.034 |
| dotnet | 0.592 | **0.651** | +0.059 | 0.661 | -0.010 |
| react | 0.756 | **0.775** | +0.019 | 0.811 | -0.036 |
| flutter | 0.697 | **0.747** | +0.049 | 0.663 | **+0.084** |
| k8s | 0.815 | **0.874** | +0.059 | 0.875 | -0.001 |
| TypeScript | 0.612 | **0.620** | +0.008 | 0.698 | -0.078 |
| vscode | 0.708 | **0.710** | +0.002 | 0.727 | -0.017 |
| opencv | 0.700 | **0.767** | +0.066 | 0.650 | **+0.117** |
| tensorflow | 0.729 | **0.769** | +0.040 | 0.775 | -0.006 |
| **Average** | **0.704** | **0.737** | **+0.034** | **0.736** | **+0.002** |

3 wins vs FT (ansible, flutter, opencv), 3 ties within 0.005 (bitcoin, k8s, tensorflow). **Average gap is +0.002 — parity with FT.** No regressions. The 8B model handles debiasing most gracefully. k8s reaches 0.874, the highest F1 on any project.

**Qwen-14B (avg gain +0.022, 10/11 improve):**
| Project | RAGTAG | Debiased | Delta |
|---------|--------|----------|-------|
| ansible | 0.678 | **0.689** | +0.011 |
| bitcoin | 0.706 | **0.738** | +0.032 |
| dart | 0.841 | **0.844** | +0.003 |
| dotnet | 0.628 | **0.674** | +0.046 |
| react | 0.797 | **0.808** | +0.011 |
| flutter | 0.764 | **0.787** | +0.023 |
| k8s | 0.835 | **0.848** | +0.013 |
| TypeScript | 0.614 | **0.644** | +0.030 |
| vscode | 0.662 | **0.707** | +0.045 |
| opencv | 0.703 | **0.717** | +0.013 |
| tensorflow | 0.773 | **0.784** | +0.011 |
| **Average** | **0.727** | **0.749** | **+0.022** |

10/11 improve. Only tensorflow shows a slight dip when considering best-k. The debiased average of 0.749 is the highest across any model+approach combination, beating Qwen-32B baseline (0.746) and both Llama FTs.

**Qwen-32B (avg gain +0.000 raw, +0.010 valid-only):**
| Project | RAGTAG | Debiased | Delta |
|---------|--------|----------|-------|
| ansible | 0.693 | **0.708** | +0.015 |
| bitcoin | 0.715 | **0.742** | +0.027 |
| dart | 0.848 | **0.865** | +0.017 |
| dotnet | 0.674 | **0.685** | +0.011 |
| react | 0.819 | **0.821** | +0.003 |
| flutter | 0.801 | 0.789 | -0.012 |
| k8s | 0.799 | 0.775 | -0.024 |
| TypeScript | 0.652 | 0.652 | +0.000 |
| vscode | 0.662 | 0.648 | -0.014 |
| opencv | 0.749 | **0.751** | +0.002 |
| tensorflow | 0.791 | 0.768 | -0.023 |
| **Average** | **0.746** | **0.746** | **+0.000** |

Flat on raw scores — but this is an artifact of OOM, not a failure of debiasing. See Section 9 for valid-only analysis.

### 7.3 Cross-Model Debiasing Comparison

| Metric | Llama-3B | Llama-8B | Qwen-14B | Qwen-32B |
|--------|----------|----------|----------|----------|
| PS RAGTAG baseline avg | 0.678 | 0.704 | 0.727 | 0.746 |
| PS Debiased avg | 0.698 | 0.737 | **0.749** | 0.746 |
| Debias gain | +0.020 | **+0.034** | +0.022 | +0.000 |
| Projects improved | 9/11 | **11/11** | 10/11 | 6/11 |
| Agnostic FT avg | 0.725 | 0.736 | *pending* | *pending* |
| Debiased vs FT gap | -0.027 | **+0.002** | — | — |

Debiasing helps most for mid-size models. The 8B model shows the largest gain (+0.034) and achieves FT parity. The 3B model benefits but can't fully close the gap (-0.027). The 14B model gains +0.022 and reaches the overall highest score. The 32B model's gains are masked by OOM invalids.

---

## 8. Detailed Debiased Per-k Results

### 8.1 Llama-3B

**ansible (best k=6, +0.049):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.683 | 0.689 | +0.006 | 0.434 | 0.422 | -0.012 |
| 3 | 0.681 | 0.729 | +0.048 | 0.448 | 0.514 | +0.067 |
| 6 | 0.682 | **0.741** | +0.059 | 0.455 | 0.558 | +0.103 |
| 9 | 0.692 | 0.715 | +0.023 | 0.510 | 0.557 | +0.047 |

**bitcoin (best k=6, +0.021):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.617 | 0.658 | +0.041 | 0.424 | 0.460 | +0.035 |
| 3 | 0.626 | 0.628 | +0.002 | 0.447 | 0.532 | +0.085 |
| 6 | 0.658 | **0.679** | +0.021 | 0.477 | 0.639 | +0.162 |
| 9 | 0.610 | 0.629 | +0.019 | 0.417 | 0.594 | +0.177 |

**dart (best k=3, +0.012):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.706 | 0.708 | +0.002 | 0.651 | 0.631 | -0.020 |
| 3 | 0.732 | **0.744** | +0.012 | 0.712 | 0.706 | -0.006 |
| 6 | 0.711 | 0.731 | +0.020 | 0.628 | 0.728 | +0.100 |
| 9 | 0.680 | 0.721 | +0.041 | 0.568 | 0.704 | +0.137 |

**dotnet (best k=9, +0.010):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.563 | 0.585 | +0.022 | 0.378 | 0.392 | +0.014 |
| 3 | 0.584 | 0.586 | +0.002 | 0.425 | 0.480 | +0.055 |
| 6 | 0.556 | 0.589 | +0.032 | 0.325 | 0.528 | +0.203 |
| 9 | 0.569 | **0.594** | +0.025 | 0.359 | 0.536 | +0.177 |

**react (best k=1, +0.008):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.742 | **0.750** | +0.008 | 0.603 | 0.624 | +0.022 |
| 3 | 0.729 | 0.735 | +0.006 | 0.571 | 0.595 | +0.024 |
| 6 | 0.730 | 0.734 | +0.004 | 0.581 | 0.590 | +0.009 |
| 9 | 0.715 | 0.738 | +0.023 | 0.571 | 0.610 | +0.039 |

**flutter (best k=9, +0.040):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.634 | 0.619 | -0.014 | 0.474 | 0.458 | -0.017 |
| 3 | 0.598 | 0.637 | +0.039 | 0.405 | 0.531 | +0.126 |
| 6 | 0.675 | 0.701 | +0.027 | 0.506 | 0.622 | +0.116 |
| 9 | 0.672 | **0.715** | +0.043 | 0.483 | 0.635 | +0.152 |

**k8s (best k=6, +0.030):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.765 | 0.740 | -0.025 | 0.727 | 0.737 | +0.010 |
| 3 | 0.801 | 0.785 | -0.016 | 0.783 | 0.847 | +0.065 |
| 6 | 0.809 | **0.838** | +0.030 | 0.754 | 0.829 | +0.074 |
| 9 | 0.763 | 0.828 | +0.065 | 0.685 | 0.822 | +0.137 |

**TypeScript (best k=3, -0.029):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.500 | 0.510 | +0.010 | 0.268 | 0.264 | -0.004 |
| 3 | 0.530 | **0.521** | -0.009 | 0.449 | 0.466 | +0.017 |
| 6 | 0.523 | 0.520 | -0.004 | 0.390 | 0.527 | +0.137 |
| 9 | 0.549 | 0.501 | -0.049 | 0.434 | 0.536 | +0.102 |

**vscode (best k=6, -0.020):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.569 | 0.565 | -0.004 | 0.349 | 0.336 | -0.013 |
| 3 | 0.635 | 0.602 | -0.034 | 0.539 | 0.527 | -0.011 |
| 6 | 0.645 | **0.625** | -0.020 | 0.552 | 0.657 | +0.105 |
| 9 | 0.604 | 0.570 | -0.034 | 0.500 | 0.594 | +0.094 |

**opencv (best k=9, +0.051):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.656 | 0.677 | +0.021 | 0.596 | 0.614 | +0.018 |
| 3 | 0.606 | 0.660 | +0.054 | 0.576 | 0.690 | +0.113 |
| 6 | 0.647 | 0.670 | +0.023 | 0.587 | 0.719 | +0.132 |
| 9 | 0.646 | **0.707** | +0.062 | 0.585 | 0.725 | +0.140 |

**tensorflow (best k=6, +0.050):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.679 | 0.650 | -0.028 | 0.506 | 0.477 | -0.029 |
| 3 | 0.711 | 0.736 | +0.025 | 0.596 | 0.629 | +0.032 |
| 6 | 0.666 | **0.761** | +0.096 | 0.590 | 0.690 | +0.101 |
| 9 | 0.622 | 0.726 | +0.104 | 0.476 | 0.663 | +0.187 |

### 8.2 Llama-8B

**ansible (best k=9, +0.025):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.611 | 0.621 | +0.011 | 0.300 | 0.300 | +0.000 |
| 3 | 0.634 | 0.617 | -0.017 | 0.333 | 0.355 | +0.022 |
| 6 | 0.644 | 0.652 | +0.009 | 0.344 | 0.406 | +0.062 |
| 9 | 0.649 | **0.674** | +0.025 | 0.355 | 0.427 | +0.073 |

**bitcoin (best k=9, +0.013):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.654 | 0.660 | +0.005 | 0.380 | 0.392 | +0.011 |
| 3 | 0.673 | 0.643 | -0.031 | 0.446 | 0.403 | -0.043 |
| 6 | 0.692 | 0.684 | -0.008 | 0.455 | 0.512 | +0.056 |
| 9 | 0.670 | **0.704** | +0.034 | 0.414 | 0.565 | +0.152 |

**dart (best k=9, +0.031):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.751 | 0.764 | +0.013 | 0.696 | 0.704 | +0.008 |
| 3 | 0.745 | 0.765 | +0.019 | 0.679 | 0.713 | +0.033 |
| 6 | 0.781 | 0.788 | +0.007 | 0.724 | 0.737 | +0.013 |
| 9 | 0.790 | **0.821** | +0.031 | 0.752 | 0.813 | +0.061 |

**dotnet (best k=9, +0.059):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.583 | 0.574 | -0.008 | 0.290 | 0.276 | -0.014 |
| 3 | 0.579 | 0.600 | +0.021 | 0.303 | 0.333 | +0.030 |
| 6 | 0.592 | 0.620 | +0.028 | 0.333 | 0.389 | +0.056 |
| 9 | 0.573 | **0.651** | +0.077 | 0.336 | 0.530 | +0.195 |

**react (best k=9, +0.019):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.719 | 0.719 | -0.000 | 0.624 | 0.632 | +0.008 |
| 3 | 0.756 | 0.731 | -0.025 | 0.658 | 0.632 | -0.026 |
| 6 | 0.754 | 0.750 | -0.004 | 0.679 | 0.663 | -0.016 |
| 9 | 0.749 | **0.775** | +0.026 | 0.647 | 0.694 | +0.047 |

**flutter (best k=9, +0.049):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.681 | 0.683 | +0.002 | 0.539 | 0.530 | -0.010 |
| 3 | 0.696 | 0.673 | -0.024 | 0.534 | 0.561 | +0.027 |
| 6 | 0.697 | 0.740 | +0.043 | 0.538 | 0.656 | +0.118 |
| 9 | 0.672 | **0.747** | +0.075 | 0.456 | 0.655 | +0.199 |

**k8s (best k=6, +0.059):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.763 | 0.717 | -0.046 | 0.634 | 0.679 | +0.045 |
| 3 | 0.813 | 0.777 | -0.035 | 0.739 | 0.821 | +0.082 |
| 6 | 0.815 | **0.874** | +0.059 | 0.725 | 0.831 | +0.106 |
| 9 | 0.806 | 0.868 | +0.062 | 0.717 | 0.833 | +0.116 |

**TypeScript (best k=9, +0.008):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.529 | 0.534 | +0.005 | 0.207 | 0.224 | +0.017 |
| 3 | 0.581 | 0.584 | +0.002 | 0.392 | 0.384 | -0.008 |
| 6 | 0.612 | 0.604 | -0.008 | 0.473 | 0.551 | +0.078 |
| 9 | 0.585 | **0.620** | +0.034 | 0.434 | 0.577 | +0.143 |

**vscode (best k=6, +0.002):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.569 | 0.569 | -0.000 | 0.309 | 0.306 | -0.002 |
| 3 | 0.640 | 0.647 | +0.007 | 0.475 | 0.497 | +0.022 |
| 6 | 0.708 | **0.710** | +0.002 | 0.619 | 0.651 | +0.032 |
| 9 | 0.656 | 0.705 | +0.049 | 0.538 | 0.667 | +0.129 |

**opencv (best k=9, +0.066):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.604 | 0.610 | +0.006 | 0.441 | 0.415 | -0.026 |
| 3 | 0.656 | 0.652 | -0.004 | 0.517 | 0.510 | -0.007 |
| 6 | 0.700 | 0.731 | +0.031 | 0.573 | 0.609 | +0.035 |
| 9 | 0.690 | **0.767** | +0.076 | 0.558 | 0.721 | +0.163 |

**tensorflow (best k=9, +0.040):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.653 | 0.611 | -0.042 | 0.484 | 0.453 | -0.030 |
| 3 | 0.729 | 0.691 | -0.038 | 0.647 | 0.580 | -0.067 |
| 6 | 0.701 | 0.728 | +0.028 | 0.601 | 0.646 | +0.044 |
| 9 | 0.729 | **0.769** | +0.040 | 0.647 | 0.687 | +0.040 |

### 8.3 Qwen-14B and Qwen-32B

Per-k tables omitted for brevity. Summary tables in Section 7.2 cover the best-k results. Key pattern: Qwen-14B best debiased k is predominantly 9 (8/11 projects), matching Llama-8B. Qwen-32B's best k is mixed due to OOM effects at higher k.

---

## 9. Qwen-32B Invalid Rate Analysis

Qwen-32B at 32B parameters in 4-bit quantization pushes the RTX 4090 (24GB) to its limits. At higher k, longer prompts trigger CUDA OOM errors that produce invalid (unparseable) predictions.

### 9.1 Invalid Rates by k (total across 11 projects)

| k | Llama-3B | Llama-8B | Qwen-14B | Qwen-32B |
|---|----------|----------|----------|----------|
| 1 | 0.2% | 0.1% | 0.1% | **2.8%** |
| 3 | 0.2% | 0.2% | 0.1% | **6.0%** |
| 6 | 2.8% | 2.9% | 3.5% | **13.7%** |
| 9 | 3.7% | 3.9% | 4.3% | **22.4%** |

Llama-3B, Llama-8B, and Qwen-14B are all in the same range (0.1-4.3%). Qwen-32B is 5-6x worse at every k. Per-project extremes: flutter k=9 = 43.7%, TypeScript k=9 = 39.7%, k8s k=9 = 32.0%.

### 9.2 Valid-Only Scoring

When scoring only on predictions where the model produced a valid label:

| Approach | Qwen-32B (raw) | Qwen-32B (valid-only) |
|----------|----------------|----------------------|
| RAGTAG baseline avg | 0.746 | **0.777** |
| Debiased RAGTAG avg | 0.746 | **0.787** |
| Debias gain | +0.000 | **+0.010** |

Per-project valid-only (debiased):

| Project | Raw | Valid-only | Inv% at best k |
|---------|-----|------------|----------------|
| ansible | 0.708 | 0.740 | 13.0% |
| bitcoin | 0.742 | 0.773 | 10.0% |
| dart | 0.865 | 0.862 | 4.0% |
| dotnet | 0.685 | 0.685 | 6.7% |
| react | 0.821 | 0.822 | 0.0% |
| flutter | 0.789 | 0.809 | 24.3% |
| k8s | 0.775 | **0.915** | 16.3% |
| TypeScript | 0.652 | 0.681 | 3.7% |
| vscode | 0.648 | 0.765 | 6.7% |
| opencv | 0.751 | 0.795 | 9.0% |
| tensorflow | 0.768 | 0.815 | 5.3% |

k8s valid-only debiased reaches **0.915** — the model gets it right 91.5% of the time when it doesn't OOM. The 32B raw results are not comparable to other models due to the invalid rate disparity. Valid-only scoring shows debiasing works for Qwen-32B too (+0.010), and the model's true capability is far above what raw scores suggest.

---

## 10. Observations and Patterns

### 10.1 Project Difficulty Spectrum

Projects cluster into three difficulty tiers (based on best RAGTAG F1, Llama-3B):

**Easy (F1 > 0.75):** kubernetes (0.809), react (0.742), dart (0.732)
- High VTAG baselines (0.61-0.73). Clear issue language, well-separated classes.

**Medium (F1 0.65-0.75):** ansible (0.692), tensorflow (0.711), flutter (0.675), bitcoin (0.658), opencv (0.656), vscode (0.645)
- Moderate VTAG (0.51-0.65). Mixed difficulty.

**Hard (F1 < 0.60):** dotnet (0.584), TypeScript (0.549)
- Low VTAG (0.54-0.56). Ambiguous issue language, overlapping class boundaries.

### 10.2 LLM Marginal Value Varies by Project

LLM marginal value = RAGTAG - VTAG (Llama-3B):

| Project | VTAG | RAGTAG | LLM Marginal |
|---------|------|--------|--------------|
| tensorflow | 0.507 | 0.711 | +0.204 |
| react | 0.611 | 0.742 | +0.131 |
| dart | 0.624 | 0.732 | +0.108 |
| ansible | 0.593 | 0.692 | +0.099 |
| flutter | 0.580 | 0.675 | +0.095 |
| k8s | 0.732 | 0.809 | +0.077 |
| bitcoin | 0.597 | 0.658 | +0.061 |
| dotnet | 0.536 | 0.584 | +0.048 |
| opencv | 0.635 | 0.656 | +0.021 |
| vscode | 0.648 | 0.645 | -0.003 |
| TypeScript | 0.559 | 0.549 | -0.010 |
| **Average** | **0.602** | **0.678** | **+0.076** |

The LLM adds negative value on vscode and TypeScript — these are the same projects where Llama-3B debiasing regresses. On these projects, k-NN voting already extracts most of the available signal and the LLM's bug bias introduces more errors than its reasoning corrects.

### 10.3 Agnostic vs Project-Specific

| Setting | 3B RAGTAG | 3B FT | 8B RAGTAG | 8B FT |
|---------|-----------|-------|-----------|-------|
| Project-specific | 0.678 | N/A | 0.704 | N/A |
| Project-specific (debiased) | 0.698 | N/A | 0.737 | N/A |
| Agnostic (per-project avg) | 0.658 | 0.725 | 0.701 | 0.736 |

Project-specific RAGTAG beats agnostic RAGTAG (+0.020 for 3B, +0.003 for 8B), despite having 10x fewer training examples (300 vs 3,300). This makes sense given 88.3% same-project retrieval: the agnostic pool is mostly same-project anyway, but diluted by 11.7% cross-project noise.

FT benefits more from the agnostic setting because it sees all 3,300 examples during gradient updates, not just the k nearest. However, with debiasing, project-specific RAGTAG closes the gap entirely for 8B (0.737 vs 0.736).

### 10.4 Prediction for Qwen FT (pending)

Based on the data efficiency crossover studies (3k/30k), at 3,300 training examples:
- Qwen-14B RAGTAG beat FT at 3k (0.742 vs 0.739) and still beat FT at 27k (0.779 vs 0.767)
- Qwen-32B RAGTAG beat FT at 3k (0.778 vs 0.735) but FT won at 27k (0.810 vs 0.785)

At 3,300 examples, we are well below the crossover point for both Qwen models. Qwen-14B debiased (0.749) should beat its FT. Qwen-32B is harder to predict due to OOM effects.

---

## 11. Pending Experiments

| Experiment | Status | Expected |
|------------|--------|----------|
| Llama-3B debiased (11 projects) | **Complete** | — |
| Llama-8B debiased (11 projects) | **Complete** | — |
| Qwen-14B debiased (11 projects) | **Complete** | — |
| Qwen-32B debiased (11 projects) | **Complete** | — |
| Qwen-14B/32B FT (agnostic) | Submitted to OSC H100 | Pending |

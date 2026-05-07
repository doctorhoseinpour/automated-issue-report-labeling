# Paper TODO

Items the next session should pick up. Live status of in-flight runs is in [SESSION_HANDOFF.md](SESSION_HANDOFF.md).

## In flight (no action needed yet — wait for completion)

- [ ] **k=12/15 extension at ctx=8192.** Local for 3B/7B/14B (`run_k12_k15_local_8k.sh`); NRP for 32B (waves 6 and 7 in `scripts/nrp/plan.yaml`). Both PA RAGTAG and PS RAGTAG, plus PS Debiased RAGTAG (margin=3). Idempotent skip on `preds_k15.csv`. Outputs land in canonical `ragtag/` and `ragtag_debias_m3/` directories. When done: re-run analysis utilities to see whether the {0,1,3,6,9} grid still represents the plateau.
- [ ] **DeBERTa-v3-large PA fine-tune (exploratory).** Running locally via `run_transformer_ft.py`. Outputs to `results/issues11k/agnostic/microsoft_deberta_v3_large/finetune_transformer/`. Not yet committed to the paper — decide inclusion after seeing the F1 vs LLM-FT.

## After in-flight runs land

- [ ] **k=12/15 verdict.** If F1 stays flat or drops past k=9, we have a documented null result that strengthens §3.3's k-grid defense. If a model bumps up at k=12 or k=15, decide whether to extend the published grid or footnote the divergence.
- [ ] **DeBERTa decision.** If DeBERTa-v3-large PA F1 is competitive with or above LLM-FT-PA, decide whether to add an encoder baseline section. If not, drop it from the paper entirely.

## Writing — section drafts

- [ ] **§5.1 VOTAG retrieval floor.** Three paragraphs essentially done (peak+plateau / PA-vs-PS / per-class with bug-bias inference). Two-panel figure (kcurve + per-class bar chart) generated under pooled aggregation. Two inline `% TODO:` blocks remaining in [`05_evaluations.tex`](sections/05_evaluations.tex):
  - Motivate the k-sweep (why scan k=1..30 and not just one or two values).
  - Write the RQ1 → §5.2 transition (position 0.60 as the retrieval-only floor; flag bug-bias as recurring across approaches).
- [ ] **§5.2 Bug-bias diagnosis.** Stub only. Groundwork is in place from §5.1 (per-class F1 numbers, embedding-space inference, mechanism→misclass→geometric chain). Extend to the LLM-based methods (zero-shot, RAGTAG, Debiased RAGTAG, FT) and show the asymmetry persists across approaches and model scales.
- [ ] **§5.3 RAGTAG / Debiased RAGTAG analysis (RQ2).** Method recap + headline numbers + per-model curves. Margin-based retrieval debiasing (m=3) is the published intervention; keep wording careful since the FT baseline got tighter at 3 epochs and the Debias gap shrank for some cells. **User picks this up next session.**
- [ ] **§6 Discussion / §7 Threats.** Draft after §5 lands.

## Statistics (after all runs land)

- [ ] **Bootstrap 95% CIs on macro F1** for every method × model × setting cell. 1,000 resamples. Report alongside point estimates.
- [ ] **McNemar's test** for headline pairwise comparisons:
  - Debiased RAGTAG_PS vs FT_PS (per project + aggregated, per model)
  - Debiased RAGTAG_PS vs FT_PA per-project eval (per project + aggregated, per model)
  - RAGTAG_PA vs FT_PA (per model)
- [ ] Implementation: ~50 lines of Python in **`scripts/paper/significance_tests.py`** (new-script convention; do not use `scripts/analysis/`). Use `statsmodels.stats.contingency_tables.mcnemar` and `numpy` resampling for CIs. Compute over pooled predictions (concat per-project preds for PS). Document methodology in §4.4 (1,000 bootstraps; McNemar's with continuity correction; p < 0.05 threshold; b vs c counts in supplementary).

## Hardware specs

- [ ] **§4.5 placeholder.** Need exact server specs. Run `kubectl describe node <node-name>` (or `kubectl describe pod <pod>` for the running mega-runner) to capture: GPU model + VRAM, CPU model + core count, RAM, OS/CUDA versions. Local 4090 specs already known.

## Forward-pointers

- [ ] **§3.3 RAGTAG → §5.3 Debias signpost.** Once §5.3 has a stable label, insert in §3.3:
  > *We additionally introduce a retrieval-debiasing intervention applied on top of \ragtag; we describe its algorithm and present its empirical motivation in \cref{sec:debias}.*

## Conventions / methodology

- [x] **Pooled aggregation for all paper metrics** (decided 2026-05-06; methodology paragraph in [`04_setup.tex`](sections/04_setup.tex) §"Evaluation Metrics"). All macro $F_1$ numbers — PA and PS, every approach — are computed by concat-then-evaluate on the 3,300-issue test set. Per-project mean is not used for headline numbers.
- [ ] **New paper-figure scripts go in `scripts/paper/`** with pooled aggregation baked in. Existing `scripts/analysis/*.py` use per-project mean for PS — do not retrofit, do not consume their PS outputs for paper artifacts. Two scripts already exist: `fig_vtag_kcurve.py`, `tab_vtag_peak.py`. Future additions for RQ2 (RAGTAG analysis), RQ3 (debiasing), bug-bias diagnosis, leaderboard, scaling curves, etc., should follow the same pattern.

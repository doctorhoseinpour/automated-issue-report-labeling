# SANER 2027 submission sources

IEEE-format port of the ESEM 2026 paper (`../paper/`, LIPIcs) for the
**SANER 2027 Research Track**. This directory is the only place the paper
should be edited from now on; `../paper/` is frozen.

## Build

```bash
./build.sh          # pdflatex -> bibtex -> pdflatex x2, then reports pages / overfull boxes
```

Main file: `SANER2027-main-labeling.tex`. `IEEEtran.cls` and `IEEEtran.bst`
(v1.8b / v1.14, the official CTAN versions) are vendored here so the paper
builds on any TeX Live, including machines without `texlive-publishers`.
Overleaf works too: upload the whole directory and set the main file.

## Layout

```
SANER2027-main-labeling.tex   preamble, anonymous author block, \input list
sections/00_abstract.tex      abstract + IEEE keywords
sections/01_intro ... 08_conclusion.tex
sections/09_data_availability.tex   unnumbered section, directly after Conclusion (SANER-mandatory)
tables/                       results_master (table*, generated), method_comparison_ci (column)
figures/                      kcurves, per_project_diff (generated; see below) + fao.pdf (approach overview)
refs.bib                      bibliography (IEEEtran style)
```

## Submission rules that this template enforces

| Rule (SANER 2027 CfP) | How it is handled |
|---|---|
| `\documentclass[10pt,conference]{IEEEtran}`, no `compsoc` | main file |
| 10 pages main text + up to 2 pages of references only | check `build.sh` output; references start on the page shown by `pdftotext` |
| Double-blind | `Anonymous Author(s)` block; real block kept in a `comment` environment for camera-ready; self-citations must stay third person |
| Data Availability statement after the Conclusion | `sections/09_data_availability.tex` (`\section*`) |
| Anonymous artifact link | `replication_package` entry in `refs.bib` (Figshare private link) |

## What changed relative to the first paste

Text of all sections was kept as pasted (including the supervisor's copy-edits).
Only structure and layout were changed:

- Preamble rebuilt: added the packages the sections actually use
  (`comment`, `xspace`, `booktabs`, `amsmath`, tikz `positioning`/`calc`),
  dropped unused ones (`listings`, `float`, `subcaption`, which also breaks
  IEEEtran captions), added `microtype`, `balance`, `hidelinks`, and IEEE-style
  cleveref names (`Fig.`, `Table`, `Section`).
- Abstract converted from the ESEM structured form (Background/Aims/...) to a
  single IEEE paragraph; the structured version is kept in a comment block.
- Two-panel figures (`vtag_kcurve`, `ragtag_kcurve`, `finetune_comparison`,
  `per_project_diff`) and the approach overview are `figure*` (full text
  width); `bragtag_kcurve` stays in the column.
- Prompt figure (tikz) re-laid-out to exactly one column width (was a
  hard-coded 10.3 cm box that overflowed by 146 pt).
- `method_comparison` is a `table*` (was a 12-column table shrunk into one
  column); `bragtag_results` fits the column at `\footnotesize`.
- Data Availability is an unnumbered section before the references;
  the `\clearpage` before the bibliography was removed.
- Bibliography style `IEEEtran` instead of `ieeetr`.

## Page budget (as of 2026-09-16)

`build.sh` reports 12 pages. References start on page 11 after about 560 words
(the Conclusion and the Data Availability statement), i.e. the main text runs
to roughly page 10.7 against a hard limit of 10. **About 0.7 page (~1.4
columns) of main text must be cut** (or made up by the reframe in the revision
plan). Figures are already at readable minimum widths; do not shrink them
further to make room.

`tables/encoder_baselines.tex` (SetFit / RoBERTa baselines) is copied here but
is not yet `\input` anywhere; it is a `table*` and will cost roughly a third
of a page when integrated.

## Still to do before submission (content, not layout)

See `../docs/SANER_REVISION_PLAN.md`. In particular: integrate the encoder
baselines (`../paper/tables/encoder_baselines.tex`,
`../docs/ENCODER_BASELINES_RESULTS.md`), reframe title/abstract, and trim the
main text to the 10-page limit.

## Metric-triangulation rewrite (2026-09-17)

Response to ESEM reviewer B ("methodological triangulation"). Precision, recall
and the *bug share* of predictions were added to RQ1–RQ4; **accuracy is not
reported anywhere** (author decision: on the balanced test set it equals macro
recall). Two candidate layouts were drafted; the author chose the one that
extends the existing floats: `tables/bragtag_results_ext.tex`,
`tables/method_comparison_ext.tex`, and precision/recall panels in
`figures/vtag_kcurve_pr.pdf`, `figures/ragtag_kcurve_pr.pdf`,
`figures/finetune_comparison_pr.pdf`. All changed text is blue (`\fixed{}`)
until the supervisor pass. Build: 14 pages, bibliography starts on page 13, so
about 2 pages of main text must be cut (this includes the 0.7 page noted above).

Regenerate with `../scripts/paper/tab_triangulation.py` and the `--pr-panel`
option of the three figure scripts (lab machine). Every number in the blue
prose is in `../paper/tables/triangulation_all_cells.csv`. The alternative
layout's assets (`paper/tables/triangulation.tex`, `paper/figures/bias_plane.*`,
`*_kcurve_single.*`) remain under `../paper/` for reference only.

## Presentation pass on Section V (2026-09-22)

The eight floats of the evaluation section (five figures, three tables, two
float-only pages, figure type scaled to ~4.5 pt) were consolidated into four
floats drawn at their printed size:

| Float | File | Generator (runs on any machine) |
|---|---|---|
| Fig. k-curves (a) VOTAG, (b) RAGTAG, (c) BRAGTAG vs RAGTAG; `figure*` | `figures/kcurves.pdf` | `scripts/paper/fig_kcurves.py` |
| Table results master: VOTAG, zero-shot, RAGTAG, BRAGTAG, FT-PS, FT-PA per size, raw predictions only; `table*` | `tables/results_master.tex` | `scripts/paper/tab_results_master.py` (`--fallback-column` restores the +VOTAG column) |
| Fig. per-project heatmap (11 projects x 4 sizes, transposed); column | `figures/per_project_diff.pdf` | `scripts/paper/fig_per_project_diff.py` |
| Table CI: macro-F1 differences vs PA fine-tuning, one raw row and one +VOTAG row per model and for "All sizes", bold = CI excludes zero; column | `tables/method_comparison_ci.tex` | `scripts/paper/tab_method_comparison_ci.py` from `../paper/tables/method_comparison_ci.csv` (written by `significance_method_comparison.py`) |

All generators read the small CSVs in `../paper/tables/` (exempted from the
global `*.csv` ignore rule in `.gitignore`, so commit them with the figures):
`triangulation_all_cells.csv` (every cell; written by `tab_triangulation.py`
on the lab machine), `fallback_macro_f1.csv` (also written by
`tab_triangulation.py`; only used with `--fallback-column`),
`per_project_diff.csv` (written by `fig_per_project_diff.py --from-results`)
and `method_comparison_ci.csv` (written by `significance_method_comparison.py`).
`method_comparison_ci.csv` is computed (issue-level resampling, 2026-09-24).
`fallback_macro_f1.csv` and `per_project_diff.csv` are still seeded by hand from
the previously published numbers (3 d.p.); rerun their lab-machine scripts to
replace them. A missing CI would print as `--`.

Style: `scripts/paper/_figstyle.py` (shared model colours, Okabe-Ito set
validated for colour-blind safety; Qwen-14B is now green instead of brown;
type 6.5-8 pt at print size). The k-curve figure and the master table are
`\input` at the top of Section V so they land on the section's first full
page; the other floats stay next to the RQ that uses them.

Superseded floats and drafts that are no longer `\input`/included (old k-curve,
per-class and cost figures, approach-diagram variants, the error-profile figure,
the encoder-baselines table, and the old comparison/BRAGTAG tables) were removed
on 2026-09-25 so the folder uploads cleanly to Overleaf; recover them from git
history (commit 501e9d8 or earlier) if needed.

Build after this pass: 12 pages, main text (through Data Availability) ends
on the last line of page 10 and the references start on page 11, i.e. the
10 + 2 limit is met with no spare room; any added prose must be paid for
elsewhere. Moving the +VOTAG column into Table II (six extra rows) had pushed
one line onto page 11; it was recovered with `\arraystretch{0.97}` in Table I
(set in `tab_results_master.py`). Note that savings in the page-8 floats
(Table II, the heatmap) do not move the final break, page-7 floats do.

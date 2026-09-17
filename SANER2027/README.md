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
tables/                       bragtag_results (column), method_comparison (full-width table*)
figures/                      PDFs copied from ../paper/figures (regenerate with ../scripts/paper/*.py on the lab machine)
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

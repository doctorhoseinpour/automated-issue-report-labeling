# S1: names, number format, draft colors, tables and figures (mechanical pass)

**Read `docs/reframe/BRIEF.md` in full first, then `docs/reframe/LOG.md`.** You are S1. S2 may be running
in parallel in another terminal: it only writes `docs/reframe/RW_DOSSIER.md` and
`docs/reframe/refs_candidates.bib`. Never stage or commit those files, and never use `git add -A`.

## Goal

After S1, the paper uses the new method names and the percent format **everywhere**: prose, tables,
figures and the approach diagram. It has no draft colors, and it reads exactly as consistently as
before. This is a mechanical pass. Do **not** restructure paragraphs, reframe claims or remove numbers;
S3 does that. Change a sentence only as far as the new names or number format force it to change.

Work in the order below. Each numbered step ends with a build and `pagecheck.py`. Commit after steps 2,
4, 5 and 7.

## 1. Preflight

1. Confirm `pwd` is `/home/alireza/Desktop/my_projects/saner-reframe` and the branch is
   `saner-reframe`. Read LOG.md.
2. Read these files end to end before changing anything:
   - `SANER2027/SANER2027-main-labeling.tex`;
   - every file in `SANER2027/sections/`;
   - `SANER2027/tables/{results_master,method_comparison_ci,method_cost}.tex`;
   - `scripts/paper/{tab_results_master,tab_method_comparison_ci,fig_kcurves,fig_per_project_diff,_figstyle}.py`.
3. **References on page 11.** In `SANER2027/SANER2027-main-labeling.tex`, put `\clearpage` on its own
   line directly before `\balance` (above `\bibliographystyle{IEEEtran}`). The user requires the
   references to start on page 11 (BRIEF §1). Build and confirm with `pagecheck.py` that "References
   start: page 11, left column", the PDF has at most 12 pages, and page 10 has the known ~9 free lines.
   Commit this one-line change on its own: `SANER reframe S1: start references on page 11`.
4. Build (`SANER2027/build.sh`), run `python3 docs/reframe/pagecheck.py`, and save
   `pdftotext -layout SANER2027/SANER2027-main-labeling.pdf` to your scratchpad as `baseline.txt`. This
   is the baseline for the color-removal diff, taken after the `\clearpage`.

## 2. Remove the draft colors (text must not change)

The sections contain 83 `\fixed{...}`, 22 `\coauthor{...}` and 4 `coauthoranswer` environments (all in
`05_evaluations.tex`).

1. Write a **brace-aware** Python script in your scratchpad. Do not use sed: the arguments contain nested
   braces and span lines. The script unwraps `\fixed{X}` → `X` and `\coauthor{X}` → `X` in
   `SANER2027/sections/*.tex`, and replaces `\begin{coauthoranswer}` / `\end{coauthoranswer}` with
   `\begin{myboxi}` / `\end{myboxi}`. It leaves comment lines (starting with `%`) and `\begin{comment}`
   blocks alone, or handles them identically, as long as the output compiles.
   - **Space trap:** `\color` inside these macros swallows a leading space. Where the source has
     `word\fixed{ (x)}` or `\fixed{A}\fixed{ B}`, check that unwrapping gives exactly one space where the
     PDF had one. The text diff in point 4 catches any mistake.
2. In `scripts/paper/tab_results_master.py` and `scripts/paper/tab_method_comparison_ci.py`, change
   `\centering\color{blue}` to `\centering`. In the hand-written `SANER2027/tables/method_cost.tex`, do
   the same.
3. In the preamble, delete the definitions that are now unused: `\fixed`, `\coauthor`, `\coauthornote`,
   `coauthorcolor`, the `coauthoranswer` environment, the `\pdfstringdefDisableCommands` line for
   `\coauthor`, `\change`, `\rev`, `\minor`, and the "Metric-triangulation rewrite" comment. **Keep
   `\fixme`** (commented uses exist).
4. Rerun the two table generators (`python3 scripts/paper/tab_results_master.py`,
   `python3 scripts/paper/tab_method_comparison_ci.py`) and rebuild. Then run
   `pdftotext -layout` into `nocolor.txt` and `diff baseline.txt nocolor.txt`. **The diff must be
   empty.** If it is not, fix the unwrapping until it is.
5. Commit: `SANER reframe S1: drop draft colors (text unchanged)`.

## 3. Numbers sheet (before touching any number)

Write `scripts/paper/numbers_sheet.py`. It reads `paper/tables/triangulation_all_cells.csv`,
`paper/tables/method_comparison_ci.csv` and the GB/hour values of `SANER2027/tables/method_cost.tex`
(parse the .tex or hard-code them with a comment citing the file), and writes
**`docs/reframe/NUMBERS.md`**.

- Put a header saying the file is generated, when, and from which files.
- Every value is in the new format (scores in % with one decimal; differences in points with one
  decimal; shares in whole %, with one decimal beside each value in brackets so boundary cases are
  visible).
- Every range (min–max over the four model sizes) is computed from **exact** values and then rounded.
- Group the sections like this:
  1. **Best configurations.** For each method (zero-shot, RAG PS, RAG PA, filtered RAG PS, fine-tuning
     PS, fine-tuning PA, $k$NN voting PS and PA) and each model: best k, macro F1/P/R, per-class P/R,
     invalid rate, bug share of predictions, question→bug and question→feature rates. "Best" means the
     highest macro F1 on the grid, the same rule as `tab_results_master.py`'s `best_configs`.
  2. **RQ1 (RAG).**
     - Zero-shot range.
     - $k$NN voting peaks (PS k = 15, PA k = 16) and its question→bug rate.
     - The number of nonzero-k RAG configurations beating both zero-shot and the best $k$NN voting in
       macro F1, macro P and macro R (the text says all 48).
     - RAG best-k range; mean gain over zero-shot.
     - Whether any model improves from k = 12 to 15.
     - Question→bug and question→feature at zero-shot vs RAG best k.
     - Share of predictions labeled bug, bug precision and question recall, zero-shot vs RAG
       (averaged over models).
     - The share of the macro-F1 gain that comes from question-class F1 (the text says 56–70%).
     - The fraction of false bug predictions that are questions (the text says about 80%).
  3. **RQ2 (filtered RAG).**
     - Filtered RAG minus RAG at every k in {1, 3, 6, 9, 12, 15} per model: the minimum at k ≥ 6 and
       the maximum shortfall at k ≤ 3.
     - The difference at each method's best k.
     - Per-class changes: question F1, question P and R, bug P and R, bug F1, feature F1.
     - Question→bug and question→feature changes.
     - Question→bug averaged over k in [1, 15] for RAG and filtered RAG.
     - The share of queries left with no examples at k = 1 and k = 3 (40% and 18% in the text). This is
       **not** in the CSV; copy it from the text and mark it "from the draft, not recomputable here".
     - Validation-split calibration numbers (89.1%, 56.7%, −1.36, the 95th percentile of 5): same
       treatment.
     - The per-model CIs of filtered RAG minus RAG from the draft text ([+0.002, +0.030], etc.),
       converted to points and marked "from the draft".
  4. **RQ3 (fine-tuning).**
     - Same-data differences (RAG − FT PS, filtered RAG − FT PS) per model, with ranges.
     - FT PA − FT PS gain per model and range.
     - Pooled differences with CIs from `method_comparison_ci.csv` (raw and fallback, per model and
       "All"), converted to points.
     - Which CIs exclude zero.
     - **Fixed-k gaps:** filtered RAG PS and RAG PS minus FT PA for each k in {6, 9, 12, 15}, per model
       and mean.
     - Error profile: bug share, bug recall and precision, question→bug for FT PA and filtered RAG.
     - Macro recall and precision gaps at 7B and 14B.
     - Invalid-rate ranges for RAG, filtered RAG and FT PA.
     - The share of true bugs with invalid outputs for filtered RAG at 7B and larger (the text says
       6–8%); compute it from `conf_bug_to_invalid`.
     - Memory: GB per method and model, % saving per model, mean.
     - Times (h).
     - $k$NN voting cost (0.2 GB, under 4 s).
  5. **Discussion.** Failure-analysis counts (116/20/14 of 150; 44/14/2 of 60 invalid outputs; from the
     draft) and the $k$NN voting vs Qwen-3B zero-shot gap.
  6. **Heatmap.** Filtered RAG minus FT PA per project and model, from `per_project_diff.csv` (points,
     one decimal), and the count of cells ≥ 0. Note that the CSV is hand-seeded at 3 d.p. and that one
     cell is exactly 0.000.
- Run it and read NUMBERS.md. Check it against BRIEF §9. If an anchor disagrees, trust the script after
  checking your code, and note the disagreement in LOG.md.

## 4. Rename the methods

1. **Preamble.** Delete `\votag`, `\ragtag` and `\bragtag`, so any leftover use breaks the build. Add:
   ```latex
   \newcommand{\knn}{$k$NN voting\xspace}
   \newcommand{\rag}{RAG\xspace}
   \newcommand{\frag}{filtered RAG\xspace}
   \newcommand{\Frag}{Filtered RAG\xspace}
   ```
   Keep `\irc`.
2. **Replace every use** in `SANER2027/sections/*.tex` and the tables, and in the generator strings
   (`tab_results_master.py` emits `\ragtag\ (PS)` and similar). Use `\Frag` at the start of a sentence.
   Avoid starting a sentence with `\knn`: rephrase, for example "With \knn, ..." or "The \knn baseline
   ...". Watch for:
   - Possessives: `\bragtag's` → `\frag's`.
   - `\ragtag\ ` (backslash-space) vs `\ragtag` (xspace). Both must become valid.
   - Plurals and groupings that included VOTAG: "our three proposed retrieval-based IRC methods (VOTAG,
     RAGTAG, BRAGTAG)", "RAG-based methods", "retrieval-based methods", "both retrieval-based methods".
     $k$NN voting is now a baseline, so make these phrases mean "RAG and filtered RAG" (for example,
     "both RAG variants") and describe $k$NN voting separately as the retrieval-only baseline. Make the
     smallest grammatical change; S3 rewrites these paragraphs.
   - Phrases like "\votag PS" → "\knn with the target project's issues" or "\knn (PS)".
   - "Fine-Tune" in prose → "fine-tuning"; "LoRA" alone used as a method name ("than LoRA's") →
     "fine-tuning".
3. **Rewrite the sentences that define the old names,** keeping their content:
   - `01_intro.tex`: the paragraph with **Vo**ting-based **TAG**ging, **R**etrieval-**A**ugmented
     **G**enerative **TAG**ging and **B**alanced RAGTAG. Remove the bold-letter acronym expansions.
     Introduce each by what it does, with the name in parentheses (BRIEF §7 rule 3).
   - `03_approach.tex`:
     - subsection titles: "$k$NN Voting Baseline", "Retrieval-Augmented Few-Shot Prompting (RAG)",
       "Filtered RAG";
     - the section intro that calls all three "our three proposed retrieval-based IRC methods";
     - the approach-figure caption → "Overview of the $k$NN voting baseline, retrieval-augmented few-shot
       prompting (RAG), filtered RAG, and the LoRA fine-tuning baseline."
   - `05_evaluations.tex`: the end of old RQ2, "we ... refer to the resulting method as **B**alanced
     RAGTAG (BRAGTAG)" → "we refer to the resulting variant as filtered RAG".
   - `00_abstract.tex` and `08_conclusion.tex`: the parenthetical names.
   - The prompt figure caption in `03_approach.tex`.
4. **Tables.**
   - `tab_results_master.py` row labels: `$k$NN voting (PS)`, `$k$NN voting (PA)`, `Zero-shot`,
     `RAG (PS)`, `Filtered RAG (PS)`, `Fine-tuning (PS)`, `Fine-tuning (PA)`.
   - `tab_method_comparison_ci.py`: column heads "RAG" and "Filtered RAG" under a spanning head
     "Difference from PA fine-tuning", or similar, as long as it fits the column. The `+\votag` sub-rows
     become `\quad + fallback`. The caption explains once: "+ fallback: invalid outputs of both methods
     labeled by $k$NN voting."
   - `method_cost.tex`: rows `$k$NN voting`, `RAG`, `Filtered RAG`, `Fine-tuning`, and the caption.
5. Rebuild. It must compile with no undefined control sequences. Grep:
   `grep -rn '\\votag\|\\ragtag\|\\bragtag' SANER2027/ scripts/paper/tab_*.py scripts/paper/fig_*.py`.
   Only comments may remain.
6. Commit: `SANER reframe S1: rename methods (kNN voting, RAG, filtered RAG)`.

## 5. Number format in tables and figures

1. **`tab_results_master.py`.**
   - Print P, R and macro F1 as `f"{100*x:.1f}"`. The invalid rate is already in %.
   - Add "(%)" to the header or caption: "All values in percent."
   - Bold the best macro F1 per model block by exact value. If another value in the same block prints
     identically, bold it too.
   - Check the column widths: the table must not get wider than `\textwidth` (the build reports
     overfull boxes).
2. **`tab_method_comparison_ci.py`.** Print Δ and CI bounds in points with one decimal
   (`f"{100*x:+.1f}"`). The caption says "(percentage points)". Keep the bold rule (CI excludes zero).
3. **`fig_kcurves.py`.**
   - Panel titles "(a) $k$NN voting", "(b) RAG", "(c) Filtered RAG vs. RAG (PS)".
   - The annotation "VOTAG best (...)" → "$k$NN voting best (59.5)" or similar, computed.
   - Legend labels.
   - y values ×100 and the y label "Macro $F_1$ (%)"; tick labels without three decimals.
   - Keep sizes, fonts and colors from `_figstyle.py` unchanged.
   Run `python3 scripts/paper/fig_kcurves.py --help` first, then run it with defaults.
4. **`fig_per_project_diff.py`.** Cell text `f"{100*v:+.1f}"` and the colorbar label
   "Filtered RAG − fine-tuning (points)". Run it **without** `--from-results` (that flag needs the lab
   machine).
5. Update the captions in `05_evaluations.tex` (`fig:kcurves`, `fig:per-project-diff`) for the new
   names and units.
6. Check every regenerated figure with `pdftotext figures/<name>.pdf -` for old names or three-decimal
   numbers, and look at the PNG/PDF (Read tool) to confirm that nothing overlaps.
7. Rebuild, run pagecheck, and commit: `SANER reframe S1: tables and figures in percent`.

## 6. The approach diagram (`SANER2027/figures/fao.pdf`)

The diagram is a draw.io export whose source XML is embedded in the PDF's **Subject** metadata
(`pdfinfo SANER2027/figures/fao.pdf` shows `%3Cmxfile ...`).

1. Extract it with Python: read the Subject/Info field (for example with `pypdf` if installed, or parse
   `pdfinfo -meta` / `strings` output) and URL-decode it. If the `<diagram>` element's content is
   compressed (base64 text instead of `<mxGraphModel>`), decode it with base64, then raw inflate
   (`zlib.decompress(data, -15)`), then URL-decode. Save the full `<mxfile>` to
   `SANER2027/figures/src/fao.drawio` and commit it.
2. List every `value="..."` label. Replace old names and wording that no longer fits:
   - "A. VOTAG: retrieval-only classification" → "A. $k$NN voting: retrieval-only baseline";
   - "B. RAGTAG: retrieval-augmented few-shot prompting" → "B. RAG: retrieval-augmented few-shot
     prompting";
   - the BRAGTAG box → "C. Filtered RAG: ...", keeping its description;
   - "Proposed Methods" → "Retrieval-based classifiers";
   - any other occurrence.
   Keep geometry unchanged, and keep the new labels no longer than the old ones so boxes do not
   overflow. If the diagram uses math (`math="1"`), `$k$NN` may render as math; if unsure, use "kNN".
3. **Export.** Try the draw.io desktop CLI. Download the latest Linux AppImage from the official releases
   page `https://github.com/jgraph/drawio-desktop/releases` into your scratchpad, `chmod +x` it, then run
   `./drawio.AppImage --no-sandbox -x -f pdf --crop --embed-diagram -o fao_new.pdf fao.drawio`. If it
   needs FUSE, use `--appimage-extract` and run the extracted binary. If there is no display, try
   `xvfb-run` if installed. Compare the result with the old PDF by rendering both to PNG
   (`pdftoppm -r 80`) and viewing them. If they match apart from the labels, replace
   `SANER2027/figures/fao.pdf`.
4. **If the export cannot be made to work within about 20 minutes,** keep the old `fao.pdf`, commit the
   edited `fao.drawio`, and put a **MANUAL STEP** at the top of your LOG entry: "Open
   `SANER2027/figures/src/fao.drawio` in https://app.diagrams.net, File → Export as → PDF, tick Crop and
   Include a copy of my diagram, save over `SANER2027/figures/fao.pdf`." The old diagram still shows
   VOTAG/RAGTAG/BRAGTAG until the user does this; say so.

## 7. Numbers in the prose

1. Convert **every** number in `SANER2027/sections/*.tex` (and table and figure captions) to the new
   format, **using NUMBERS.md**. Never shift the decimal point of the old rounded text; look each value
   up. Examples of what changes:
   - "0.595 macro $F_1$" → "59.5\% macro $F_1$";
   - "by 0.021--0.060 macro~$F_1$" → "by 2.1--6.0 points" (only if NUMBERS.md gives exactly that range);
   - "trails it by 0.028 macro~$F_1$" → "trails it by 2.8 points";
   - "(+0.001 macro~$F_1$ on average)" → "(0.1 point ahead on average)";
   - "bug recall is approximately 0.73 and precision is 0.55--0.56" → "bug recall is about 73\% and
     precision 55--56\%".
   - CIs quoted in prose ("$[+0.002, +0.030]$", "95\% CI $[-0.020, +0.001]$"): convert to points for
     now (S3 moves them out of the prose).
   - Leave shares already in % unchanged unless NUMBERS.md shows that a boundary value was rounded
     wrongly.
2. In `04_setup.tex` (Evaluation Metrics), add one sentence: "We report scores in percent and
   differences between methods in percentage points (points)."
3. **Do not delete numbers or restructure sentences**; S3 does the number diet. Change only the format
   and the unit words the format needs ("points").
4. **Checks.**
   - Run `pdftotext SANER2027/SANER2027-main-labeling.pdf - | grep -nE '(^|[^0-9.])0\.[0-9]{3}'`; it
     must return nothing. Also grep for `[0-9]\.[0-9]{3}` generally.
   - Old names in the PDF text: `grep -nE 'VOTAG|RAGTAG|BRAGTAG'` must return nothing (unless the
     diagram export failed; then only the figure's text may remain).
   - Build with no undefined references, then run pagecheck. The format change should not change the
     length much. If the main text now crosses page 10, shorten a few words of the number phrases you
     just changed and log it.
5. Commit: `SANER reframe S1: prose numbers in percent and points`.

## 8. Finish

Append your LOG.md entry: files changed, the pagecheck output, the manual diagram step if any, anchor
disagreements, the heatmap 0.000 cell, and anything S3 must know (for example, sentences that now read
awkwardly because of the renaming, which S3 should rewrite). End with `Status: DONE`. Push. Tell the
user, in a short message, what changed and whether the manual diagram export is needed.

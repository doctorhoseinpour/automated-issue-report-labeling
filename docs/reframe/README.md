# SANER 2027 reframe: how to run the sessions

This folder is a **separate copy** of the repository (git worktree, branch `saner-reframe`). The draft as
it was before the revision is preserved on branch `encoder-baselines` at tag `saner-draft-2026-09-25`,
pushed to GitHub, in `~/Desktop/my_projects/automated-issue-report-labeling`. No session writes there.

## What the sessions fix

1. Too many capitalized method names (VOTAG, RAGTAG, BRAGTAG) → **$k$NN voting** (a baseline),
   **RAG**, and **filtered RAG**.
2. Related Work only surveys → every paragraph **compares and contrasts** with this study.
3. The framing sells small F1 gains → the paper leads with **"matches LoRA fine-tuning's macro F1 with
   11× less labeled data per project, no training, and a third less GPU memory"**. Numbers are in
   percent and points, with no three-decimal values, and there are three RQs instead of four.
4. AI-sounding, vague or ambiguous passages → rewritten; Threats rewritten.

The user's layout requirement for the final draft: **exactly 10 full pages of main text; References
start on page 11.**

`BRIEF.md` has every rule and decision. Each `S<n>_*.md` is one session's prompt.

## Run order

| Step | Session | Start when | Rough time |
|---|---|---|---|
| 1 | **S1** names, percent format, colors, floats | now | 1–1.5 h |
| 1 | **S2** related-work research dossier (parallel with S1, second terminal) | now | 1 h |
| 2 | **S3** reframe, three RQs, number diet | S1 logged DONE | 1.5–2 h |
| 3 | **S4** Related Work rewrite | S2 and S3 logged DONE | 1 h |
| 4 | **S5** tone, text-level accuracy, Threats, fill page 10 | S4 logged DONE | 1–1.5 h |
| 5 | **S6** verification, reviewer panel, change-view PDF | S5 logged DONE | 45 min |

## How to start a session

```bash
cd ~/Desktop/my_projects/saner-reframe
claude
```
Then paste, for S1 for example:

> Read docs/reframe/BRIEF.md and docs/reframe/S1_names_numbers.md, then carry out S1.

(Replace S1 and the file name for the other sessions. S2's file is `S2_related_work_research.md`, S3's
is `S3_reframe.md`, S4's is `S4_related_work.md`, S5's is `S5_tone_accuracy_threats.md`, S6's is
`S6_verify.md`.)

Tips:
- Close this folder's `.tex` files in VS Code while a session runs; VS Code's build on save can
  overwrite a session's edit with a stale buffer.
- Each session commits and pushes its own work and appends to `LOG.md`. Read the LOG entry, and skim
  the PDF, before starting the next session.
- Each session stops and asks when a decision is yours: novelty risks, De Vito et al., unverifiable
  claims.
- To see what changed so far: `git log --oneline saner-draft-2026-09-25..HEAD` and
  `git diff saner-draft-2026-09-25 -- SANER2027/sections`.

## Tools

- `SANER2027/build.sh`: build the PDF.
- `python3 docs/reframe/pagecheck.py [--final]`: page budget. It shows where the main text ends, the
  free lines on page 10, and where the references start and end. `--final` fails unless page 10 is full
  and the references start on page 11.
- `scripts/paper/numbers_sheet.py` (written by S1) → `docs/reframe/NUMBERS.md`: every number the prose
  may use, in the new format.

## Before submitting (manual)

- If S1 could not export the approach diagram, open `SANER2027/figures/src/fao.drawio` in
  app.diagrams.net and export it as PDF (Crop, and include a copy of the diagram) over
  `SANER2027/figures/fao.pdf`.
- Paste the new abstract into the EasyChair abstract field (the registered one is from 2026-09-21).
- Optional, needs the lab machine: `fig_per_project_diff.py --from-results` replaces the hand-seeded
  heatmap CSV; one cell is exactly 0.000, and the "16 of 44" sentence relies on it.
- When you are happy, merge `saner-reframe` back (or submit from here), and remove the worktree with
  `git worktree remove ../saner-reframe` from the main clone.

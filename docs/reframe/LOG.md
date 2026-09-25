# Reframe log

Every session appends one entry at the bottom, in this format:

```
## S<n>: <short title> (<date, time>)
- Changed: <file> → <one line>, ...
- Page state: <pagecheck.py output, the key lines>
- Numbers: <anything about NUMBERS.md, anchors, disagreements>
- Open questions for the user: <list, or "none">
- Notes for the next session: <list>
Status: DONE | PARTIAL (<why>)
```

## S0: setup (2026-09-25)

- Preserved the draft: commits `c6cd613` (SANER2027 fixes) and `b93deee` (follow-up study notes and
  scripts) on `encoder-baselines`, pushed; tag `saner-draft-2026-09-25` pushed.
- Created this worktree on branch `saner-reframe` from the tag, pushed with upstream.
- Wrote `docs/reframe/` (README, BRIEF, S1–S6 prompts, this LOG, `pagecheck.py`).
- Page state at the tag: 12 pages. The main text ends on page 10, right column, with 9 free lines; the
  references start on page 10 and end on page 12 (page 12 holds only the last entry). A test build with
  `\clearpage` before `\balance` puts the references at the top of page 11 and ending on page 12 with
  room to spare (about 100 body-line equivalents). S1 adds that `\clearpage`.
- Known build warnings at the tag: two `Overfull \vbox ... while \output is active` from `\balance` on
  the reference pages (harmless).
- User decisions: names $k$NN voting / RAG / filtered RAG; percent with one decimal and points; three
  RQs in build-up order; exactly 10 full pages of main text with references from page 11; work on the
  paper's text only, with no checks against the implementation.
- Open questions for the user: De Vito et al. (TOSEM 2026), to be revisited by S2; the heatmap's 0.000
  cell (lab machine).
Status: DONE

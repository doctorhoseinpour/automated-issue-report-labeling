# Paper TODO

Items to circle back to. Not blocking the current writing thread.

## During / after evaluation sections (§5–§7)

- [ ] **Significance testing in §4.4 Metrics.** I (the author) am not sure what statistical-significance test fits this study yet. Decide while writing \cref{sec:rq1,sec:rq2,sec:rq3} whether to add paired tests, bootstrap CIs, or McNemar's. If yes, come back to §4.4 and document the methodology there. If no, no action needed.

- [ ] **Hardware specs in §4.5.** Need exact server specs. Run `kubectl describe node <node-name>` on the cluster (or `kubectl describe pod <pod>` for the running mega-runner) to capture: GPU model + VRAM, CPU model + core count, RAM, OS/CUDA versions. Fill in §4.5 placeholder once known.

## After §7 evaluation section is drafted

- [ ] **Add debiasing forward-pointer to §3.3 RAGTAG.** Insert after the existing RAGTAG content:
  > *We additionally introduce a retrieval-debiasing intervention applied on top of \ragtag; we describe its algorithm and present its empirical motivation in \cref{sec:rq3}.*
  
  Replace `\cref{sec:rq3}` with the actual subsection label once §7.1 is written (likely `sec:debias` or `sec:rq3-debias`).

  **Why deferred:** debiasing's algorithm and motivation live in §7.1 alongside its empirical evidence (Option B in the framing discussion). The §3.3 sentence is a one-line signpost so methodologically-conservative reviewers don't wonder where the third contribution went.

  **Context:** §3.6 was originally planned as a debiasing subsection but dropped after the content-economy argument: the algorithm is trivially derivable from RAGTAG, and the substantive content (bug-bias diagnosis + filter rationale) lives in §6 RQ2 and §7 RQ3. Writing it twice would be bloat.

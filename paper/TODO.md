# Paper TODO

Items the next session should pick up. Live state in [SESSION_HANDOFF.md](SESSION_HANDOFF.md).

All NRP campaigns are done; all 3-epoch FT cells and the k=12/15 extension cells are in `results/issues11k/`. The §5 evaluation prose is drafted; the paper is now blocked on §1/§2/§6/§7/§8/§9 drafts and the statistical-significance numbers.

## Inline `% TODO` blocks remaining in §5

- [ ] **§5.1: motivate the $k$-sweep.** 1–2 sentences before "We evaluate \votag…" in [`05_evaluations.tex`](sections/05_evaluations.tex#L19) explaining why we scan $k$=1..30 (only knob in VOTAG; plateau anchors the RAGTAG $k$-grid; range chosen to show diminishing returns).
- [ ] **§5.1: write the RQ1 → §5.2 transition.** [`05_evaluations.tex`](sections/05_evaluations.tex#L35) — position \votag's ~0.60 macro $F_1$ as the retrieval-only floor LLM-based methods must clear; flag bug-bias as recurring across approaches.
- [ ] **§5.3: re-audit bootstrap CI methodology.** [`05_evaluations.tex`](sections/05_evaluations.tex#L93) — current paragraph cites paired bootstrap 95% CIs on the (\bragtag − \ragtag) macro $F_1$ difference for §5.3. Author note flags re-checking the methodology and footnote wording, and considering Wilcoxon/McNemar as supplementary tests. Stats produced by [`scripts/paper/significance_bragtag.py`](../scripts/paper/significance_bragtag.py).

## Statistics (paper-blocking)

- [ ] **Bootstrap 95% CIs on macro $F_1$** for every method × model × setting cell shown in [`tables/method_comparison.tex`](tables/method_comparison.tex). 1,000 paired resamples on the 3,300-issue test set. Report alongside point estimates in §5.5.
- [ ] **McNemar's test** for the §5.5 headline pairwise comparisons (same matrix as the table): \bragtag-PS vs Fine-Tune-PA per model; \ragtag-PS vs \bragtag-PS per model.
- [ ] **Implementation:** new [`scripts/paper/significance_tests.py`](../scripts/paper/significance_tests.py). Use `statsmodels.stats.contingency_tables.mcnemar` and `numpy` resampling. Compute over pooled, rescued predictions (consistent with §5.5 methodology). Document in §4 Setup (1,000 bootstraps; McNemar's with continuity correction; p < 0.05 threshold; b vs c counts in supplementary).

## Sections still to draft (in order)

- [ ] **§6 Discussion.** Weave in the actionable insights captured below.
- [ ] **§7 Threats to Validity.** Multi-seed FT validation gap (3-epoch FT-PA on 14B is unexpectedly higher than on 32B — possible single-seed variance), invalid-rate framing if held over from supervisor question, generalization to non-Qwen LLMs, dataset scope (11 OSS projects).
- [ ] **§8 Conclusion.**
- [ ] **§1 Introduction.**
- [ ] **§2 Related Work.** Stub already cites Heo, Aracena, Colavito, Izadi, Trautsch, etc.

## Discussion — insights to weave into §6

Concrete §5 findings that need explicit treatment in §6 (actionable, not just descriptive).

- [ ] **Targeted retrieval can match large-scale fine-tuning on minority classes.** §5.5 [`tables/method_comparison.tex`](tables/method_comparison.tex) shows \bragtag\ delivers the best question $F_1$ on three of four models while fine-tune wins on bug/feature. Plausible cause: question is a minority label within IRC, so few-shot quality matters more than training-set volume on that class. **§6 framing:** when an IRC deployment has a rare/project-specific label with little labeled data, retrieval-augmented few-shot with bias correction may match or beat LoRA fine-tuning at a fraction of the cost. **Future-work hook:** map the fine-tune-vs-context trade-off curve in low-data / novel-label regimes (synthetic-rare-label sweep, or transfer to a new project with a custom label space).
- [ ] **Fine-tune is cheaper than retrieval at large model size.** §5.5 cost analysis: at Qwen-32B, fine-tune total time (2.71 h) is below \ragtag/\bragtag (4.15–4.62 h) because RAGTAG inference at $k$=12 processes ~20M prompt tokens, exceeding fine-tune's training+inference token volume. Implication: at sufficiently large models with long best-$k$, the conventional "fine-tuning is more expensive" framing inverts.
- [ ] **\bragtag is the most class-balanced approach.** Lowest per-class $F_1$ std (0.053) and highest worst-class floor (0.694) averaged across the four models. Practical implication for skewed-class deployments.

## Forward-pointers / cross-section consistency

- [ ] **§3.3 RAGTAG → §5.3 BRAGTAG signpost.** Insert in §3.3:
  > *We additionally introduce a retrieval-debiasing intervention applied on top of \ragtag; we describe its algorithm and present its empirical motivation in \cref{sec:bragtag}.*

## Fine-tune memory mitigations (§4 + §7)

- [ ] **Document the FT memory-saving knobs in §4 and reference in §7 Threats.** Our LoRA fine-tune uses Unsloth's gradient checkpointing ([`fixed_fine-tune.py:354`](../fixed_fine-tune.py#L354): `use_gradient_checkpointing="unsloth"`) and gradient accumulation ([`fixed_fine-tune.py:398`](../fixed_fine-tune.py#L398): `gradient_accumulation_steps=16`). The FT peak GPU RAM numbers reported in §5.5 [`tables/method_comparison.tex`](tables/method_comparison.tex) (5.5/9.9/16.7/31.5 GB for 3B/7B/14B/32B) are post-mitigation; without these knobs the peaks would be higher. **§4 (Setup):** add a sentence in the fine-tuning setup paragraph noting both mitigations are enabled. **§7 (Threats):** flag that absolute peak FT memory is implementation-dependent — different memory-optimization choices would shift the FT/RAG memory ratio reported in §5.5, though the ordering (FT > RAG/BRAG) is robust to these choices because activations + gradients + optimizer state are unique to training.

## Hardware specs (§4.5)

- [ ] **§4.5 placeholder.** Local 4090 specs known. Need NRP node specs for L40 / L40S used for 14B/32B FT and 32B retrieval. Run `kubectl --context nautilus -n bgsu-cs-heydarnoori describe node <node-name>` (node names are recoverable from pod history). Capture: GPU model + VRAM, CPU model + core count, RAM, OS/CUDA versions.

## Held-over supervisor question

- [ ] **§5.4/§5.5 invalid-rate / \votag-rescue framing.** Three open questions: (a) is "the model learns the output format during training" the right causal claim for the FT invalid-rate drop, or do we need a stricter analysis? (b) is the \votag-rescue methodology sound for cross-method comparison, or does it bias toward \ragtag/\bragtag (which share the same retrieval index)? (c) should rescue-vs-no-rescue numbers be reported side-by-side, or only the rescued ones?

## Conventions / methodology (reference, do not edit)

- [x] **Pooled aggregation for all paper metrics** (decided 2026-05-06; methodology paragraph in [`04_setup.tex`](sections/04_setup.tex) §"Evaluation Metrics"). Concat-then-evaluate on the 3,300-issue test set (PA and PS alike). Per-project mean is not used for headline numbers.
- [x] **\votag-rescue is §5.5-only.** Cross-method best-config comparison rescues invalid LLM outputs with \votag; all other §5 numbers are raw.
- [x] **Paper-figure scripts go in [`scripts/paper/`](../scripts/paper/)** with pooled aggregation baked in. Existing `scripts/analysis/*.py` use legacy per-project mean for PS — do not retrofit, do not consume their PS outputs for paper artifacts.

## Done / closed (kept for audit)

- [x] All NRP campaigns (waves w1–w7) finished and integrated into `results/issues11k/`.
- [x] k=12/15 extension at ctx=8192 — complete for all four Qwen sizes (3B/7B/14B local, 32B NRP). Best-$k$ ladder confirmed in §5.2/§5.3.
- [x] DeBERTa-v3-large PA fine-tune — exploratory experiment failed (mode-collapse on `bug`, macro $F_1$ = 0.167). Dropped from the paper; outputs preserved for audit.
- [x] §5.1 RQ1 \votag — drafted (3 paragraphs + 2-panel figure), 2 inline TODOs remain.
- [x] §5.2 \ragtag — drafted (6 paragraphs + kcurve fig + perclass fig).
- [x] §5.3 Balanced \ragtag — drafted (4 paragraphs + kcurve fig + perclass fig + bragtag_results table); 1 author TODO on bootstrap CI re-audit.
- [x] §5.4 Fine-Tuning — drafted (4 paragraphs + finetune_comparison fig).
- [x] §5.5 Method Comparison — drafted (7 paragraphs incl. cost analysis, method_comparison table, method_pareto fig).

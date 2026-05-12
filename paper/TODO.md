# Paper TODO

Items the next session should pick up. Live state in [SESSION_HANDOFF.md](SESSION_HANDOFF.md).

The §5 evaluation prose is fully drafted with significance numbers in §5.3 and §5.5. The paper is now blocked on §1/§2/§6/§7/§8/§9. **Next session focus: §6 Discussion. Read the headline narrative arc entry below first — it is the locked-in framing for the paper.**

## Discussion — insights to weave into §6

Concrete §5 findings that need explicit treatment in §6 (actionable, not just descriptive). The first entry is the paper-level narrative arc — read it before any §6 drafting.

- [ ] **Headline narrative arc for §6 (paper's main contribution framing).** Prior IRC work tried fine-tuning but did not explore retrieval-augmented few-shot. Our results show: (1) vanilla \ragtag\ is competitive with fine-tune in macro $F_1$ (within 0.020 in aggregate, statistically tied at Qwen-32B) at $11\times$ less labeled data and lower GPU cost — establishing retrieval-augmented few-shot as a viable, underexplored alternative for IRC; (2) \bragtag's debiasing intervention closes the remaining gap to statistical equivalence with fine-tune, demonstrating that identifying and correcting model classification biases and retrieval biases is a practical lever for further improvement. **Two contributions:** (a) RAG is a cost-efficient alternative to fine-tuning that produces competitive results; (b) the gap to fine-tune can be closed via bias-correction interventions, opening a research direction (more biases, more interventions). **§6 must lead with these two claims so reviewers don't anchor on TOST/debiasing methodology nitpicks at the expense of the underexplored-retrieval-for-IRC point.**
- [ ] **Targeted retrieval can match large-scale fine-tuning on minority classes.** §5.5 [`tables/method_comparison.tex`](tables/method_comparison.tex) shows \bragtag\ delivers the best question $F_1$ on three of four models while fine-tune wins on bug/feature. Plausible cause: question is a minority label within IRC, so few-shot quality matters more than training-set volume on that class. **§6 framing:** when an IRC deployment has a rare/project-specific label with little labeled data, retrieval-augmented few-shot with bias correction may match or beat LoRA fine-tuning at a fraction of the cost. **Future-work hook:** map the fine-tune-vs-context trade-off curve in low-data / novel-label regimes (synthetic-rare-label sweep, or transfer to a new project with a custom label space).
- [ ] **Fine-tune is cheaper than retrieval at large model size.** §5.5 cost analysis: at Qwen-32B, fine-tune total time (2.71 h) is below \ragtag/\bragtag (4.15–4.62 h) because RAGTAG inference at $k$=12 processes ~20M prompt tokens, exceeding fine-tune's training+inference token volume. Implication: at sufficiently large models with long best-$k$, the conventional "fine-tuning is more expensive" framing inverts.
- [ ] **\bragtag is the most class-balanced approach.** Lowest per-class $F_1$ std (0.053) and highest worst-class floor (0.694) averaged across the four models. Practical implication for skewed-class deployments.

## Sections still to draft (in order)

- [ ] **§6 Discussion.** Lead with the headline narrative arc above. Weave in the other three insights as supporting points.
- [ ] **§7 Threats to Validity.** Multi-seed FT validation gap (3-epoch FT-PA on 14B is unexpectedly higher than on 32B — possible single-seed variance), invalid-rate framing if held over from supervisor question, generalization to non-Qwen LLMs, dataset scope (11 OSS projects), FT memory mitigations are implementation-dependent (see entry below). **Also include BRAGTAG margin-selection defense (see dedicated entry below).**
- [ ] **§8 Conclusion.**
- [x] **§1 Introduction.** Drafted end-to-end on 2026-05-11. Polish items and open decisions tracked in the next section below.
- [ ] **§2 Related Work.** Stub already cites Heo, Aracena, Colavito, Izadi, Trautsch, etc. **Concurrent-work additions to absorb:** (a) Dinç & Tüzün, *"Judge the Votes"* (2025) — Bugzilla single-project, binary VALID/INVALID, fixed $k{=}5$ retrieval+few-shot; defense via differentiator stack (different task, single project, no $k$-curve, no model-scale study, no debiasing). (b) LLM-Cure (Assi et al., *ACM TOSEM* 2025) — app-review feature assignment with fixed five-shot prompts; cite as further evidence that retrieval-augmented few-shot is gaining traction in SE-adjacent text classification but does not constitute a systematic IRC evaluation.
- [ ] **§9 Data Availability.** Stub and **ESEM-mandatory** (desk-reject if missing). ESEM 2026 requires the section placed immediately after Conclusions. Cover: dataset source (Heo et al. 2025 11-project benchmark), code repository, reproducibility scripts in `scripts/paper/`, model weights handling (Qwen2.5-Instruct from Hugging Face), retrieval index reproducibility.

## §1 / §5 polish items (loose ends from 2026-05-11 session)

§1 Introduction was drafted end-to-end. The following items remain open before submission.

**Decisions to make:**

- [ ] **Add explicit Contributions paragraph to §1?** §1 currently has findings paragraphs (¶19–¶23) flowing into a concluding "Overall, results indicate…" sentence (¶25). No `\textbf{Contributions}`-style enumeration. If the abstract or §8 Conclusion leans on a "we contribute X, Y, Z" framing, §1 should mirror it; otherwise the findings-driven structure stands as-is.
- [ ] **Re-add 77%-template-misuse finding to §1?** Currently lives only in [`§6.1`](sections/06_discussion.tex). Was in an earlier §1 draft but dropped during restructuring. Add back as a 4th finding paragraph if the abstract's *Results* or *Conclusions* sub-section mentions it; otherwise leave it as a §6-only insight.

**§5 structural work:**

- [ ] **§5 opener listing RQ1–RQ4.** §5 currently begins with `% TODO` at [`05_evaluations.tex`](sections/05_evaluations.tex#L8). After retitling §5.2/§5.3/§5.5 with RQ-style titles and merging §5.4+§5.5 into the RQ4 subsection, the RQs now appear without an enumeration up front. Draft a short opener that lists RQ1–RQ4 and notes that §5.4 (Fine-Tuning) is preparatory for RQ4's two-part answer (Classification Performance + Compute and Data Cost).

## Inline `% TODO` blocks remaining in §5

- [ ] **§5.1: motivate the $k$-sweep.** 1–2 sentences before "We evaluate \votag…" in [`05_evaluations.tex`](sections/05_evaluations.tex#L19) explaining why we scan $k$=1..30 (only knob in VOTAG; plateau anchors the RAGTAG $k$-grid; range chosen to show diminishing returns).
- [ ] **§5.1: write the RQ1 → §5.2 transition.** [`05_evaluations.tex`](sections/05_evaluations.tex#L35) — position \votag's ~0.60 macro $F_1$ as the retrieval-only floor LLM-based methods must clear; flag bug-bias as recurring across approaches.
- [ ] **§5.3: re-audit bootstrap CI methodology.** [`05_evaluations.tex`](sections/05_evaluations.tex#L93) — current paragraph cites paired bootstrap 95% CIs on the (\bragtag − \ragtag) macro $F_1$ difference for §5.3. Author note flags re-checking the methodology and footnote wording, and considering Wilcoxon/McNemar as supplementary tests. Stats produced by [`scripts/paper/significance_bragtag.py`](../scripts/paper/significance_bragtag.py).

## Fine-tune memory mitigations (§4 + §7)

- [ ] **Document the FT memory-saving knobs in §4 and reference in §7 Threats.** Our LoRA fine-tune uses Unsloth's gradient checkpointing ([`fixed_fine-tune.py:354`](../fixed_fine-tune.py#L354): `use_gradient_checkpointing="unsloth"`) and gradient accumulation ([`fixed_fine-tune.py:398`](../fixed_fine-tune.py#L398): `gradient_accumulation_steps=16`). The FT peak GPU RAM numbers reported in §5.5 [`tables/method_comparison.tex`](tables/method_comparison.tex) (5.5/9.9/16.7/31.5 GB for 3B/7B/14B/32B) are post-mitigation; without these knobs the peaks would be higher. **§4 (Setup):** add a sentence in the fine-tuning setup paragraph noting both mitigations are enabled. **§7 (Threats):** flag that absolute peak FT memory is implementation-dependent — different memory-optimization choices would shift the FT/RAG memory ratio reported in §5.5, though the ordering (FT > RAG/BRAG) is robust to these choices because activations + gradients + optimizer state are unique to training.

## BRAGTAG margin-selection defense (§7)

Pre-emptive strike against the "you tuned $m{=}3$ on the same test data you evaluate on" reviewer attack. Two text-level moves; no new experiments required.

- [ ] **Move 1 — Reframe the $m$ selection signal as a retrieval-prior, not a test-$F_1$-posterior.** The 79%/41% fire-rate trade-off used to pick $m$ at [`05_evaluations.tex:78`](sections/05_evaluations.tex#L78) depends only on the retrieval index's neighbor-label distribution and the test set's ground-truth labels — *not* on any model's predictions. Add one sentence to §5.3 (or §7) making this explicit:
  > "The fire-rate trade-off used to select $m$ depends only on the retrieval index's neighbor-label distribution, not on any model's predictions; test-set labels enter the selection signal, but no test-set classifier outcome does."
- [ ] **Move 2 — Make the Qwen-3B → larger-models transfer explicit.** Edit [`05_evaluations.tex:78`](sections/05_evaluations.tex#L78) to clarify $m{=}3$ was locked on Qwen-3B and held fixed across Qwen-7B/14B/32B *without* retuning. This converts three of four BRAGTAG cells into transfer results, narrowing the selection-bias attack to one cell. Suggested wording:
  > "We select $m{=}3$ empirically on Qwen-3B and hold it fixed across Qwen-7B, Qwen-14B, and Qwen-32B without retuning. The BRAGTAG gains reported for those three models are therefore transfer results from a Qwen-3B-tuned hyperparameter."
- [ ] **§7 mention (optional, single sentence).** Acknowledge that a fully held-out validation split was not used for $m$ selection; cite the Qwen-3B → 7B/14B/32B transfer as the mitigation. Do *not* expand into a long confession — that signposts the weakness.

## Hardware specs (§4.5)

- [ ] **§4.5 placeholder.** Local 4090 specs known. Need NRP node specs for L40 / L40S used for 14B/32B FT and 32B retrieval. Run `kubectl --context nautilus -n bgsu-cs-heydarnoori describe node <node-name>` (node names are recoverable from pod history). Capture: GPU model + VRAM, CPU model + core count, RAM, OS/CUDA versions.

## Industry-anchor for §6.2 *Wrong templates cause misclassifications* (added 2026-05-10)

- [ ] **GitHub Copilot for issue creation already implements the workflow we propose** (public preview since **May 19, 2025**). Copilot drafts title, body, labels, assignees, issue type from a natural-language description and routes the draft into the repository's preferred template. This is exactly the "*classify-then-template*" inversion in our §6.2 paragraph. Action: add 1–2 sentences and a footnote to §6.2 citing this as live industry validation, and re-frame our contribution as the *methodological grounding* (RAG-based classification + bias correction + the 77% template-confound finding) for a workflow that is already shipping but not empirically evaluated in the literature.
  - [GitHub Changelog: Creating issues with Copilot on github.com is in public preview](https://github.blog/changelog/2025-05-19-creating-issues-with-copilot-on-github-com-is-in-public-preview/)
  - [GitHub Docs: Using GitHub Copilot to create or update issues](https://docs.github.com/en/copilot/how-tos/copilot-on-github/copilot-for-github-tasks/use-copilot-to-create-or-update-issues)
  - [GitHub Docs: Triaging an issue with AI](https://docs.github.com/en/issues/tracking-your-work-with-issues/administering-issues/triaging-an-issue-with-ai)
  - [GitHub Blog: Building AI-powered GitHub issue triage with the Copilot SDK](https://github.blog/ai-and-ml/github-copilot/building-ai-powered-github-issue-triage-with-the-copilot-sdk/)
  - [GitHub Blog: Continuous AI for accessibility (2026)](https://github.blog/ai-and-ml/github-copilot/continuous-ai-for-accessibility-how-github-transforms-feedback-into-inclusion/)
  - [GitLab Duo Security Analyst Agent (vulnerability triage; Nov 2025 beta, Jan 2026 GA)](https://about.gitlab.com/blog/vulnerability-triage-made-simple-with-gitlab-security-analyst-agent/) — adjacent (security-focused) but same agentic-triage paradigm.
  - [InfoQ: GitLab 18.8 Marks General Availability of the Duo Agent Platform (Jan 2026)](https://www.infoq.com/news/2026/01/gitlab-18-8-duo-agent-platform/)


## Held-over supervisor question

- [ ] **§5.4/§5.5 invalid-rate / \votag-rescue framing.** Three open questions: (a) is "the model learns the output format during training" the right causal claim for the FT invalid-rate drop, or do we need a stricter analysis? (b) is the \votag-rescue methodology sound for cross-method comparison, or does it bias toward \ragtag/\bragtag (which share the same retrieval index)? (c) should rescue-vs-no-rescue numbers be reported side-by-side, or only the rescued ones?

## Optional follow-ups (not paper-blocking)

- [ ] **Extend significance to a full method × model × setting matrix.** §5.3 and §5.5 headline comparisons are covered by [`scripts/paper/significance_bragtag.py`](../scripts/paper/significance_bragtag.py) and [`scripts/paper/significance_method_comparison.py`](../scripts/paper/significance_method_comparison.py). A broader matrix (\ragtag-PA vs FT-PA, \ragtag-PS vs \bragtag-PS, etc.) would only matter if a reviewer asks.
- [ ] **method_pareto figure cleanup.** [`paper/figures/method_pareto.{pdf,png}`](figures/) and [`scripts/paper/fig_method_pareto.py`](../scripts/paper/fig_method_pareto.py) are orphaned (superseded by `cost_analysis`). Delete or keep as supplementary — author's call.

## Conventions / methodology (reference, do not edit)

- [x] **Pooled aggregation for all paper metrics** (decided 2026-05-06; methodology paragraph in [`04_setup.tex`](sections/04_setup.tex) §"Evaluation Metrics"). Concat-then-evaluate on the 3,300-issue test set (PA and PS alike). Per-project mean is not used for headline numbers.
- [x] **\votag-rescue is §5.5-only.** Cross-method best-config comparison rescues invalid LLM outputs with \votag (\votag-PS for \ragtag/\bragtag, \votag-PA for fine-tune); all other §5 numbers are raw.
- [x] **TOST equivalence margin: $\delta=0.01$.** Aggregate \bragtag-vs-fine-tune CI [-0.007, +0.008] fits inside ±0.01.
- [x] **§5.5 narrative arc.** \ragtag-as-competitive leads, \bragtag-closes-gap supports. See SESSION_HANDOFF.md §6 "Decisions logged".
- [x] **Paper-figure scripts go in [`scripts/paper/`](../scripts/paper/)** with pooled aggregation baked in. Existing `scripts/analysis/*.py` use legacy per-project mean for PS — do not retrofit, do not consume their PS outputs for paper artifacts.

## Done / closed (kept for audit)

- [x] All NRP campaigns (waves w1–w7) finished and integrated into `results/issues11k/`.
- [x] k=12/15 extension at ctx=8192 — complete for all four Qwen sizes (3B/7B/14B local, 32B NRP). Best-$k$ ladder confirmed in §5.2/§5.3.
- [x] DeBERTa-v3-large PA fine-tune — exploratory experiment failed (mode-collapse on `bug`, macro $F_1$ = 0.167). Dropped from the paper; outputs preserved for audit.
- [x] §5.1 RQ1 \votag — drafted (3 paragraphs + 2-panel figure), 2 inline TODOs remain.
- [x] §5.2 \ragtag — drafted (6 paragraphs + kcurve fig + perclass fig).
- [x] §5.3 Balanced \ragtag — drafted (4 paragraphs + kcurve fig + perclass fig + bragtag_results table); 1 author TODO on bootstrap CI re-audit.
- [x] §5.4 Fine-Tuning — drafted (4 paragraphs + finetune_comparison fig).
- [x] §5.5 Method Comparison — drafted (cost analysis paragraphs, method_comparison table, cost_analysis figure). Significance numbers in place: paired bootstrap CIs + McNemar + TOST for both \ragtag-vs-FT and \bragtag-vs-FT.
- [x] §5.5 statistical-significance scripts: [`scripts/paper/significance_bragtag.py`](../scripts/paper/significance_bragtag.py) covers §5.3 (\bragtag vs \ragtag); [`scripts/paper/significance_method_comparison.py`](../scripts/paper/significance_method_comparison.py) covers §5.5 (both methods vs fine-tune, includes TOST + aggregate).

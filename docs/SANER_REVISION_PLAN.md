# SANER Revision Plan (post-ESEM 2026 rejection)

**Target:** SANER (submission ~September 2026). IEEE conference format (two-column, ~10–11 pages + refs) — the LIPIcs 17-page draft must be reformatted and will roughly fit after trimming.

**Review outcome:** A = Accept (novice), B = Weak Reject (some familiarity), C = Weak Reject (expert). The expert review (C) is the one that killed the paper; its three points are the acceptance-critical items.

---

## P0 — Acceptance-critical (Reviewer C, expert)

### 1. Add encoder-based baselines: SetFit + a BERT-family fine-tune
- **Key fact:** the SetFit paper reviewer C links (`10.1016/j.infsof.2025.107758`) is Colavito et al., *Benchmarking LLMs for automated labeling* (IST 2025) — already cited as `colavito2025benchmarking` for the zero-shot baseline. We cite the paper but ignored its SetFit baseline; the reviewer noticed.
- **Action:** run **SetFit** (sentence-transformers, e.g. `all-mpnet-base-v2` or same `all-MiniLM-L6-v2` used for retrieval — nice symmetry: same embedding space, contrastively tuned head) on the 11k benchmark, both PS and PA. SetFit trains in minutes; cost is trivial.
- **Action:** run a **RoBERTa-base fine-tune** (CatIss-style, `izadi2022catiss`) as the classic encoder baseline, PS + PA. Note: our earlier DeBERTa-v3-large attempt mode-collapsed (predicts only `bug`, macro F1 = 0.167, preserved in `results/issues11k/agnostic/microsoft_deberta-v3-large/`). Diagnose (likely LR too high for large encoder / insufficient warmup) rather than reuse; RoBERTa-base with standard HF Trainer settings is the established recipe and much less collapse-prone. `run_transformer_ft.py` already emits cost_metrics in the right schema.
- **Reporting:** add both to the RQ4 comparison table (macro/per-class F1, GPU memory, train+infer time, and token/cost columns from item 8). Whatever the outcome, the paper's efficiency argument must be made **relative to these cheaper baselines**, not only vs LoRA:
  - If RAGTAG beats SetFit clearly → the "is the LLM worth it?" question gets a positive answer and the contribution strengthens.
  - If SetFit is close → reframe honestly: RAGTAG's advantage is no-training deployment + incremental index updates + model-swap robustness, and quantify the F1 premium the LLM buys.
- New RQ or fold into RQ4: "How do retrieval-augmented LLM methods compare against lightweight encoder baselines (SetFit, RoBERTa) and LoRA fine-tuning?"

### 2. Justify / analyze similarity-based neighbor selection (topic vs. label alignment)
- Reviewer C's core technical objection: cosine similarity retrieves by **topic**, not **label**; a feature request and a bug about the same module are near-neighbors with different labels.
- **Action — label-alignment analysis (no GPU, uses existing `neighbors_k*.csv`):** compute per-class *label homophily* — fraction of top-k neighbors sharing the query's true label, per class and per k. Expected result: bug/feature neighbors are label-aligned, question neighbors are not (we already know mean N_bug − N_question ≈ −1.36 for true questions). This turns BRAGTAG from an ad-hoc fix into a *measured response to quantified label-topic misalignment* — a much stronger arc: RQ2 finds the weakness → homophily analysis explains it → BRAGTAG exploits the measured distribution.
- **Action — contextualize question→bug confusion in prior IRC literature** instead of presenting it as a novel observation. Kallis et al. (TicketTagger), Izadi et al. (CatIss), Colavito et al. all document question as the hardest class / question-bug confusion. Add 3–5 sentences in RQ2 + related work explicitly stating this is a known phenomenon that persists under retrieval-augmented LLMs, and that our contribution is measuring *why* (retrieval label misalignment) and mitigating it.
- **Action — discuss label-aware retrieval** as a design axis: e.g., embeddings fine-tuned with label supervision (SetFit-style contrastive), stratified per-label retrieval, or re-ranking by label-discriminative features. Small experiment if time permits (e.g., retrieve top-k per label = stratified prompt); otherwise a substantive discussion paragraph + future work.

### 3. Reframe the title (and headline claim) around the problem, not the method comparison
- Current: *"Can Retrieval-Augmented Few-Shot Prompting Match LoRA Fine-Tuning for Issue Report Classification? An Empirical Study"*.
- Candidates (problem-first, contribution-forward):
  - "Cost-Efficient Issue Report Classification with Retrieval-Augmented In-Context Learning"
  - "Training-Free Issue Report Classification: How Far Can Retrieval-Augmented LLMs Go?"
  - "Retrieval-Augmented Issue Report Classification: A Cost-Effectiveness Study Against Fine-Tuned LLMs and Encoder Baselines"
- Also soften the headline "RAGTAG beats fine-tuning by 0.021–0.040" — reviewer C calls the margin marginal. Lead instead with the **cost-for-performance trade-off** (equivalent F1 at 33% less GPU memory, no retraining, 11× less data) which is the durable claim, and with the *lessons* (retrieval saturation, label misalignment, balanced-retrieval fix) reviewer B asked to emphasize.

---

## P1 — Methodological rigor (Reviewer B)

### 4. Restructure as an empirical study with Wohlin/Jedlitschka-style design reporting
- Stop calling it a "case study" (abstract Methods paragraph, conclusion). Frame as an **empirical study / technology comparison experiment** in the MSR paradigm.
- Add an *Experimental Design* subsection in Setup: **independent variables** (method: VOTAG/RAGTAG/BRAGTAG/SetFit/RoBERTa/LoRA-FT; model scale: 3B/7B/14B/32B; data scope: PS/PA; k), **dependent variables** (macro & per-class F1, precision, recall, accuracy, invalid rate, peak GPU memory, GPU time, tokens per classification), **fixed factors** (embedding model, prompt template, temperature 0.1, quantization, max seq len), **potential confounds** (prompt-format sensitivity, dataset label noise ~9% per our own failure analysis, template-induced bias, quantization). Cite Wohlin et al. *Experimentation in Software Engineering* and/or Jedlitschka & Pfahl reporting guidelines.

### 5. Systematic inferential statistics
- Use the **same protocol in every RQ**: paired bootstrap 95% CIs on F1 differences everywhere (RQ1–RQ4), not just RQ3/RQ4. State **H0/H1 explicitly** for each tested comparison.
- **Justify TOST δ per Lakens (2017)** as a smallest-effect-size-of-interest argument, e.g.: δ = 0.01 macro F1 ≈ ~33 label flips on the 3,300-issue test set ≈ noise floor observed across repeated runs / below the inter-run variance of the LLM itself. Currently δ = 0.01 and δ = 0.02 both appear unexplained — pick and defend.
- Add a **statistics summary table**: comparison, H0, test, point estimate, CI, verdict — one row per claim (reviewer B explicitly asked; also fixes their "RQ4 should be a table" comment).
- There's a standing author-TODO in `05_evaluations.tex` (line 77) to re-audit the bootstrap CI methodology — do it as part of this.

### 6. Metric triangulation in RQ1–RQ3
- Report accuracy + macro precision + macro recall alongside macro F1 (cheap: `evaluate.py` already computes per-label P/R; just surface them). Balanced 3-class dataset means accuracy is meaningful. Add to figures/tables or a compact appendix-style table per RQ.

### 7. Studied-projects description table
- One table: project, domain, primary language, ~stars, issue volume, and the 600-issue/200-per-class sample. Half a page, kills the "no description of the projects" objection and supports the generalizability discussion.

### 8. Presentation fixes (Reviewer B minor)
- **[17] FAISS:** cite the definitive Johnson, Douze & Jégou 2019, *IEEE Trans. Big Data* 7(3):535–547 (`billion-scale similarity search with GPUs`) — optionally alongside `douze2025faiss`; at minimum add volume/issue.
- **[26] arXiv preprint now in ICSE 2026 proceedings:** identify which entry is [26] in the compiled bbl (likely `akhavan2026linkanchor` or `assi2026llm`) and update to the official proceedings.
- Clearer BRAGTAG-vs-others presentation (ties into the stats summary table).

---

## P2 — Practical-cost & robustness analyses (Reviewer A, cheap and strengthens vs C too)

### 9. Token / cost-per-classification analysis
- Reviewer A: GPU memory isn't the real-world cost of RAG — **tokens are**. Add per-method: mean prompt + completion tokens per classification, and cost per 1,000 classifications at representative API prices (plus local GPU-hour framing). RAGTAG at k=12 pushes ~8k-token prompts vs LoRA-FT's ~2k — be honest that RAG loses on inference tokens.
- Add an **amortization / break-even analysis**: FT pays a fixed training cost + cheap inference; RAG pays zero training + expensive inference. Compute the break-even number of classified issues per model size (and note re-training frequency shifts the break-even in RAG's favor). This turns a weakness into the paper's most practitioner-useful figure.
- No GPU needed: re-tokenize stored prompts/preds offline.

### 10. Retrieval-miss analysis ("what if there's no good match?")
- Reviewer A: what happens when the index has no similar issue? Using existing neighbors CSVs: bin test queries by top-1 (or mean top-k) similarity, plot accuracy per bin for VOTAG/RAGTAG/BRAGTAG.
- If accuracy degrades at low similarity → propose/discuss a similarity-threshold guard (fall back to zero-shot below threshold) and, if cheap, evaluate it. If it doesn't degrade much → that's a nice robustness result to report. Either way, add a paragraph to Discussion on cold-start (empty/young index) deployments.

### 11. Tighten Results, expand Implications
- Reviewer A: shorten result narration (RQ2/RQ4 prose repeats numbers already in figures/tables), and add a concise **Implications for Research and Practice** discussion. Reviewer B: emphasize **model-agnostic lessons** that survive model churn: (i) retrieval-only voting bounds useful k; (ii) nearest-neighbor label misalignment on ambiguous classes is the failure mode, not retrieval quality; (iii) class-conditional retrieval balancing is a training-free bias correction; (iv) few-shot gains grow with model scale but saturate; (v) fallback-to-retrieval eliminates invalid outputs for free. These become the enduring-contribution paragraph.

---

## P3 — Mechanical

- Reformat LIPIcs → IEEE conference class (IEEEtran), re-check page budget; SANER is typically 10–12 pages + refs, double-blind.
- Update abstract to the new framing (drop "case study", add encoder baselines, lead with cost-effectiveness).
- Data availability: replace `\cite{replication_package}` TODO with the actual Figshare link (reviewer B verified the artifact — keep it in sync with new SetFit/RoBERTa runs).
- Fill remaining TODOs (`main.tex` author metadata for camera-ready branch, `09_data_availability.tex`).

---

## Suggested execution order

1. **Experiments first (long pole, but small):** SetFit PS+PA, RoBERTa PS+PA (fix collapse), optional stratified-retrieval probe. All fit on the local 4090; est. days, not weeks.
2. **Offline analyses:** label homophily, similarity-bin robustness, token accounting, extra metrics — all from existing CSVs.
3. **Rewrite pass:** title/abstract/intro reframe, Wohlin design subsection, stats protocol + summary table, projects table, related-work additions (question-bug confusion literature, SetFit positioning), implications section.
4. **Reformat to IEEE + trim.**

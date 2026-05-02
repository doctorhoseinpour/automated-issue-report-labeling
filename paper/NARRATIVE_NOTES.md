# Narrative notes — for intro writing and method-section reminders

Reference doc for narrative beats, contribution claims, and the scientific
reasoning behind our methodological choices. Use when:
- Writing the introduction (primary purpose).
- Adding a brief reminder sentence in the methodology if a reviewer would
  otherwise wonder "why is this the way it is?"

Not prose. Points only.

---

## 1. Contribution framing

### Gap claim

- **"Nobody has tried it for IRC with these labels."** Clean gap claim.
- Verified manually against `aracena2025applying`, `heo2025study`,
  `koyuncu2025exploring`, `colavito2024leveraging`. None of them did
  RAGTAG-style retrieval-augmented few-shot for IRC with the
  bug/feature/question label space.

### Systematic-study framing (over "best config")

- The contribution is **systematic variation across the dimensions we vary**,
  not "we found the optimal config."
- We **vary**: k ∈ {0, 1, 3, 6, 9}, model size (3B, 7B, 14B, 32B), setting
  (project-agnostic vs project-specific).
- We **hold fixed**: context length (8192), embedder, prompt template,
  voting schemes, decoding parameters.
- Avoid the absolute claim ("best config"). A reviewer will ask "did you
  try k=15? a larger embedder?"
- Defensible framing: *"We systematically vary k and model size to
  characterize when retrieval-augmented prompting matches fine-tuning."*

---

## 2. Methodological story (the honest "how we got here")

- **"We thought of VOTAG because we're researchers, not retrofitting."**
  Reviewers respect papers that explain methodological choices rather
  than presenting them as inevitable.
- The actual mental map:
  1. Saw fine-tuning works for IRC but is computationally expensive.
  2. Saw retrieval-augmented few-shot works for adjacent SE tasks
     (`huang2025back`).
  3. Knew zero-shot is weak for IRC (`aracena2025applying`,
     `wang2021well`).
  4. Hypothesized that retrieval-based example selection would help,
     based on prior in-context learning work (`liu2022makes`) and the
     k-NN principle that similar instances share labels
     (`cover1967nearest`).
  5. Recognized that RAGTAG mixes retrieval and LLM contributions, so
     a retrieval-only baseline is needed to decompose them. That is
     VOTAG.
  6. Realized VOTAG has additional value as a cost-efficient floor and
     as a design tool for choosing the RAGTAG k grid.

---

## 3. The three roles of VOTAG (the linchpin)

VOTAG is **independently defensible on three axes**. Stacking the three
justifies its own subsection rather than a footnote.

1. **Ablation control.** Isolates how much of RAGTAG's accuracy is
   attributable to retrieval alone vs. the LLM.
2. **Cost-efficient floor.** Runs on CPU, uses no model at inference;
   any LLM-based approach must clear it to justify its computational
   cost.
3. **Design tool.** Its accuracy curve over k informs the k grid we
   evaluate for RAGTAG. Reported as RQ1.

---

## 4. Reasoning X — academic backing for retrieved few-shot

The single-sentence Reasoning X:

> *In-context learning is empirically sensitive to demonstration
> relevance (`liu2022makes`); k-NN's foundational principle is that
> semantically similar instances share labels (`cover1967nearest`,
> Dudani 1976); retrieval over a labeled corpus exploits both at once.*

### Citations carrying the weight

- **`liu2022makes`** — Liu et al. 2022, *"What Makes Good In-Context
  Examples for GPT-3?"* Empirically shows retrieval-based example
  selection beats random across multiple tasks. **Direct backing.**
- **`cover1967nearest`** — Cover & Hart 1967, *"Nearest neighbor pattern
  classification."* Foundational k-NN result, asymptotically
  Bayes-optimal. **Theoretical backing.**
- **Dudani 1976** — *"The distance-weighted k-nearest-neighbor rule."*
  Source for similarity-weighted voting (used by VOTAG). Not yet in bib;
  add when describing VOTAG voting schemes in §3.3.

### Bonus citations

- **`huang2025back`** — adjacent SE-task validation
  (issue→commit traceability).
- **`khandelwal2019generalization`** — establishes the kNN-augmented
  inference pattern more broadly. Optional.

### Minimum to carry the argument

Two citations: **`liu2022makes`** + one of the k-NN papers
(`cover1967nearest`). The rest is bonus.

---

## 5. Calibration warnings (Option A discipline)

We use **Option A** throughout the paper: claim only what our
experiments measure.

- We **do** measure: per-issue inference cost, one-shot training cost,
  classification accuracy, GPU memory.
- We **do not** measure: cross-project transfer, label-distribution
  drift over time, deployment dynamics, retrieval-index maintenance
  cost at scale.
- Avoid framing fine-tuning's cost in deployment terms. Stick to
  training compute.
- Symmetric scope discipline applies to RAGTAG and VOTAG too. If you
  raise unmeasured concerns about FT in §8 Discussion, raise the
  parallel concerns about retrieval-based methods in the same paragraph.
- The **single Discussion paragraph** (§8) is where unmeasured concerns
  are flagged as future work, symmetrically across all three approaches.
  Not in intro. Not in methods.

---

## 6. Phrases I (Claude) should not import into your prose

These are AI-flavored phrasings from prior failed drafts. Watch for them
when reviewing.

- "Fine-tuning is the dominant strategy in this line of work, and it is
  also its dominant cost."
- "the resulting weights are bound to the task"
- "The reasoning rests on a long-standing observation"
- "VOTAG plays three roles in this paper" (the *idea* is yours; the
  summary sentence is mine)
- "the floor any LLM-based approach must clear to justify its
  computational cost"
- "A careful reader will ask which part is doing the work"

When writing the intro, don't anchor on these. Write each beat in your
own register.

---

## 7. Implications for paper structure

(To be filled in / referenced from existing PAPER_NARRATIVE.md as the
narrative settles. Current structure: §3 retrieval → VOTAG → RAGTAG →
fine-tuning → debiasing; §5 RQ1 = VOTAG plateau analysis informs
RAGTAG k grid; §7 RQ3 = debiasing as gap-closer; random-shot ablation
in evaluations only.)

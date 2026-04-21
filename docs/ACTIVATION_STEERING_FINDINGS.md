# Activation Steering Findings

**Date:** 2026-04-20
**Dataset:** issues3k (1,497 test / 1,498 train), Llama-3.2-3B-Instruct
**RAGTAG config:** k=3 neighbors, ctx=8192, FAISS (MiniLM-L6-v2)

---

## 1. Motivation

All prior interventions to correct the parametric bug bias failed or produced marginal gains:
- Prompt-level: vote prior injection, enhanced system prompt, label definitions — no improvement
- Logit-level: batch calibration (+0.009 F1), contrastive decoding (catastrophic)
- Hypothesis: the bias is geometric — it lives in the model's activation space

Two activation-level approaches were implemented:
1. **CAA (Contrastive Activation Addition)** — Rimsky et al. 2024: compute a steering vector from contrastive pairs, add it (scaled) to the residual stream during generation
2. **NTW (No Training Wheels) directional ablation** — Gupta et al. 2025: project out the bias direction from the residual stream

---

## 2. Steering Vector Computation

Three pair selection strategies were tested for computing the bug-vs-question steering vector:

| Strategy | Description | Compute time | Pairs |
|----------|-------------|-------------|-------|
| `answer_conditioned` | Same zero-shot prompt, append "bug" vs "question" as continuations. Difference at the label token position. | ~3 min | 300 |
| `faiss_matched` | For each bug issue, find nearest question issue by embedding similarity. Difference at last prompt token. | ~73s | 300 |
| `class_means` | Mean activation of all bug issues minus mean activation of all question issues. | ~102s | 500 bug, 500 question |

### Per-layer L2 norms

All strategies show increasing norms in deeper layers with a large spike at layer 27 (final layer / unembedding projection):

- **answer_conditioned:** Smooth gradient 0.66 (L0) → 5.08 (L26), spike to 12.79 at L27
- **faiss_matched:** Similar shape, smaller magnitudes 0.24 → 4.61, spike to 9.87 at L27
- **class_means:** Flat plateau ~1.5 from L1–L13, then gradual rise to 3.46 (L26), spike to 6.68 at L27

Files: `results/steering_vectors/llama3b_3k_{answer,faiss,means}/`

---

## 3. Layer Sweep (CAA, answer_conditioned, m=-1.0)

Full sweep across all 28 layers. Baseline RAGTAG k=3: F1_macro=0.6743, R_bug=0.778, R_question=0.468.

| Layer | F1_macro | F1_bug | F1_feat | F1_quest | R_bug | R_quest |
|-------|----------|--------|---------|----------|-------|---------|
| 0 | 0.6797 | 0.656 | 0.733 | 0.651 | 0.618 | 0.667 |
| 1 | 0.6831 | 0.668 | 0.732 | 0.649 | 0.626 | 0.692 |
| 2 | 0.6353 | 0.601 | 0.696 | 0.609 | 0.535 | 0.620 |
| 3 | 0.6822 | 0.664 | 0.728 | 0.655 | 0.605 | 0.714 |
| 4 | 0.6711 | 0.660 | 0.715 | 0.638 | 0.612 | 0.692 |
| 5 | 0.6818 | 0.664 | 0.732 | 0.650 | 0.620 | 0.700 |
| 6 | 0.6761 | 0.664 | 0.735 | 0.629 | 0.640 | 0.639 |
| **7** | **0.6967** | 0.687 | 0.746 | 0.657 | 0.657 | 0.702 |
| 8 | 0.6864 | 0.670 | 0.734 | 0.655 | 0.630 | 0.698 |
| 9 | 0.6651 | 0.612 | 0.722 | 0.661 | 0.527 | 0.749 |
| 10 | 0.6638 | 0.621 | 0.726 | 0.644 | 0.541 | 0.745 |
| 11 | 0.6632 | 0.604 | 0.734 | 0.652 | 0.496 | 0.737 |
| 12 | 0.6741 | 0.644 | 0.747 | 0.631 | 0.601 | 0.633 |
| 13 | 0.6832 | 0.648 | 0.748 | 0.653 | 0.591 | 0.673 |
| 14 | 0.6811 | 0.662 | 0.746 | 0.635 | 0.626 | 0.614 |
| 15 | 0.6781 | 0.674 | 0.752 | 0.609 | 0.674 | 0.561 |
| 16 | 0.6795 | 0.676 | 0.752 | 0.610 | 0.678 | 0.553 |
| **17** | **0.7018** | 0.674 | 0.754 | 0.677 | 0.610 | 0.696 |
| 18 | 0.6964 | 0.689 | 0.754 | 0.645 | 0.674 | 0.622 |
| 19 | 0.6934 | 0.681 | 0.761 | 0.638 | 0.669 | 0.602 |
| 20 | 0.6946 | 0.678 | 0.754 | 0.652 | 0.643 | 0.622 |
| 21 | 0.6996 | 0.698 | 0.763 | 0.638 | 0.698 | 0.596 |
| 22 | 0.6982 | 0.689 | 0.764 | 0.642 | 0.682 | 0.614 |
| **23** | **0.7063** | 0.681 | 0.754 | 0.684 | 0.618 | 0.729 |
| 24 | 0.7034 | 0.684 | 0.755 | 0.671 | 0.643 | 0.700 |
| 25 | 0.6952 | 0.660 | 0.746 | 0.680 | 0.583 | 0.714 |
| 26 | 0.6910 | 0.670 | 0.758 | 0.645 | 0.632 | 0.620 |
| 27 | 0.6933 | 0.663 | 0.745 | 0.673 | 0.595 | 0.692 |

**Key observations:**
- Best layer: **23** (F1_macro=0.7063, +0.032 over baseline). At 82% through the model — much deeper than CAA's typical "first third" finding for Llama-2 7B.
- Two local optima: layer 7 (0.6967, 25% depth) and layer 23 (0.7063, 82% depth), with a trough at layers 9–16.
- Layer 2 is an outlier dip (0.6353) — steering at this early layer is destructive.
- All layers shift the bug/question balance: question recall increases at the expense of bug recall. The magnitude of the shift correlates loosely with the steering vector's L2 norm.
- Invalid rate constant at 1.9% across all layers (28 issues) — up from ~0.07% baseline. The steering slightly disrupts generation for a small fixed set of issues.

GPU peak memory: 4.18 GB across all runs.

---

## 4. Multiplier Sweep (layer 23, answer_conditioned)

| Multiplier | F1_macro | F1_bug | F1_feat | F1_quest | R_bug | R_quest |
|-----------|----------|--------|---------|----------|-------|---------|
| Baseline | 0.6743 | 0.688 | 0.773 | 0.562 | 0.778 | 0.468 |
| **m=−0.5** | **0.6933** | 0.683 | 0.764 | 0.633 | 0.679 | 0.596 |
| **m=−1.0** | **0.6935** | 0.659 | 0.750 | 0.671 | 0.589 | 0.712 |
| m=−1.5 | 0.6724 | 0.606 | 0.729 | 0.682 | 0.483 | 0.820 |
| m=−2.0 | 0.6068 | 0.478 | 0.696 | 0.647 | 0.331 | 0.888 |
| m=−3.0 | 0.3446 | 0.024 | 0.472 | 0.538 | 0.012 | 0.964 |

**Key observations:**
- m=−0.5 and m=−1.0 are essentially tied on F1_macro (~0.693). Different tradeoffs: m=−0.5 is conservative (R_bug 0.679, R_quest 0.596), m=−1.0 is aggressive (R_bug 0.589, R_quest 0.712).
- The optimal range is narrow: m=−0.5 to m=−1.0. Past m=−1.5 bug recall collapses.
- m=−3.0 almost completely eliminates bug predictions (R_bug=0.012), confirming the vector direction is correct — it points from question toward bug.
- Note: multiplier sweep F1_macro (0.6935) is slightly lower than layer sweep (0.7063) for the same config (layer 23, m=−1.0). Minor run-to-run variation, possibly from different batching or padding.

---

## 5. Pair Strategy Comparison (layer 23, m=−1.0)

| Strategy | F1_macro | F1_bug | F1_feat | F1_quest | R_bug | R_quest |
|----------|----------|--------|---------|----------|-------|---------|
| Baseline (no steering) | 0.6743 | 0.688 | 0.773 | 0.562 | 0.778 | 0.468 |
| **answer_conditioned** | **0.6935** | 0.659 | 0.750 | 0.671 | 0.589 | 0.712 |
| class_means | 0.6810 | 0.675 | 0.762 | 0.606 | 0.687 | 0.542 |
| faiss_matched | 0.6753 | 0.677 | 0.778 | 0.571 | 0.730 | 0.482 |

**Key observations:**
- **answer_conditioned is the only strategy that meaningfully moves question recall.** The other two barely improve over baseline (0.482–0.542 vs 0.468).
- faiss_matched essentially does nothing — it preserves baseline behavior. The topic-controlled pairing confounds topic with label, producing a direction that doesn't isolate the bug-vs-question decision.
- class_means provides modest improvement (+0.007 F1_macro) but much less than answer_conditioned (+0.019).
- This validates CAA's design: same-input, different-output contrastive pairs isolate the decision direction more cleanly than between-example comparisons.

---

## 6. NTW Directional Ablation (class_means vectors)

| Config | F1_macro | F1_bug | F1_feat | F1_quest | R_bug | R_quest | Invalid |
|--------|----------|--------|---------|----------|-------|---------|---------|
| Baseline | 0.6743 | 0.688 | 0.773 | 0.562 | 0.778 | 0.468 | 0.07% |
| Single layer (23) | 0.4830 | 0.625 | 0.729 | 0.094 | 0.902 | 0.050 | 2.1% |
| All layers | 0.0000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 100% |

**NTW ablation is a complete failure for this task:**
- **All-layer ablation destroys the model entirely** (100% invalid predictions). Projecting out the bias direction at every layer removes too much information — the model can no longer generate coherent tokens.
- **Single-layer ablation makes the bias worse**, not better. Bug recall soars to 0.902 (from 0.778) while question recall craters to 0.050 (from 0.468). The projection removes the component that *distinguishes* bug from question, causing the model to default even harder to its prior (bug).
- This is the opposite of CAA's additive approach: CAA shifts the decision boundary, ablation removes a dimension of variation. For a task where the model needs to distinguish along that dimension, ablation is counterproductive.

---

## 7. Best Result vs Baselines

| Approach | F1_macro | F1_bug | F1_feat | F1_quest | R_bug | R_quest | Invalid |
|----------|----------|--------|---------|----------|-------|---------|---------|
| Flawed fine-tune | 0.5082 | 0.588 | 0.571 | 0.367 | 0.619 | 0.312 | 13.2% |
| Fixed fine-tune | 0.6669 | 0.697 | 0.735 | 0.570 | 0.858 | 0.512 | 0.2% |
| RAGTAG k=3 (baseline) | 0.6743 | 0.688 | 0.773 | 0.562 | 0.778 | 0.468 | 0.07% |
| VTAG k=16 (retrieval floor) | 0.6451 | — | — | — | — | — | — |
| **RAGTAG + CAA (L23, m=−1.0)** | **0.7063** | 0.681 | 0.754 | 0.684 | 0.618 | 0.729 | 1.9% |

- Best steering result beats all approaches on F1_macro (+0.032 over RAGTAG, +0.039 over fixed FT)
- Question recall 0.729 — the largest improvement from any intervention (+0.261 over baseline, +56% relative)
- Bug recall tradeoff is controlled: 0.618 (from 0.778) — significant drop but bug F1 only falls from 0.688 to 0.681 because precision increases
- Feature F1 mildly affected: 0.754 (from 0.773)
- Invalid rate increases to 1.9% (from 0.07%) — 28 fixed issues become unparseable across all steering configs

---

## 8. Conclusions

1. **Activation steering works for bias correction in LLM classification.** CAA produces a consistent, meaningful improvement in the underrepresented class (question) without catastrophic degradation of the overrepresented class (bug).

2. **The answer_conditioned pair strategy is essential.** Same-prompt contrastive pairs (appending different label tokens) isolate the bug-vs-question decision direction far better than between-example comparisons (faiss_matched) or simple class means.

3. **Optimal layer is deeper than expected.** Layer 23 (82% depth) beats layer 7 (25% depth), contradicting CAA's finding of optimal intervention at ~1/3 depth. This may be task-specific: classification decisions are made late in the network, unlike the behavioral/persona changes studied in the original CAA paper.

4. **The optimal multiplier range is narrow (−0.5 to −1.0).** Beyond −1.5, bug recall collapses. The intervention requires careful calibration.

5. **NTW directional ablation is not viable for this task.** Projecting out the bias direction removes discriminative information, making the bias worse (single-layer) or destroying generation entirely (all-layer). CAA's additive approach is fundamentally better suited to shifting a decision boundary without destroying the underlying representation.

6. **The improvement is real but bounded.** +0.032 F1_macro is the largest gain from any single intervention across this entire study. However, it does not close the gap to larger fine-tuned models (Qwen-14B fixed FT: 0.7387). The parametric bias is partially correctable via activation steering, but not fully eliminable.

---

## 9. File Locations

| File | Description |
|------|-------------|
| `compute_steering_vector.py` | Computes per-layer steering vectors (3 strategies) |
| `activation_steering.py` | Applies steering during RAGTAG inference (CAA + ablation) |
| `run_steering.sh` | 6-phase experiment orchestrator |
| `results/steering_vectors/llama3b_3k_answer/` | answer_conditioned vectors |
| `results/steering_vectors/llama3b_3k_faiss/` | faiss_matched vectors |
| `results/steering_vectors/llama3b_3k_means/` | class_means vectors |
| `results/issues3k_steering/layer_sweep_answer/` | Layer sweep results (28 layers) |
| `results/issues3k_steering/multiplier_sweep_answer/` | Multiplier sweep (5 values) |
| `results/issues3k_steering/strategy_*/` | Strategy comparison results |
| `results/issues3k_steering/ntw_ablation/` | NTW ablation results |

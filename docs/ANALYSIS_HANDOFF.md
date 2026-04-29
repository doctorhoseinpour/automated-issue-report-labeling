# Analysis-Session Handoff

**Date:** 2026-04-29
**Purpose:** Bring a fresh Claude Code session up to speed on the 11k-projects paper so it can propose and execute a comprehensive results analysis without the noise of the long migration / runs conversation.

---

## Bootstrap prompt for the new session

Paste this verbatim into a fresh Claude Code session opened at `~/llm-labler`:

```
Read CLAUDE.md, then docs/ANALYSIS_HANDOFF.md. The experimental
campaign is complete. Survey the data in results/issues11k/, propose
a comprehensive analysis plan covering whatever angles you think
matter for the paper, then execute it once I approve. Don't anchor
to the patterns already documented — those are starting facts, not
boundaries.
```

---

## 1. Status

- The experimental campaign is **complete**. No more model runs are planned.
- Final coverage: **4 models × 4 LLM-based approaches × 11 projects × 2 settings**, plus **VTAG** as a no-LLM voting-based retrieval baseline that is model-independent and runs once per setting per k.
- NRP cluster is in clean steady-state (no jobs, no pods). PVCs (`hf-cache-pvc`, `results-pvc`) are preserved with model weights and synced results, but no workloads are running.
- Llama-3B and Llama-8B data from earlier experiments is preserved on disk under `results/issues11k/.../unsloth_Llama_*`. It is **not** part of the active paper lineup. Treat as historical reference. The user explicitly chose to switch to a Qwen-only family for clean cross-scale comparison without a model-family confound.

## 2. Active model lineup

Four Qwen2.5-Instruct bnb-4bit models, uniform quantization across the family for a clean parameter-size-only scale axis:

| Model | HF ID |
|---|---|
| Qwen-3B | `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` |
| Qwen-7B | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| Qwen-14B | `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` |
| Qwen-32B | `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` |

## 3. Approaches under comparison

VTAG is a first-class baseline alongside the LLM-based approaches — we built it (`vtag.py`) to establish the pure-retrieval floor any LLM result must clear to justify the LLM. Treat it as a co-equal entry in the leaderboard, not just a sanity check.

| Approach | Description |
|---|---|
| **Zero-shot** | No retrieval, no training. The model sees only the issue. |
| **VTAG** (no LLM) | Voting-based k-NN over the same FAISS index RAGTAG uses. Voting schemes available: `similarity` (Dudani-weighted, paper default), `shepard` (sim²), `majority`. Zero GPU at inference, no LLM call. Model-independent — one VTAG number per setting per k applies to all four models. |
| **Debiased VTAG (margin = 3)** | Same bug-class debias logic as RAGTAG-debias, applied to VTAG voting. Added 2026-04-29 as a mechanism ablation. Both agnostic and project-specific. See section 5.4b for results. |
| **RAGTAG** | FAISS retrieval + few-shot prompt with `<label>X</label>` XML tags. k ∈ {1, 3, 6, 9}; k = 0 is treated as zero-shot. |
| **Debiased RAGTAG (margin = 3)** | Retrieval-time class-rebalancing intervention. Project-specific only by design. k ∈ {1, 3, 6, 9}. |
| **Fine-tuning** | LoRA via Unsloth on the same train split that RAGTAG retrieves from. Both agnostic (3,300 examples) and project-specific (300 examples per project). |

## 4. Files the new session should read first (in order)

1. `CLAUDE.md` — project overview, three RQs, conventions, model list (already updated to the Qwen-only lineup as of 2026-04-29).
2. `docs/ANALYSIS_HANDOFF.md` — this document.
3. `docs/11K_BENCHMARK_FINDINGS.md` — Section 0 dated 2026-04-28 has a master table with **the original 4 models** (Llama-3B, Llama-8B, Qwen-14B, Qwen-32B). It does **not** yet include Qwen-3B / Qwen-7B. Those models' eval CSVs exist on disk and need to be folded in.
4. `docs/PAPER_NARRATIVE.md` — Apr 25 framing doc. Predates the Qwen-3B/7B addition and the Llama drop. Likely needs refresh.
5. `docs/NRP_MIGRATION_STATUS.md` — historical record of the cluster migration. Background only.
6. `results/issues11k/` — raw eval CSVs.

## 5. What's been established so far

These are **starting facts to verify and extend**, not the boundary of what's worth analyzing.

### 5.1 Per-model best-config leaderboard (f1_macro)

| Model | zero-shot | RAGTAG | debias_m3 | FT | Winner |
|---|---:|---:|---:|---:|---|
| Qwen-3B | 0.613 ag | 0.694 ag k3 | 0.709 ps k6 | 0.652 ag | debias k6 |
| Qwen-7B | 0.662 ag | 0.714 ps k6 | 0.730 ps k6 | 0.741 ag | FT agnostic |
| Qwen-14B | 0.645 ag | 0.717 ag k9 | 0.742 ps k9 | 0.715 ag | debias k9 |
| Qwen-32B | 0.688 ag | 0.759 ag k9 | 0.775 ps k9 | 0.746 ag | debias k9 |

`ag` = agnostic, `ps` = project-specific.

### 5.2 Pattern: debiased RAGTAG vs plain RAGTAG (proj-spec avg, by k)

| Model | k=1 | k=3 | k=6 | k=9 |
|---|---:|---:|---:|---:|
| Qwen-3B | −0.009 | −0.001 | +0.031 | +0.031 |
| Qwen-7B | −0.010 | −0.002 | +0.016 | +0.020 |
| Qwen-14B | −0.006 | −0.009 | +0.019 | +0.025 |
| Qwen-32B | −0.009 | −0.010 | +0.012 | +0.017 |

Observation: hurts (slightly) at low k, helps clearly at k ≥ 6, with the largest wins at k = 9 on every model. Invalid-output rates roughly halve under debias at high k.

### 5.3 Pattern: FT vs best debias by scale

| Model | best FT | best debias | FT − bestDeb |
|---|---:|---:|---:|
| Qwen-3B | 0.652 | 0.709 | −0.057 |
| Qwen-7B | 0.741 | 0.730 | +0.011 |
| Qwen-14B | 0.715 | 0.742 | −0.026 |
| Qwen-32B | 0.746 | 0.775 | −0.028 |

Non-monotonic within the family: 7B is the only size where FT wins.

### 5.4 VTAG floor (model-independent)

VTAG numbers are the same regardless of model (no LLM in the loop). Best k for both settings is k = 9.

| Setting | k=1 | k=2 | k=3 | k=6 | k=9 |
|---|---:|---:|---:|---:|---:|
| Agnostic | 0.565 | 0.565 | 0.579 | 0.591 | 0.598 |
| Project-specific (avg) | 0.556 | 0.556 | 0.563 | 0.572 | 0.578 |

Every Qwen approach above zero-shot beats this floor in both settings — even Qwen-3B zero-shot agnostic (0.613) clears VTAG-9 agnostic (0.598). Report this anchor: it justifies the LLM cost and frames how much work the LLM is doing on top of retrieval.

### 5.4b Debiased VTAG — same intervention, no LLM (mechanism ablation)

We ran the same margin-3 bug-class debias on VTAG (drop all bug neighbors when `bug_count - question_count ≤ 3`, fall back to original top-k if that would empty the set). Outputs at `results/issues11k/{agnostic,project_specific/<proj>}/vtag_debias_m3/{predictions,evaluations}/eval_k{1,3,6,9}.csv`. Implementation: `vtag.py --debias_retrieval --debias_margin 3` (added 2026-04-29).

Macro-F1 deltas (debias − plain):

| Setting | k=1 | k=3 | k=6 | k=9 |
|---|---:|---:|---:|---:|
| Agnostic | +0.000 | −0.013 | +0.000 | **+0.006** |
| Project-specific (avg) | +0.000 | −0.011 | −0.000 | **+0.007** |

Tiny aggregate wins at k=9 (+0.006/+0.007), much smaller than RAGTAG-debias gains for any Qwen size (+0.017 to +0.052 at k=9). Per-project: debias-VTAG wins 7/11 at k=9 — mediocre, with `microsoft_vscode` losing 0.094 and `tensorflow_tensorflow`/`flutter_flutter` winning 0.09+.

Per-class shift, agnostic k=9 (smoking gun for the mechanism story):

| class | precision Δ | recall Δ |
|---|---:|---:|
| bug | **+0.098** | **−0.192** |
| feature | −0.030 | +0.048 |
| question | −0.054 | **+0.155** |

Debias-VTAG works exactly as expected: bug recall crashes 19 points (bug evidence is thrown away in many cases), question recall jumps 15 points (question wins by default in borderline cases), bug precision rises (the bugs that survive are very confident). The macro-F1 gain is marginal because the bug-recall crash nearly cancels the question/feature recall gains.

The contrast with RAGTAG-debias matters: an LLM uses the rebalanced few-shots intelligently (it can still predict bug from the issue text itself), so RAGTAG-debias gives clean +0.017–0.052 wins. VTAG can only vote on what's in the bag, so the same intervention trades bug recall for question recall almost 1-for-1. This is a clean ablation isolating the "LLM prior correction" part of the debias story from the "general class-balancing" part.

### 5.5 Pattern: FT proj-spec collapse

| Model | FT agnostic | FT proj-spec | drop |
|---|---:|---:|---:|
| Qwen-3B | 0.652 | 0.575 | −0.077 |
| Qwen-7B | 0.741 | 0.511 | −0.230 |
| Qwen-14B | 0.715 | 0.637 | −0.079 |
| Qwen-32B | 0.746 | 0.713 | −0.033 |

Per-project FT (300 train issues per project) is unstable except at the upper end. Plain RAGTAG and debiased RAGTAG use the same 300 issues at retrieval time without training and remain strong.

## 6. Data layout

```
results/issues11k/
  agnostic/
    <model_tag>/
      ragtag/{predictions, evaluations}/{eval_zero_shot, eval_k{1,3,6,9}}.csv
      finetune_fixed/eval_finetune_fixed.csv     # OR finetune_fixed/evaluations/eval_finetune_fixed.csv
    vtag/{predictions, evaluations}/eval_k*.csv
    vtag_debias_m3/{predictions, evaluations}/eval_k{1,3,6,9}.csv
    neighbors/                                    # FAISS-derived
  project_specific/<11 projects>/
    <model_tag>/
      ragtag/{predictions, evaluations}/...
      ragtag_debias_m3/{predictions, evaluations}/eval_k{1,3,6,9}.csv
      finetune_fixed/...                          # path varies as above
    vtag/{predictions, evaluations}/eval_k*.csv
    vtag_debias_m3/{predictions, evaluations}/eval_k{1,3,6,9}.csv
    neighbors/
```

The 11 projects: `ansible_ansible`, `bitcoin_bitcoin`, `dart-lang_sdk`, `dotnet_roslyn`, `facebook_react`, `flutter_flutter`, `kubernetes_kubernetes`, `microsoft_TypeScript`, `microsoft_vscode`, `opencv_opencv`, `tensorflow_tensorflow`.

Active model_tags:
- `unsloth_Qwen2_5_3B_Instruct_bnb_4bit`
- `unsloth_Qwen2_5_7B_Instruct_bnb_4bit`
- `unsloth_Qwen2_5_14B_Instruct_bnb_4bit`
- `unsloth_Qwen2_5_32B_Instruct_bnb_4bit`

Llama tags also exist on disk but are not part of the active lineup:
- `unsloth_Llama_3_2_3B_Instruct`
- `unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit`

For the Qwen-32B RAGTAG and debias runs there are `_v2` suffixed directories from a clean rerun on NRP A6000 hardware. Use those `_v2` paths for Qwen-32B RAGTAG / debias data — the originals were OOM-corrupted during a prior 4090 run and are kept only for the historical record.

Eval CSV schema (shared across approaches):
```
model, top_k, total_issues, invalid_count, invalid_rate, accuracy,
precision_bug, recall_bug, f1_bug, support_bug,
precision_feature, recall_feature, f1_feature, support_feature,
precision_question, recall_question, f1_question, support_question,
precision_macro, recall_macro, f1_macro,
precision_weighted, recall_weighted, f1_weighted
```

Some prediction directories also contain `cost_metrics.csv` files with token counts and wall-clock times.

## 7. What the user has noticed (observation, not directive)

In conversation the user flagged two things they found compelling. Captured here for context. The new session should not treat these as scope-setting — confirm, extend, complicate, or reframe as the data warrants:

- A bug-class over-prediction pattern in plain RAGTAG and zero-shot that the debias intervention appears to mitigate.
- At large scale (Qwen-14B, Qwen-32B), the best zero-training configuration uses 300 project-specific retrieval examples and beats fine-tuning trained on 3,300 agnostic issues, despite an ~11× difference in example count.

## 8. Output destinations and conventions

- Update `docs/11K_BENCHMARK_FINDINGS.md` with the new analyses. The Section 0 master table will need an update — either drop Llama and present a clean 4-Qwen story, or extend to 6 models. The new session should choose with the user.
- Refresh `docs/PAPER_NARRATIVE.md` if findings warrant.
- `results/` is **paper-archival**. Read-only — never delete, move, or rename.
- Any time analysis excludes model-load time.
- "Log findings" = write into `docs/`, not memory.
- The Qwen FT runs include `cost_metrics.csv` files for cost / latency framing if useful.

## 9. NRP / cluster state (background)

Already covered in `docs/NRP_MIGRATION_STATUS.md`. Short version: campaign finished 2026-04-28, mega-runner Job and Pod deleted, outbox emptied, both PVCs bound and preserved. No active workloads. The NRP config files (`scripts/nrp/plan.yaml`, `scripts/nrp/submit.py`, `scripts/nrp/manifests/job-warm-cache.yaml`) still reference Llama tags from before the lineup swap; they are inert until another remote run is queued.

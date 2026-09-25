# New-LLM study: the decision-state read-out vs a validation-tuned RAG baseline on three new open models

Living lab notebook, started 2026-09-24. **For a new paper**, separate from the SANER 2027 submission. It builds on the rag_next study (`docs/RAG_NEXT_STUDY.md`), where a linear read-out of Qwen2.5's zero-shot answer-position state beat SetFit-PS and every RAG configuration (R-32B 0.8445).

- **Code:** `scripts/experiments/newllms/` (local clone and lab machine; OSC copy under `~/nm/repo`)
- **Outputs:** `results/issues11k/exploration/newllms/` (lab machine only)
- Existing `results/` data, rag_next outputs (read-only inputs here), the pipeline scripts, `SANER2027/` and `paper/` were not modified.

---

## 0. Questions

1. Does the rag_next read-out, carried over **unchanged**, work on three new open model families?
   - `Qwen/Qwen3.5-9B` (hybrid Gated-DeltaNet/attention)
   - `google/gemma-4-12B-it`
   - `mistralai/Ministral-3-8B-Instruct-2512`
2. How does it compare with a **RAG baseline whose K is tuned on validation** for the same model? The zero-shot model (K = 0) is the floor.

The encoder baselines (SetFit etc.) already exist and are reused from `results/`.

## 1. Protocol (fixed before any run)

| Item | Choice |
|---|---|
| Test set | The paper's 3,300-issue test split. Pooled macro F1 over all 3,300 via `evaluate.py`. |
| Validation for K | **val495** (`make_val.py`): the newest **15** issues of each (project, label) group of the rag_next dev split. That split is itself the newest 30 train issues per group. So val495 is 495 issues, 15% of train, covering all 33 cells, and never touches test. Retrieval for validation queries uses *inner* only (the older 70 train issues per group, same project): the paper's temporal train → test design, one step earlier. |
| Why 495 | It covers every project × class cell and is scored under all 9 K values on the **same** issues. The K comparisons are therefore paired: a paired bootstrap gives how often each K wins and the CI of each gap. It costs half the full dev split, and dev tracked test in rag_next. |
| Models | Official bf16 weights quantized on load to **4-bit NF4** (bitsandbytes, double quant, bf16 compute), the same scheme as the paper's Unsloth bnb-4bit Qwen2.5 checkpoints. Loaded with plain transformers 5.17.0 (SDPA, batch 1), not Unsloth. Unsloth lacks full support for these architectures, and its patched base model had the causal-mask pitfall of rag_next §4.3. Context 8,192, as in the paper. |
| Prompt | Byte-identical RAGTAG prompt: `llm_labeler.build_chat_messages` with proportional truncation to 8192 − 50 − 20 tokens. It is rendered with each model's chat template with thinking off (`enable_thinking=False`), followed by the `<label>` prefill. It is tokenized without extra special tokens, because the Gemma and Ministral templates already contain BOS. |
| RAG baseline | **RAGTAG-PS**: the paper's MiniLM neighbors (rank-exact with the paper on test), K demonstrations, greedy decoding, max 50 new tokens with a stop at `</label>`, and the paper's `parse_label`; invalid outputs count as wrong. PS only: in the paper, PA and PS RAGTAG are within 0.01 at every k for all Qwen sizes. The paper sampled at T = 0.1; greedy is its deterministic equivalent. The first-step log-probs of the three label tokens are also recorded, giving the constrained-decoding variant from the same run. |
| Read-out (R) | **Frozen rag_next recipe**, not re-tuned. Answer-position states at depth fractions {0.625, 0.75, 0.875, 1.0}: layers 20/24/28/32 of 32 (Qwen3.5-9B), 30/36/42/48 of 48 (Gemma 4 12B), 21/26/30/34 of 34 (Ministral 3 8B). Per-layer AUG logistic regression with C = 0.003 on standardized states, probabilities averaged. Fit on the 3,300 train labels. |
| kNN vote (K) | Frozen: final-layer state standardized with the index's unlabeled statistics, cosine similarity, PS index, k = 9, similarity-weighted vote. Training-free. |
| Hardware | OSC Ascend A100 (`preemptible-nextgen`, 1 GPU per job), with sharded, checkpointed jobs. CPU fits and evaluation on the lab machine. |

## 2. Pre-registration (written before any validation or test result)

**K selection rule (per model).** K* = argmax of val495 macro F1 under the paper's protocol (parsed generation, invalid = wrong), over K ∈ {1, 3, 5, 7, 9, 10, 12, 15}. Ties go to the smaller K. K = 0 is reported separately as the zero-shot baseline.

**Test evaluations (18, all logged in `results/issues11k/exploration/newllms/test_eval_log.csv`):**

| Per model (× 3) | What | Role |
|---|---|---|
| R-<model> | decision-state read-out, frozen recipe | primary |
| K-<model> | decision-state kNN vote, frozen recipe | primary (training-free) |
| ZS-<model> | RAGTAG K = 0, generation | primary baseline |
| RAG-<model> | RAGTAG-PS at K*, generation | primary baseline |
| ZSc-<model>, RAGc-<model> | the same two runs, constrained decoding | descriptive |

Nothing on test is used for any choice.

**Comparisons.** Each comparison is a paired issue-level bootstrap with 2,000 resamples.
- R and K against the same model's RAG@K* and ZS.
- Against SetFit-PS (0.810), FT-PA-14B (0.785), BRAGTAG-PS-32B (0.781) and RAGTAG-PS-32B (0.767).
- Against rag_next's Qwen2.5 R-32B (0.8445) and K-32B (0.799).

Also reported: per-class F1, q→bug, invalid rate, and per-project macro F1 vs SetFit-PS.

**Hypotheses.**
- H1: R beats RAG@K* for every model.
- H2: R is at or above SetFit-PS for at least one new model.
- H3: K (training-free) beats RAG@K* for every model.

## 3. Environment and smoke tests

**Stack.** OSC Ascend, `NVIDIA A100-PCIE-40GB` with driver 580.95.05.
- torch 2.10.0+cu128, transformers 5.17.0, bitsandbytes 0.50.2, flash-linear-attention 0.5.2.
- Python 3.12.14, uv-managed, in `~/nm/venv-nm` on OSC.

The venv lives in `$HOME`, not `/fs/ess`: the shared `/fs/ess/PCS0289` fileset is at its 200k-inode limit (196,584 used). The weights stay in `/fs/ess/PCS0289/rag_next/hf_cache`. The OSC scripts are in `osc/`.

**Smoke test (`smoke.py`, 10 validation issues, one per project; all hard checks pass for all three models):**

| | Qwen3.5-9B | Gemma 4 12B | Ministral 3 8B |
|---|---|---|---|
| Loaded weights (NF4) | 7.3 GB | 7.2 GB | 5.8 GB |
| Tokenizer, parity with `tokenizer.json` | verbatim `TokenizersBackend`† | `GemmaTokenizer`, OK | verbatim, OK |
| BOS in prompt | none (none expected) | exactly 1 | exactly 1 |
| Prompt tail | `<think>\n\n</think>\n\n<label>` | `<\|channel>thought\n<channel\|><label>` | `[/INST]<label>` |
| Label words after `<label>` | 1 token each, distinct | same | same |
| Causality (Δ log-prob when tokens are appended) | 0.13 labels / 0.25 top-10, same argmax | 0.13 / 0.16, same argmax | 0.00 / 0.00 |
| lm_head(norm(h_last)) vs logits | exact | exact (bf16 softcap) | exact |
| Generation first step vs states pass | identical | identical | identical |
| Parse rate, K=0 / K=15 | 100% / 100% | 100% / 100% | 100% / 100% |
| s per prompt: states K=0 / gen K=0 / gen K=15 (~6.1K tok) | 0.08‡ / — / 0.75–1.0‡ | 0.35 / 0.65 / 1.92 | 0.17 / 0.30 / 0.79 |
| Peak GPU memory | 8.7 GB | 10.9 GB | 7.8 GB |

† `AutoTokenizer` maps Qwen3.5 to `Qwen2Tokenizer`, which rebuilds a Qwen2 pre-tokenizer (`\p{L}+` rather than `[\p{L}\p{M}]+`). That changes how combining marks such as ⚠️ are split. The runner therefore checks every tokenizer against `tokenizer.json` and falls back to the verbatim backend.
‡ Steady state. On first use in each job, fla's Triton kernels autotune for about 12 s per new shape.

The causality gaps of 0.13–0.25 nats are two to four bf16 steps of the logits, with an unchanged argmax. Qwen's chunked linear attention and Gemma's 48 layers amplify rounding. The Unsloth masking bug in rag_next §4.3 moved logits by 1.3–10 nats.

**Fixes found before production.** A read-only review of the transformers v5.17 source and the first smoke round found these:
1. The Qwen3.5 tokenizer mismatch (above).
2. `generate(stop_strings=...)` rebuilds its matcher with a full-vocabulary scan on every call. That cost 0.9–2.3 s per prompt, so the runner now builds the matcher once.
3. The capture hooks now copy only the last position. A view of it kept every layer's full activation alive.
4. Jobs submitted from Cardinal inherited `CC=icc`, which does not exist on Ascend and broke Triton's compilation of the fla kernels. The job script now sets `CC=/usr/bin/gcc`.

None of these changes the prompt or the method.

## 4. Validation: RAGTAG-PS K curves and K* (val495; `select_k.py`)

All 9 K values were scored on the same 495 validation issues. The index is *inner* (PS). Scoring is macro F1 under the paper's protocol: parsed greedy generation, invalid outputs counted as wrong.

| K | 0 | 1 | 3 | 5 | 7 | 9 | 10 | 12 | 15 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | 0.713 | 0.732 | 0.762 | 0.763 | 0.776 | **0.786** | 0.784 | 0.780 | 0.776 |
| Gemma 4 12B | 0.736 | 0.758 | 0.777 | 0.778 | 0.796 | 0.795 | 0.794 | **0.806** | 0.793 |
| Ministral 3 8B | 0.656 | 0.683 | 0.698 | 0.720 | 0.719 | 0.731 | 0.735 | 0.734 | **0.741** |
| Mean prompt tokens (Qwen / Gemma / Ministral) | 791 / 885 / 851 | | | | | 5.5K / 5.7K / 5.7K | | | 6.3K / 6.5K / 6.4K |
| Truncated to fit 8,192 (Ministral) | 1% | 3% | 9% | 20% | 30% | 44% | 50% | 59% | 73% |

**Selected K\*** by the pre-registered rule (§2):

| Model | K* | Val F1 at K* | Share of bootstrap resamples where K* wins | Runner-up |
|---|---|---|---|---|
| Qwen3.5-9B | **9** | 0.786 | 48% | K=10, 28% |
| Gemma 4 12B | **12** | 0.806 | 72% | K=9, 10% |
| Ministral 3 8B | **15** | 0.741 | 56% | K=10, 22% |

- **Retrieval helps every model, and the gain is significant.** From K=0 to K\* validation F1 rises by +0.07 to +0.085. The paired bootstrap CI of F1(K\*) − F1(K=0) excludes zero for all three: Qwen [+0.044, +0.106], Gemma [+0.036, +0.104], Ministral [+0.049, +0.120].
- **The curve plateaus from about K=7** for Qwen and Gemma. Every K ≥ 7 is within the bootstrap CI of K\*, so the exact choice costs little. Ministral is still rising at K=15, where 73% of its prompts are truncated to fit 8,192 tokens.
- **The question→bug bias is strongest at zero-shot.** At K=0 the models send 48% (Qwen), 38% (Gemma) and 60% (Ministral) of validation questions to *bug*. At K\* that falls to 24%, 13% and 34%.
- **The constrained variant equals generation almost exactly.** Invalid outputs are 0% for Qwen and Ministral at every K, and 0.4–1.4% for Gemma. The first generated token decides the label.
- **GPU cost of the sweep** (495 × 9 prompts, A100): 47 min (Qwen), 106 min (Gemma), 44 min (Ministral). The jobs ran as 6, 8 and 4 parallel shards.

K\* was fixed before any test run at K\* was launched.

## 5. Test results (18 of 18 pre-registered evaluations, none unplanned)

All results are pooled over the 3,300 test issues and scored by `evaluate.py`; the log is `results/issues11k/exploration/newllms/test_eval_log.csv`. CIs are paired issue-level bootstraps with 2,000 resamples. Differences are the method minus the baseline.

### 5.1 Main table

| Test macro F1 | Qwen3.5-9B | Gemma 4 12B | Ministral 3 8B |
|---|---|---|---|
| ZS: RAGTAG K=0 (generation) | 0.6841 | 0.6963 | 0.6376 |
| RAG@K\*: RAGTAG-PS at the validation-chosen K (generation) | 0.7516 (K=9) | 0.7741 (K=12) | 0.7388 (K=15) |
| **K: decision-state kNN@9 (training-free)** | **0.8002** | 0.7856 | 0.7836 |
| **R: decision-state read-out (frozen recipe)** | **0.8292** | **0.8270** | **0.8266** |
| *Constrained ZS / constrained RAG@K\** (descriptive) | 0.6839 / 0.7506 | 0.6988 / 0.7799 | 0.6375 / 0.7387 |

**Comparisons of R, K and RAG@K\*:**

| Comparison | Qwen3.5-9B | Gemma 4 12B | Ministral 3 8B |
|---|---|---|---|
| RAG@K\* − ZS | +0.068 [+0.056, +0.079] | +0.078 [+0.066, +0.091] | +0.101 [+0.088, +0.115] |
| **R − RAG@K\*** | **+0.078 [+0.065, +0.091]** | **+0.053 [+0.041, +0.064]** | **+0.088 [+0.075, +0.102]** |
| R − ZS | +0.145 [+0.130, +0.160] | +0.131 [+0.116, +0.145] | +0.189 [+0.174, +0.205] |
| K − RAG@K\* | +0.049 [+0.036, +0.061] | +0.012 [−0.000, +0.023] (tie) | +0.045 [+0.031, +0.058] |

**Comparisons with the existing baselines:**

| Comparison | Qwen3.5-9B | Gemma 4 12B | Ministral 3 8B |
|---|---|---|---|
| R − SetFit-PS (0.8100) | +0.019 [+0.006, +0.032] | +0.017 [+0.004, +0.030] | +0.017 [+0.004, +0.029] |
| R − LoRA FT-PA-14B (0.7853) | +0.044 [+0.030, +0.057] | +0.042 [+0.028, +0.055] | +0.041 [+0.027, +0.055] |
| R − BRAGTAG-PS-32B (0.7811) | +0.048 [+0.036, +0.061] | +0.046 [+0.034, +0.058] | +0.046 [+0.033, +0.058] |
| R − Qwen2.5 R-32B (0.8445, rag_next) | −0.015 [−0.024, −0.006] | −0.017 [−0.027, −0.008] | −0.018 [−0.028, −0.009] |
| K − SetFit-PS | −0.010 [−0.023, +0.004] (tie) | −0.024 [−0.037, −0.010] | −0.026 [−0.040, −0.013] |
| RAG@K\* − paper RAGTAG-PS-32B (0.7665) | −0.015 [−0.025, −0.004] | +0.008 [−0.003, +0.019] (tie) | −0.028 [−0.039, −0.017] |

**Per class, question → bug and per project:**

| | Qwen3.5-9B | Gemma 4 12B | Ministral 3 8B |
|---|---|---|---|
| F1 bug / feat / question: R | 0.834 / 0.860 / 0.793 | 0.827 / 0.854 / 0.800 | 0.836 / 0.855 / 0.788 |
| F1 bug / feat / question: RAG@K\* | 0.765 / 0.830 / 0.659 | 0.784 / 0.829 / 0.709 | 0.751 / 0.836 / 0.629 |
| F1 bug / feat / question: ZS | 0.721 / 0.826 / 0.505 | 0.721 / 0.813 / 0.554 | 0.693 / 0.815 / 0.405 |
| q→bug: ZS / RAG@K\* / K / R (SetFit 0.183) | 0.52 / 0.36 / 0.18 / 0.17 | 0.46 / 0.26 / 0.20 / 0.16 | 0.63 / 0.40 / 0.21 / 0.17 |
| Invalid outputs: ZS / RAG@K\* (R, K: 0) | 0.03% / 0.03% | 0.70% / 1.24% | 0.03% / 0.03% |
| Projects where R ≥ SetFit-PS | 9/11 | 8/11 | 9/11 |

### 5.2 Post-hoc robustness (`posthoc_test.py`; descriptive, same predictions, selects nothing)

| | McNemar (right only for method / right only for baseline, p) | Project-cluster bootstrap CI |
|---|---|---|
| R vs RAG@K\* | 357/115, 301/118, 398/134 (all p < 1e-4) | [+0.059, +0.097], [+0.039, +0.072], [+0.063, +0.117] |
| R vs SetFit-PS | 264/199 (p = 0.003), 261/205 (p = 0.011), 251/195 (p = 0.009) | [−0.010, +0.043], [−0.015, +0.046], [−0.005, +0.034] |
| K vs RAG@K\* | 296/151, 241/195 (p = 0.031), 320/198 | [+0.032, +0.069], [−0.007, +0.032], [+0.028, +0.066] |

Values are in model order: Qwen3.5-9B, Gemma 4 12B, Ministral 3 8B.

- **R vs RAG@K\* holds under every analysis.**
- **R vs SetFit-PS holds per issue, not per project.** The issue-level CIs and McNemar favor R for all three models, but resampling projects gives CIs that include zero. This is the same pattern as Qwen2.5 R-14B in rag_next.
- **Per project, R wins most projects.** SetFit keeps `ansible` (0.954 vs R 0.845–0.887, the template-convention project of rag_next §2) and `flutter` (0.855 vs 0.827–0.845). R's biggest wins are on `TypeScript` (+0.07 to +0.09), `opencv` (+0.03 to +0.07) and `react` (+0.03 to +0.06).

### 5.3 Hypotheses (§2)

- **H1: supported for all three models.** R beats RAG@K\* by +0.053 to +0.088.
- **H2: supported for all three at the issue level**, +0.017 to +0.019 over SetFit-PS. It is not robust to project resampling.
- **H3: supported for 2 of 3.** K beats RAG@K\* for Qwen3.5 and Ministral. It ties for Gemma (+0.012, CI touching zero).

### 5.4 Validation → test

| | Qwen3.5-9B | Gemma 4 12B | Ministral 3 8B |
|---|---|---|---|
| RAG@K\*: val495 → test | 0.786 → 0.752 | 0.806 → 0.774 | 0.741 → 0.739 |
| R: dev (990) → test | 0.849 → 0.829 | 0.832 → 0.827 | 0.834 → 0.827 |
| K: dev → test | 0.815 → 0.800 | 0.785 → 0.786 | 0.800 → 0.784 |
| Own label (zero-shot, constrained): dev → test | 0.721 → 0.684 | 0.747 → 0.699 | 0.663 → 0.638 |

Every method drops somewhat from dev to test, the temporal shift of rag_next §7. The generative methods drop the most (about −0.03), and their q→bug rate rises most on test.

## 6. Cost (OSC A100-PCIE-40GB, batch 1)

GPU time is per-item compute summed over shards, excluding model load, as in the paper.

| | Qwen3.5-9B | Gemma 4 12B | Ministral 3 8B |
|---|---|---|---|
| **R / K**: one states pass over all 6,600 issues (train + test) | 13.6 min, 8.5 GB | 24.6 min, 8.3 GB | 12.8 min, 6.7 GB |
| R head: CPU logistic regression (4 layers, test phase) | 150 s | 220 s | 227 s |
| ZS on 3,300 test issues | 15.4 min, 8.7 GB | 26.3 min, 10.8 GB | 13.0 min, 7.7 GB |
| RAG@K\* on 3,300 test issues | 41.8 min, 8.7 GB | 107.2 min, 10.9 GB | 44.9 min, 7.8 GB |
| RAG K sweep on val495 (9 K × 495) | 46.6 min | 106.0 min | 44.5 min |
| Invalid outputs: R, K / RAG@K\* | 0 / 0.03% | 0 / 1.24% | 0 / 0.03% |

- **R's GPU pass costs less than RAG@K\* test inference alone**, even though it processes twice the issues (train + test). It also needs no tuning sweep and no generation.
- **For a new batch of test issues, R costs about half its full pass**, since the train half is done once. That is 7–12 min per 3,300 issues, against 42–107 min for RAG@K\*.
- **The study as a whole took about 8.3 A100-hours of compute**, plus smoke tests and model loading.
- **Wall-clock time:** 31 main-wave shards ran in parallel and finished in about 15 min. The 26 RAG@K\* test shards took about 20 min. There were no preemptions.
- **All GPU work ran on one GPU type**, so the rows compare like with like.

## 7. Verdict

**The read-out transfers across model families without re-tuning.**
- The frozen rag_next recipe (depth fractions, AUG, C = 0.003) gives 0.829 / 0.827 / 0.827 on three architectures it was never tuned on. One is a hybrid linear-attention model (Qwen3.5), one has sliding-window attention with soft-capping (Gemma 4), and one is a Mistral.
- All three beat SetFit-PS at the issue level, LoRA FT-14B, and every RAGTAG/BRAGTAG configuration in the paper. They sit about 0.015–0.018 below Qwen2.5-32B's read-out.

**Against a properly tuned RAG baseline, R wins clearly on every model.**
- RAGTAG-PS with K chosen on validation reaches 0.739–0.774. R adds +0.053 to +0.088, significant in every analysis, including project-cluster resampling.
- Most of the gain is on *question*: F1 0.79–0.80 vs 0.63–0.71. q→bug falls from 26–40% to 16–17%.

**The model matters for generation, not for the read-out.**
- ZS spreads 0.058 across the three models (0.638–0.696), and RAG@K\* spreads 0.035.
- R spreads only 0.003. The same forward pass read by a linear head erases most of the family differences.
- The mechanism of rag_next holds on new models. The model's own constrained answer at the answer position scores 0.64–0.70 on test, and a linear read-out of that same state scores 0.83. Generation, not representation, is the bottleneck.

**Retrieval helps prompting, but it is the weaker use of the labels.**
- Tuned retrieval adds +0.07 to +0.10 over zero-shot.
- The training-free decision-state vote beats tuned RAGTAG for two of three models (+0.045 to +0.049) and ties for Gemma, with no generation.
- The supervised read-out beats both.

**Honest limits.**
- R's lead over SetFit-PS is small, +0.017 to +0.019, and not robust to resampling projects. SetFit still owns `ansible` and `flutter`, and it is a single archival seed.
- Ministral's validation curve is still rising at K=15, where 71–73% of prompts are truncated to the paper's 8,192-token context. A longer context could help its RAG baseline, and was not tried, to keep the paper's protocol.
- These are 4-bit NF4 models loaded with plain transformers, while the Qwen2.5 references used Unsloth with the same NF4 scheme. Cross-family comparisons with rag_next carry that difference.
- Validation used 495 issues, so K choices on the plateau (K ≥ 7) are not separable. §4 shows the plateau is flat to within the bootstrap CI.

## 8. Reproduction

GPU work runs on OSC; CPU fitting and evaluation on the lab machine (`~/llm-labler/scripts/experiments/newllms`, `PY=../../../venv/bin/python`). All paths are under `results/issues11k/exploration/newllms/`.

```bash
# 0. validation split (lab) and inputs relayed to OSC via the local PC
$PY make_val.py
bash osc/sync_nm_to_osc.sh <staging with pool.csv, val495.csv, nb_PS_raw_{dev,test}.npz>
# 1. OSC environment in ~/nm (the /fs/ess fileset is at its inode limit) and weights in /fs/ess
ssh cardinal 'sbatch ~/nm/repo/scripts/experiments/newllms/osc/setup_nm.sbatch'
ssh cardinal 'cd /fs/ess/PCS0289/rag_next && for m in Qwen/Qwen3.5-9B google/gemma-4-12B-it \
    mistralai/Ministral-3-8B-Instruct-2512-BF16; do sbatch ~/nm/repo/scripts/experiments/newllms/osc/prefetch_nm.sbatch $m; done'
# 2. smoke tests (Ascend A100)
ssh cardinal 'sbatch -M ascend --partition=debug-nextgen --time=00:40:00 --job-name=nm-smoke-qw35_9b \
    ~/nm/repo/scripts/experiments/newllms/osc/gpu_nm.sbatch smoke.py --tag qw35_9b'   # same for gm4_12b, mi3_8b
# 3. main wave (states, val sweep, test K=0), then RAG@K* on test; relaunch loop from the local PC
bash osc/watch_nm.sh runs_active.txt      # runs_active.txt = runs_{mi3,gm4,qw35}.txt
bash osc/watch_nm.sh runs_kstar.txt       # written after step 4
# 4. relay, merge, choose K* (lab)
bash osc/pull_nm.sh <local staging>
$PY merge_parts.py <tag> {states,val_sweep,test_k0,test_k<K*>}
$PY select_k.py qw35_9b gm4_12b mi3_8b
# 5. read-out / kNN vote, test CSVs, the 18 test evaluations, summaries
$PY readout.py qw35_9b gm4_12b mi3_8b
$PY make_test_csvs.py <tag> test_k0 ; $PY make_test_csvs.py <tag> test_k<K*>
bash run_test_evals.sh
$PY summarize.py ; $PY posthoc_test.py
```

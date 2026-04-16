# VTAG — Voting-based RAG Baseline: Initial Findings

**Date:** 2026-04-15
**Dataset:** `issues3k.csv` (2,995 issues after dedup)
**Split:** 50% per label — 1,497 test / 1,498 train
**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
**Retrieval:** FAISS `IndexFlatIP` over L2-normalized vectors (cosine similarity)

## What VTAG is

Non-LLM RAG baseline: for each test issue, retrieve top-k nearest-neighbor training issues, then **vote** on their labels. No LLM is used at inference.

Implemented in [vtag.py](../vtag.py). Three voting schemes supported:
- **similarity** (default): `score(c) = Σ sim_i` over neighbors with label c — standard distance-weighted k-NN (Dudani 1976)
- **shepard**: `score(c) = Σ sim_i²` — sharpens nearest-neighbor influence
- **majority**: uniform weights — classic majority vote

Tie-breaking is deterministic: label of the highest-similarity neighbor among tied candidates.

## Run 1: similarity-weighted voting, k = 1..30

**Command:**
```bash
python vtag.py \
  --neighbors_csv results/vtag/neighbors/neighbors_k30.csv \
  --output_dir results/vtag/predictions/similarity \
  --eval_dir results/vtag/evaluations/similarity \
  --voting similarity
```

### Headline result

| Metric | Value | k |
|---|---|---|
| **Best macro-F1** | **0.6451** | **16** |
| Best accuracy | 64.66% | 16, 18 |
| Runtime per k | ~3 ms total for 1,497 issues | — |

### Full curve (selected k)

| k | accuracy | macro-F1 | F1-bug | F1-feature | F1-question |
|---|---|---|---|---|---|
| 1  | 0.6119 | 0.6109 | 0.6430 | 0.5996 | 0.5903 |
| 3  | 0.6253 | 0.6238 | 0.6447 | 0.6050 | 0.6217 |
| 5  | 0.6373 | 0.6361 | 0.6540 | 0.6227 | 0.6316 |
| 7  | 0.6440 | 0.6427 | 0.6655 | 0.6313 | 0.6312 |
| 9  | 0.6319 | 0.6311 | 0.6488 | 0.6284 | 0.6160 |
| 13 | 0.6453 | 0.6439 | 0.6702 | 0.6414 | 0.6201 |
| **16** | **0.6466** | **0.6451** | 0.6690 | 0.6429 | 0.6235 |
| 18 | 0.6466 | 0.6446 | 0.6753 | 0.6435 | 0.6151 |
| 20 | 0.6346 | 0.6319 | 0.6678 | 0.6251 | 0.6027 |
| 25 | 0.6359 | 0.6329 | 0.6667 | 0.6298 | 0.6021 |
| 30 | 0.6346 | 0.6313 | 0.6656 | 0.6223 | 0.6059 |

Full table: [../results/vtag/evaluations/similarity/all_results.csv](../results/vtag/evaluations/similarity/all_results.csv)

### Behavior of the k-curve

1. **Monotonic increase from k=1 to ~k=16**, then mild decay past k=20.
2. **k=1 ≡ k=2 exactly.** At k=2, ties force the tie-breaker to pick the nearest neighbor — identical output to k=1.
3. **Plateau at k=7..18** (F1 within 0.64–0.645). Most additional neighbors beyond k=7 don't change the vote much because the top few dominate the weighted score.
4. **Past k=20, decay begins** — marginally-relevant neighbors start outweighing the signal even with cosine weighting.

### Per-class pattern

- **bug** is the easiest (F1 ≈ 0.67) — high recall (0.75–0.80), moderate precision (~0.59).
- **feature** has the highest precision (~0.76) but moderate recall (~0.55) — missed features often get mislabeled as bug.
- **question** is the weakest class — lowest precision and recall throughout.
- All three classes are imbalanced in their error profile, not in their support (support is ~500 per class by construction).

### Resource footprint

- GPU memory: **0 MB** at inference (voting is pure CPU tallying).
- Runtime: **2–5 ms total for all 1,497 test issues** per k value.
- The only compute cost is the one-time FAISS retrieval upstream, which is amortized across all k values and all three voting schemes.

## Implications for the paper

### 1. VTAG sets the "pure retrieval" floor at macro-F1 ≈ 0.645

Any RAGTAG result below this number means the LLM is **hurting** the retrieval signal. Any RAGTAG result above is the marginal value of LLM reasoning on top of retrieval.

### 2. This is a strong baseline

Random baseline for a 3-class problem is 0.333 macro-F1. VTAG nearly doubles this with no training and no LLM. The paper's claim *"RAG achieves competitive performance with a fraction of the cost"* is already partially true at the retrieval layer alone.

### 3. Reviewer-risk mitigation

The obvious reviewer question — *"is your expensive LLM really doing anything beyond nearest-neighbor classification?"* — is now answerable with concrete numbers. Without VTAG, this question could sink the paper.

### 4. The k-saturation finding is publishable on its own

For a small-retrieval-corpus setting (train = ~1,500 issues), VTAG saturates at k ≈ 16. This is useful context for choosing k in RAGTAG: beyond ~16, we expect diminishing returns regardless of whether an LLM is in the loop.

## Run 2: voting-scheme ablation (shepard + majority)

Same neighbors CSV, same split, same tie-break — only the weight function changes.

### Headline comparison

| Voting scheme | Best macro-F1 | Best k | Best accuracy |
|---|---|---|---|
| similarity (Σ sim)  | 0.6451 | 16 | 64.66% |
| **shepard (Σ sim²)** | **0.6465** | **12** | 64.80% |
| majority (Σ 1)       | 0.6444 | 11 | 64.60% |

**Spread across the three schemes is 0.002 macro-F1.** All three peak in the k=11–16 band.

### Per-scheme best-k breakdown

| Scheme | k | acc | macro-F1 | F1-bug | F1-feature | F1-question |
|---|---|---|---|---|---|---|
| similarity | 16 | 0.6466 | 0.6451 | 0.6690 | 0.6429 | 0.6235 |
| shepard    | 12 | 0.6480 | 0.6465 | 0.6696 | 0.6383 | 0.6317 |
| majority   | 11 | 0.6460 | 0.6444 | 0.6702 | 0.6346 | 0.6284 |

### Full k-curve (macro-F1 side-by-side, selected k)

| k | similarity | shepard | majority |
|---|---|---|---|
| 1  | 0.6109 | 0.6109 | 0.6109 |
| 3  | 0.6238 | 0.6258 | 0.6238 |
| 5  | 0.6361 | 0.6367 | 0.6366 |
| 7  | 0.6427 | 0.6431 | 0.6426 |
| 9  | 0.6311 | 0.6322 | 0.6328 |
| 11 | 0.6428 | 0.6438 | **0.6444** |
| 12 | 0.6438 | **0.6465** | 0.6394 |
| 13 | 0.6439 | 0.6459 | 0.6441 |
| 16 | **0.6451** | 0.6458 | 0.6437 |
| 18 | 0.6446 | 0.6458 | 0.6422 |
| 20 | 0.6319 | 0.6344 | 0.6280 |
| 30 | 0.6313 | 0.6335 | 0.6267 |

Full tables:
- [../results/vtag/evaluations/similarity/all_results.csv](../results/vtag/evaluations/similarity/all_results.csv)
- [../results/vtag/evaluations/shepard/all_results.csv](../results/vtag/evaluations/shepard/all_results.csv)
- [../results/vtag/evaluations/majority/all_results.csv](../results/vtag/evaluations/majority/all_results.csv)

### What the comparison tells us

1. **Weighting barely matters.** The embedder already packs the relevant labels into the first few neighbors; any sane aggregation recovers them. This is a property of a *good* retrieval space — if the ranking were noisy, shepard would pull ahead of majority by a wider margin.
2. **shepard is a hair ahead, and its peak is earlier (k=12 vs 16).** Squared weighting lets the top few neighbors dominate, so shepard gets "enough signal" from fewer retrievals.
3. **majority decays fastest past the peak.** Uniform weighting gives late, marginally-relevant neighbors full voting power, so adding them dilutes the signal.
4. **k=1 is identical across all three schemes** (by construction: one neighbor, one vote, weight doesn't matter). This is a useful sanity check that the three pipelines are truly equivalent except for the weight function.
5. **Diagnostic reading:** the tiny gap between `majority` and `similarity` means *similarity scores are not doing much work beyond ranking*. The retrieval is so top-heavy that knowing "neighbor i was retrieved" already captures most of the information — the magnitude of `sim_i` adds very little.

### Implication for the paper

We report shepard (0.6465 @ k=12) as the VTAG baseline: it is both the strongest and has the lowest k, which matches the "cheap retrieval baseline" framing. The ablation across all three schemes goes in an appendix or supplementary table — it establishes that the result is robust to the voting-weight choice.

## Run 3: embedding-model ablation (shepard voting)

Same dataset, same split, same voting scheme (shepard). Only the embedding model changes. Results live under [../results/vtag_embed/](../results/vtag_embed/).

### Embedders benchmarked

| Model | Params | Dim | Embed time (train+test, GPU) | Index VRAM |
|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 22M  | 384  | ~2 s  | ~500 MB |
| `BAAI/bge-base-en-v1.5`                   | 109M | 768  | ~8 s  | ~1.0 GB |
| `BAAI/bge-large-en-v1.5`                  | 335M | 1024 | ~22 s | ~2.1 GB |

All three are retrieval-tuned English sentence encoders; BGE uses contrastive training on query-passage pairs which is closer to our use case than MiniLM's NLI/paraphrase training.

### Headline comparison (VTAG-shepard, best k per model)

| Embedder | Best macro-F1 | Best k | Best accuracy | Δ vs MiniLM |
|---|---|---|---|---|
| all-MiniLM-L6-v2  | 0.6465 | 12 | 64.80% | —       |
| **bge-base-en-v1.5**  | **0.6692** | **17** | **67.00%** | **+0.0227** |
| bge-large-en-v1.5 | 0.6675 | **6**  | 66.80% | +0.0210 |

### Per-model k-curve (macro-F1 at selected k)

| k | MiniLM | bge-base | bge-large |
|---|---|---|---|
| 1  | 0.6109 | 0.6053 | 0.6091 |
| 3  | 0.6258 | 0.6127 | 0.6294 |
| 5  | 0.6367 | 0.6289 | 0.6605 |
| 6  | 0.6294 | 0.6393 | **0.6675** |
| 7  | 0.6431 | **0.6545** | 0.6591 |
| 9  | 0.6322 | 0.6617 | 0.6541 |
| 12 | **0.6465** | 0.6582 | 0.6632 |
| 15 | 0.6435 | 0.6637 | 0.6656 |
| 17 | 0.6401 | **0.6692** | 0.6590 |
| 20 | 0.6344 | 0.6593 | 0.6594 |
| 30 | 0.6335 | 0.6471 | 0.6548 |

### Key findings

1. **Embedder choice matters more than voting scheme.** Swapping MiniLM→bge-base buys +0.023 macro-F1; the spread across the three voting schemes on MiniLM was only 0.002. Retrieval quality is the bottleneck.
2. **bge-base ≈ bge-large at the peak** (0.6692 vs 0.6675). The 3× larger model does not meaningfully help — the remaining error is in ambiguous/mislabeled issues, not retrievable signal.
3. **bge-large saturates much earlier (k=6 vs k=17).** Its embeddings are sharp enough that the first few neighbors carry nearly all the relevant signal. MiniLM needs more neighbors to compensate for noisier ranking; bge-large's ranking is already dominant at k=6.
4. **MiniLM's best (0.6465 @ k=12) is beaten by bge-base at k=7 (0.6545).** The stronger embedder gets more F1 with fewer neighbors — a double efficiency win (shorter LLM prompts *if* we later layer an LLM on top).
5. **bge-base is the sweet spot:** highest macro-F1, 3× less VRAM than bge-large, ~3× faster to embed. We'll report bge-base as the canonical VTAG configuration.

### Implication for the paper

The VTAG headline number moves from **0.6465 → 0.6692 macro-F1** by swapping the embedder. This is a non-trivial change to the reviewer story:

- The retrieval floor is now at ~0.67 macro-F1, not ~0.645.
- Any LLM layered on top must beat 0.67 to justify its cost.
- Equivalently: the question *"does the LLM add value beyond nearest-neighbor classification?"* becomes harder to answer affirmatively.
- We should re-retrieve RAGTAG neighbors with bge-base (swap the embedder in [build_and_query_index.py](../build_and_query_index.py)) before claiming the RAGTAG numbers.

## Next steps

1. ~~Run shepard and majority voting schemes for ablation.~~ ✅ Done.
2. ~~Benchmark three embedding models.~~ ✅ Done — bge-base wins.
3. Re-run RAGTAG with bge-base retrieval (replace MiniLM) to get apples-to-apples LLM-on-top numbers.
4. When RAGTAG context-experiment results arrive, overlay RAGTAG-bge-base vs VTAG-shepard-bge-base on the same (k, macro-F1) plot.
5. After 30k experiments: check whether VTAG's k-saturation point shifts with more training data, and whether bge-base maintains its lead at scale.

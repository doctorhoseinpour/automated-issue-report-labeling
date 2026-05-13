#!/usr/bin/env python3
"""
Self-contained held-out validation experiment for selecting the BRAGTAG
margin m. This script is intentionally independent of every other code path
in the repo: it does not import from build_and_query_index.py, llm_labeler.py,
vtag.py, or any other in-repo module. It reads only issues11k_train.csv and
writes only to results/bragtag_margin_validation/. No LLM inference is run
and no test-set data is touched.

Pipeline (one shot, deterministic given SEED):
  1) Per-project stratified split of issues11k_train.csv into 270/30.
     Validation slice = 30 issues per project, 10 per label (bug/feature/
     question). Retained pool = 270 issues per project, 90 per label.
  2) Embed validation queries and retained pool with all-MiniLM-L6-v2
     (the same embedding model used by the paper's retrieval pipeline).
  3) Per-project PS FAISS index over the 270 retained issues. Each of the
     30 validation queries searches its own project's index for top-15
     neighbors.
  4) For each (k in 1..15, m in 1..5, true_label), compute the pooled fire
     rate of the BRAGTAG trigger condition N_bug(top-k) - N_question(top-k)
     <= m, stratified by the true label of the query.
  5) Select m by argmax over m of mean (across k=1..15) Youden's J statistic
     J(k, m) = TPR_question(k, m) - FPR_bug(k, m).

Outputs (under results/bragtag_margin_validation/):
  splits/validation_queries.csv     330 held-out queries
  splits/retained_pool.csv          2,970 retained issues
  neighbors/neighbors_k15.csv       330 x 15 = 4,950 neighbor rows
  analysis/fire_rates_per_k.csv     fire rate per (k, m, true_label)
  analysis/youdens_j.csv            TPR_q / FPR_b / J per (k, m)
  analysis/selected_margin.json     winning m + criterion record
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_CSV = REPO_ROOT / "issues11k_train.csv"
OUT_DIR = REPO_ROOT / "results" / "bragtag_margin_validation"
SPLITS_DIR = OUT_DIR / "splits"
NEIGH_DIR = OUT_DIR / "neighbors"
ANALYSIS_DIR = OUT_DIR / "analysis"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
K_MAX = 15
M_GRID = [1, 2, 3, 4, 5]
VAL_PER_LABEL = 10  # 10% of 100 train issues per (project, label)

LABELS = ("bug", "feature", "question")

_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    return _WS.sub(" ", str(text)).strip()


def stratified_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (repo, label) random selection of VAL_PER_LABEL issues as validation.

    Determinism: per-group rows are sorted by their original index before
    sampling, and a single numpy Generator(seed) is consumed in repo-label order.
    """
    rng = np.random.default_rng(seed)
    val_idx: list[int] = []
    keys = sorted(df.groupby(["repo", "labels"]).groups.keys())
    for key in keys:
        idxs = np.sort(df.index[(df["repo"] == key[0]) & (df["labels"] == key[1])].to_numpy())
        chosen = rng.choice(idxs, size=VAL_PER_LABEL, replace=False)
        val_idx.extend(int(i) for i in chosen)

    val_mask = df.index.isin(val_idx)
    val_df = (
        df[val_mask]
        .copy()
        .reset_index()
        .rename(columns={"index": "orig_idx"})
    )
    pool_df = (
        df[~val_mask]
        .copy()
        .reset_index()
        .rename(columns={"index": "orig_idx"})
    )
    return val_df, pool_df


def embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )
    return embs.astype("float32")


def retrieve_per_project(
    val_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    val_vecs: np.ndarray,
    pool_vecs: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict] = []
    repos = sorted(val_df["repo"].unique())
    for repo in repos:
        v_mask = (val_df["repo"].to_numpy() == repo)
        p_mask = (pool_df["repo"].to_numpy() == repo)
        v_vecs = val_vecs[v_mask]
        p_vecs = pool_vecs[p_mask]

        v_local = val_df[v_mask].reset_index(drop=True)
        p_local = pool_df[p_mask].reset_index(drop=True)

        index = faiss.IndexFlatIP(p_vecs.shape[1])
        index.add(p_vecs)
        sims, idxs = index.search(v_vecs, K_MAX)

        for qi in range(len(v_local)):
            for rank in range(K_MAX):
                ci = int(idxs[qi, rank])
                rows.append({
                    "repo": repo,
                    "query_orig_idx": int(v_local.iloc[qi]["orig_idx"]),
                    "query_label": v_local.iloc[qi]["labels"],
                    "neighbor_rank": rank,
                    "neighbor_similarity": float(sims[qi, rank]),
                    "neighbor_label": p_local.iloc[ci]["labels"],
                    "neighbor_orig_idx": int(p_local.iloc[ci]["orig_idx"]),
                })
        print(f"  {repo}: {len(v_local)} queries x {K_MAX} neighbors over pool={len(p_local)}")
    return pd.DataFrame(rows)


def compute_fire_rates(neigh_df: pd.DataFrame) -> pd.DataFrame:
    """For each (k in 1..K_MAX, m in M_GRID, true_label), pooled fire rate."""
    neigh_df = neigh_df.sort_values(["repo", "query_orig_idx", "neighbor_rank"])
    queries: list[dict] = []
    for (repo, qidx), g in neigh_df.groupby(["repo", "query_orig_idx"], sort=False):
        labels = g["neighbor_label"].to_numpy()
        queries.append({
            "repo": repo,
            "qidx": qidx,
            "q_label": g["query_label"].iloc[0],
            "cum_bug": np.cumsum(labels == "bug"),
            "cum_q": np.cumsum(labels == "question"),
        })

    rows: list[dict] = []
    for k in range(1, K_MAX + 1):
        for m in M_GRID:
            for true_lab in LABELS:
                qs = [q for q in queries if q["q_label"] == true_lab]
                n = len(qs)
                fired = sum(1 for q in qs if int(q["cum_bug"][k - 1] - q["cum_q"][k - 1]) <= m)
                rows.append({
                    "k": k,
                    "m": m,
                    "true_label": true_lab,
                    "n_queries": n,
                    "n_fired": fired,
                    "fire_rate": (fired / n) if n else 0.0,
                })
    return pd.DataFrame(rows)


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    NEIGH_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")
    df["labels"] = df["labels"].astype(str).str.lower().str.strip()
    print(f"      {len(df)} issues, {df['repo'].nunique()} projects, "
          f"labels={df['labels'].value_counts().to_dict()}")

    print(f"[2/5] Stratified split (seed={SEED}, val={VAL_PER_LABEL} per project per label)")
    val_df, pool_df = stratified_split(df, SEED)
    print(f"      Validation: {len(val_df)}  Retained pool: {len(pool_df)}")
    val_df.to_csv(SPLITS_DIR / "validation_queries.csv", index=False)
    pool_df.to_csv(SPLITS_DIR / "retained_pool.csv", index=False)

    sanity = val_df.groupby(["repo", "labels"]).size().unstack(fill_value=0)
    if not (sanity.values == VAL_PER_LABEL).all():
        raise RuntimeError(f"Stratification violated:\n{sanity}")
    print("      Stratification OK: every (repo, label) cell has exactly "
          f"{VAL_PER_LABEL} validation queries.")

    print(f"[3/5] Embedding with {EMBED_MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    val_texts = (val_df["title"] + " " + val_df["body"]).map(clean_text).tolist()
    pool_texts = (pool_df["title"] + " " + pool_df["body"]).map(clean_text).tolist()
    print(f"      Embedding {len(val_texts)} validation queries...")
    val_vecs = embed(model, val_texts)
    print(f"      Embedding {len(pool_texts)} retained pool issues...")
    pool_vecs = embed(model, pool_texts)
    print(f"      Embedding wall time: {time.time() - t0:.1f}s")

    print(f"[4/5] Per-project PS FAISS retrieval (top-{K_MAX})")
    neigh_df = retrieve_per_project(val_df, pool_df, val_vecs, pool_vecs)
    neigh_path = NEIGH_DIR / "neighbors_k15.csv"
    neigh_df.to_csv(neigh_path, index=False)
    print(f"      Wrote {neigh_path}  ({len(neigh_df)} rows)")

    print(f"[5/5] Fire-rate analysis and margin selection")
    fire_df = compute_fire_rates(neigh_df)
    fire_df.to_csv(ANALYSIS_DIR / "fire_rates_per_k.csv", index=False)

    pivot = (
        fire_df.pivot_table(index=["k", "m"], columns="true_label", values="fire_rate")
        .reset_index()
        .rename(columns={
            "question": "tpr_question",
            "bug": "fpr_bug",
            "feature": "fpr_feature",
        })
    )
    pivot["youdens_j"] = pivot["tpr_question"] - pivot["fpr_bug"]
    pivot = pivot[["k", "m", "tpr_question", "fpr_bug", "fpr_feature", "youdens_j"]]
    pivot.to_csv(ANALYSIS_DIR / "youdens_j.csv", index=False)

    by_m = (
        pivot.groupby("m")
        .agg(
            mean_tpr_question=("tpr_question", "mean"),
            mean_fpr_bug=("fpr_bug", "mean"),
            mean_fpr_feature=("fpr_feature", "mean"),
            mean_youdens_j=("youdens_j", "mean"),
        )
        .reset_index()
        .sort_values("mean_youdens_j", ascending=False)
    )
    best_m = int(by_m.iloc[0]["m"])

    result = {
        "criterion": "argmax over m of mean Youden's J = TPR_question - FPR_bug, k=1..15",
        "m_grid": M_GRID,
        "k_range": [1, K_MAX],
        "seed": SEED,
        "val_per_label_per_project": VAL_PER_LABEL,
        "embedding_model": EMBED_MODEL_NAME,
        "selected_m": best_m,
        "by_m": by_m.to_dict(orient="records"),
    }
    (ANALYSIS_DIR / "selected_margin.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSELECTED m = {best_m}")


if __name__ == "__main__":
    main()

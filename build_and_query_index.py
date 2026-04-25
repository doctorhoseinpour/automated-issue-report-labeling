#!/usr/bin/env python3
"""
build_and_query_index.py
========================
Build a FAISS index from the TRAINING set only, then query it with test issues
to retrieve top-k neighbors. This ensures strict train/test separation (no
data leakage).

Split logic:
  - Test set: first N issues per label (balanced), preserving original order.
  - Train set: everything else (remainder after removing test issues).
  - FAISS index is built ONLY from train_df.

Usage:
  Kept as a helper module (clean_text, build_faiss_index) imported by build_11k_index.py.
  Its CLI is no longer the active entry point; use build_11k_index.py for the 11k benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
_whitespace = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    return _whitespace.sub(" ", str(text)).strip()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dedup_key(row) -> str:
    t = str(row.get("title", "") or "").strip().lower()
    b = str(row.get("body", "") or "").strip().lower()
    return hashlib.md5(f"{t}||{b}".encode("utf-8")).hexdigest()


def deduplicate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if df.empty:
        return df
    keys = df.apply(_dedup_key, axis=1)
    mask = ~keys.duplicated(keep="first")
    removed = (~mask).sum()
    if removed:
        print(f"  Removed {removed} duplicate issues from {name} dataset.")
    else:
        print(f"  No duplicates in {name} dataset.")
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Normalise label column
# ---------------------------------------------------------------------------

def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "labels" in df.columns:
        return df
    if "label" in df.columns:
        df = df.rename(columns={"label": "labels"})
        return df
    raise ValueError(f"CSV needs a 'labels' or 'label' column. Found: {list(df.columns)}")


# ---------------------------------------------------------------------------
# Stratified test/train split: balanced test set, no shuffle
# ---------------------------------------------------------------------------

def parse_test_size(value: str) -> float | int:
    """Parse --test_size: float (0,1) = fraction, int >= 1 = absolute count."""
    f = float(value)
    if f < 1.0 and f > 0.0:
        return f  # fraction
    return int(f)


def split_train_test(df: pd.DataFrame, test_size: float | int):
    """
    Split into balanced test set and remainder train set.
    Test issues are taken from the TOP of each label group (no shuffle).

    Args:
        test_size: if float in (0,1), fraction per label; if int >= 1, total
                   test count (divided equally across 3 labels).
    Returns:
        train_df, test_df (both reset-indexed)
    """
    label_col = "labels"
    labels = sorted(df[label_col].unique())
    n_labels = len(labels)

    # Determine per-label test count
    if isinstance(test_size, float):
        # fraction of each label
        per_label_counts = {}
        for lab in labels:
            group_size = (df[label_col] == lab).sum()
            per_label_counts[lab] = int(group_size * test_size)
    else:
        # absolute total, split equally
        per_label = test_size // n_labels
        per_label_counts = {lab: per_label for lab in labels}

    test_indices = []
    for lab in labels:
        group = df[df[label_col] == lab]
        n_test = min(per_label_counts[lab], len(group))
        test_indices.extend(group.index[:n_test].tolist())

    test_indices = sorted(test_indices)
    train_indices = sorted(set(df.index) - set(test_indices))

    test_df = df.loc[test_indices].reset_index(drop=True)
    train_df = df.loc[train_indices].reset_index(drop=True)

    print(f"  Train/test split (test_size={test_size}):")
    for lab in labels:
        n_test = (test_df[label_col] == lab).sum()
        n_train = (train_df[label_col] == lab).sum()
        print(f"    {lab}: test={n_test}, train={n_train}")
    print(f"  Total: test={len(test_df)}, train={len(train_df)}")

    return train_df, test_df


# ---------------------------------------------------------------------------
# Cross-set dedup: remove from train any issues whose content matches test
# ---------------------------------------------------------------------------

def remove_test_duplicates_from_train(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Remove any train rows with identical title+body as a test row."""
    test_keys = set(test_df.apply(_dedup_key, axis=1))
    train_keys = train_df.apply(_dedup_key, axis=1)
    mask = ~train_keys.isin(test_keys)
    removed = (~mask).sum()
    if removed:
        print(f"  Removed {removed} train issues that duplicate test content.")
    return train_df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# FAISS index build + query
# ---------------------------------------------------------------------------

def build_faiss_index(texts: List[str], embeddings_model):
    import faiss

    print(f"  Embedding {len(texts)} documents...")
    t0 = time.time()
    vectors = embeddings_model.embed_documents(texts)
    vectors = np.array(vectors, dtype="float32")
    elapsed = time.time() - t0
    print(f"  Embedded in {elapsed:.1f}s  (shape={vectors.shape})")

    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, vectors


def query_index(
    embeddings_model,
    index,
    test_texts: List[str],
    max_k: int,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Query the FAISS index for each test issue. The index contains only train
    data, so no self-match exclusion is needed.

    Returns a tuple of (neighbor_indices, similarity_scores). Since the index
    is IndexFlatIP over L2-normalized vectors, the returned scores are cosine
    similarities in [-1, 1] (in practice close to [0, 1] for text embeddings).
    """
    import faiss

    print(f"  Querying index for {len(test_texts)} test issues (max_k={max_k})...")
    t0 = time.time()

    test_vecs = np.array(embeddings_model.embed_documents(test_texts), dtype="float32")
    faiss.normalize_L2(test_vecs)

    fetch_k = min(max_k, index.ntotal)
    distances, indices = index.search(test_vecs, fetch_k)

    all_neighbors: List[List[int]] = []
    all_sims: List[List[float]] = []
    for qi in range(len(test_texts)):
        neighbors = []
        sims = []
        for j in range(fetch_k):
            ci = int(indices[qi, j])
            if ci >= 0:
                neighbors.append(ci)
                sims.append(float(distances[qi, j]))
        all_neighbors.append(neighbors)
        all_sims.append(sims)

    elapsed = time.time() - t0
    print(f"  Querying done in {elapsed:.1f}s")
    return all_neighbors, all_sims


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index and retrieve neighbors.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--top_ks", default="1,3,9,15", help="Comma-separated k values")
    parser.add_argument("--test_size", default="0.5",
                        help="Test set size: float (0,1) = fraction per label; int >= 1 = total count (balanced)")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", required=True, help="Directory for output CSVs")
    parser.add_argument("--cache_dir", default=".faiss_cache")
    parser.add_argument("--model_cache_dir", default=None)
    args = parser.parse_args()

    ks = sorted(set(int(x) for x in args.top_ks.split(",")))
    max_k = max(ks)
    test_size = parse_test_size(args.test_size)
    print(f"K values: {ks}  (max_k={max_k})")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load and deduplicate full dataset ---
    print("Loading data...")
    full_df = ensure_labels(pd.read_csv(args.dataset))
    full_df["body"] = full_df["body"].fillna("")
    full_df["title"] = full_df["title"].fillna("")
    full_df["labels"] = full_df["labels"].astype(str).str.lower().str.strip()
    full_df = deduplicate(full_df, "full")
    print(f"  Full corpus: {len(full_df)} issues")

    # --- Split into train and test ---
    train_df, test_df = split_train_test(full_df, test_size)

    # --- Remove from train any content-identical duplicates of test issues ---
    train_df = remove_test_duplicates_from_train(train_df, test_df)

    # --- Prepare texts ---
    train_texts = (train_df["title"] + " " + train_df["body"]).apply(clean_text).tolist()
    test_texts = (test_df["title"] + " " + test_df["body"]).apply(clean_text).tolist()

    # --- Load embedding model ---
    print(f"Loading embedding model: {args.embedding_model}")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embed_kwargs = {"model_name": args.embedding_model, "model_kwargs": {"device": "cuda"}}
    if args.model_cache_dir:
        embed_kwargs["cache_folder"] = args.model_cache_dir
    embed_model = HuggingFaceEmbeddings(**embed_kwargs)

    # --- Build or load index (from TRAIN corpus only) ---
    import faiss
    os.makedirs(args.cache_dir, exist_ok=True)
    safe_name = os.path.basename(args.dataset) + "_" + args.embedding_model.replace("/", "_")
    cache_hash = hashlib.md5(open(args.dataset, "rb").read()).hexdigest()[:12]
    # Include test_size in cache key so different splits get different indexes
    cache_path = os.path.join(args.cache_dir, f"{safe_name}_{cache_hash}_test{test_size}")
    index_file = cache_path + ".index"

    if os.path.exists(index_file):
        print(f"Loading cached FAISS index from {index_file}")
        index = faiss.read_index(index_file)
    else:
        print("Building FAISS index from TRAIN corpus...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        index, vectors = build_faiss_index(train_texts, embed_model)
        faiss.write_index(index, index_file)
        print(f"  Cached index to {index_file}")
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"  GPU peak memory during indexing: {peak:.0f} MB")

    # --- Query (test issues against train index) ---
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    all_neighbors, all_sims = query_index(
        embeddings_model=embed_model,
        index=index,
        test_texts=test_texts,
        max_k=max_k,
    )
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"  GPU peak memory during retrieval: {peak:.0f} MB")

    # --- Write output CSVs (one per k) ---
    has_created = "created_at" in test_df.columns

    for k in ks:
        rows = []
        for qi in range(len(test_df)):
            test_row = test_df.iloc[qi]
            neighbors = all_neighbors[qi][:k]
            sims = all_sims[qi][:k]
            for rank, (ci, sim) in enumerate(zip(neighbors, sims)):
                train_row = train_df.iloc[ci]
                row = {
                    "test_idx": qi,
                    "test_title": test_row["title"],
                    "test_body": test_row["body"],
                    "test_label": test_row["labels"],
                }
                if has_created:
                    row["test_created_at"] = test_row.get("created_at", "")
                row.update({
                    "neighbor_rank": rank,
                    "neighbor_similarity": sim,
                    "neighbor_title": train_row["title"],
                    "neighbor_body": train_row["body"],
                    "neighbor_label": train_row["labels"],
                })
                rows.append(row)

        out_path = os.path.join(args.output_dir, f"neighbors_k{k}.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  Wrote {out_path}  ({len(rows)} rows for {len(test_df)} test issues, k={k})")

    # --- Save split metadata ---
    meta_path = os.path.join(args.output_dir, "test_split_info.csv")
    test_meta = test_df[["title", "labels"]].copy()
    test_meta.index.name = "test_idx"
    test_meta.to_csv(meta_path)
    print(f"  Wrote test split metadata to {meta_path}")

    train_meta_path = os.path.join(args.output_dir, "train_split_info.csv")
    train_df[["title", "labels"]].to_csv(train_meta_path, index=False)
    print(f"  Wrote train split metadata to {train_meta_path}")

    # --- Save full train/test splits for downstream fine-tuning ---
    save_cols = [c for c in ["title", "body", "labels", "created_at"] if c in test_df.columns]
    test_split_path = os.path.join(args.output_dir, "test_split.csv")
    test_df[save_cols].to_csv(test_split_path, index=False)
    print(f"  Wrote full test split to {test_split_path} ({len(test_df)} rows)")

    train_split_path = os.path.join(args.output_dir, "train_split.csv")
    train_df[save_cols].to_csv(train_split_path, index=False)
    print(f"  Wrote full train split to {train_split_path} ({len(train_df)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()

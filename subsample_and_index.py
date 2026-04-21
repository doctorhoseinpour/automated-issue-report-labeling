#!/usr/bin/env python3
"""
subsample_and_index.py
======================
Stratified subsample of an existing training pool, with optional FAISS index
build + neighbor retrieval for the fixed test set.

Designed for the data efficiency crossover experiment: given the full 30k
train/test split, create smaller training pools (1.5k, 3k, 9k, 15k) and
retrieve neighbors from each subsampled index.

Usage:
  # Full pipeline (subsample + FAISS + neighbors):
  python subsample_and_index.py \\
      --train_csv results/issues30k/neighbors/train_split.csv \\
      --test_csv results/issues30k/neighbors/test_split.csv \\
      --sizes 1500,3000,9000,15000 \\
      --top_ks 3,9 \\
      --output_dir results/issues30k_efficiency \\
      --seed 42

  # Subsample only (no FAISS, for FT on remote server):
  python subsample_and_index.py \\
      --train_csv results/issues30k/neighbors/train_split.csv \\
      --test_csv results/issues30k/neighbors/test_split.csv \\
      --sizes 1500,3000,9000,15000 \\
      --output_dir results/issues30k_efficiency \\
      --seed 42 \\
      --skip_indexing
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List

import numpy as np
import pandas as pd

from build_and_query_index import clean_text, build_faiss_index


def stratified_subsample(train_df: pd.DataFrame, n_total: int, seed: int = 42) -> pd.DataFrame:
    """
    Balanced stratified subsample: n_total/3 per class, deterministic.

    Uses np.random.RandomState for reproducibility across machines.
    Samples are drawn independently per label group and sorted by original
    index to maintain a stable ordering.
    """
    labels = sorted(train_df["labels"].unique())
    n_labels = len(labels)
    n_per_label = n_total // n_labels
    remainder = n_total % n_labels

    rng = np.random.RandomState(seed)
    parts = []
    for i, label in enumerate(labels):
        group = train_df[train_df["labels"] == label]
        n = n_per_label + (1 if i < remainder else 0)
        n = min(n, len(group))
        idx = rng.choice(len(group), size=n, replace=False)
        parts.append(group.iloc[sorted(idx)])

    result = pd.concat(parts).reset_index(drop=True)
    return result


def write_neighbors_csv(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    all_indices: np.ndarray,
    all_sims: np.ndarray,
    k: int,
    output_path: str,
):
    """Write neighbors CSV in the same schema as build_and_query_index.py."""
    has_created = "created_at" in test_df.columns
    rows = []
    for qi in range(len(test_df)):
        test_row = test_df.iloc[qi]
        for rank in range(k):
            ci = int(all_indices[qi, rank])
            if ci < 0:
                continue
            sim = float(all_sims[qi, rank])
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

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"  Wrote {output_path}  ({len(rows)} rows, {len(test_df)} test issues, k={k})")


def main():
    parser = argparse.ArgumentParser(
        description="Stratified subsample of training pool + FAISS index + neighbor retrieval."
    )
    parser.add_argument("--train_csv", required=True, help="Path to full train_split.csv")
    parser.add_argument("--test_csv", required=True, help="Path to full test_split.csv")
    parser.add_argument("--sizes", required=True,
                        help="Comma-separated subsample sizes (e.g. 1500,3000,9000,15000)")
    parser.add_argument("--top_ks", default="3,9",
                        help="Comma-separated k values for neighbor retrieval")
    parser.add_argument("--output_dir", required=True, help="Root output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model_cache_dir", default=None, help="HuggingFace model cache dir")
    parser.add_argument("--skip_indexing", action="store_true",
                        help="Only write subsampled train/test CSVs, skip FAISS index + neighbors")
    args = parser.parse_args()

    sizes = sorted(int(x) for x in args.sizes.split(","))
    ks = sorted(int(x) for x in args.top_ks.split(","))
    max_k = max(ks)

    print(f"Subsample sizes: {sizes}")
    print(f"K values:        {ks}  (max_k={max_k})")
    print(f"Seed:            {args.seed}")
    print(f"Skip indexing:   {args.skip_indexing}")

    # --- Load existing splits ---
    print(f"\nLoading train split: {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    train_df["body"] = train_df["body"].fillna("")
    train_df["title"] = train_df["title"].fillna("")
    if "label" in train_df.columns and "labels" not in train_df.columns:
        train_df = train_df.rename(columns={"label": "labels"})
    train_df["labels"] = train_df["labels"].astype(str).str.lower().str.strip()

    print(f"Loading test split: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    test_df["body"] = test_df["body"].fillna("")
    test_df["title"] = test_df["title"].fillna("")
    if "label" in test_df.columns and "labels" not in test_df.columns:
        test_df = test_df.rename(columns={"label": "labels"})
    test_df["labels"] = test_df["labels"].astype(str).str.lower().str.strip()

    labels = sorted(train_df["labels"].unique())
    print(f"\nFull training pool: {len(train_df)} issues")
    for lab in labels:
        print(f"  {lab}: {(train_df['labels'] == lab).sum()}")
    print(f"Test set: {len(test_df)} issues (fixed)")

    # Validate subsample sizes
    for s in sizes:
        per_label = s // len(labels)
        for lab in labels:
            available = (train_df["labels"] == lab).sum()
            if per_label > available:
                print(f"  WARNING: subsample size {s} requests {per_label} per class "
                      f"but {lab} only has {available}")

    # --- Pre-embed test texts (once, reused across all subsample sizes) ---
    test_vecs = None
    embed_model = None
    if not args.skip_indexing:
        import faiss
        print(f"\nLoading embedding model: {args.embedding_model}")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embed_kwargs = {"model_name": args.embedding_model, "model_kwargs": {"device": "cuda"}}
        if args.model_cache_dir:
            embed_kwargs["cache_folder"] = args.model_cache_dir
        embed_model = HuggingFaceEmbeddings(**embed_kwargs)

        test_texts = (test_df["title"] + " " + test_df["body"]).apply(clean_text).tolist()
        print(f"  Pre-embedding {len(test_texts)} test texts...")
        t0 = time.time()
        test_vecs = np.array(embed_model.embed_documents(test_texts), dtype="float32")
        faiss.normalize_L2(test_vecs)
        print(f"  Test embeddings ready in {time.time() - t0:.1f}s  (shape={test_vecs.shape})")

    # --- Process each subsample size ---
    save_cols = [c for c in ["title", "body", "labels", "created_at"] if c in test_df.columns]

    for size in sizes:
        sub_dir = os.path.join(args.output_dir, f"n{size}")
        print(f"\n{'='*60}")
        print(f"  Subsample size: {size}")
        print(f"  Output dir:     {sub_dir}")
        print(f"{'='*60}")

        # Skip check
        if not args.skip_indexing:
            skip_file = os.path.join(sub_dir, f"neighbors_k{max_k}.csv")
            if os.path.exists(skip_file):
                print(f"  SKIP: {skip_file} already exists")
                continue
        else:
            skip_file = os.path.join(sub_dir, "train_split.csv")
            if os.path.exists(skip_file):
                print(f"  SKIP: {skip_file} already exists")
                continue

        os.makedirs(sub_dir, exist_ok=True)

        # Stratified subsample
        t0 = time.time()
        sub_train = stratified_subsample(train_df, size, seed=args.seed)
        elapsed = time.time() - t0

        print(f"  Subsampled {len(sub_train)} issues in {elapsed:.2f}s:")
        for lab in labels:
            print(f"    {lab}: {(sub_train['labels'] == lab).sum()}")

        # Write subsampled train split
        train_path = os.path.join(sub_dir, "train_split.csv")
        sub_train[save_cols].to_csv(train_path, index=False)
        print(f"  Wrote {train_path} ({len(sub_train)} rows)")

        # Copy test split
        test_path = os.path.join(sub_dir, "test_split.csv")
        test_df[save_cols].to_csv(test_path, index=False)
        print(f"  Wrote {test_path} ({len(test_df)} rows)")

        if args.skip_indexing:
            continue

        # Build FAISS index from subsampled training data
        import faiss
        sub_texts = (sub_train["title"] + " " + sub_train["body"]).apply(clean_text).tolist()
        print(f"\n  Building FAISS index from {len(sub_texts)} subsampled train texts...")
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        index, _vectors = build_faiss_index(sub_texts, embed_model)

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"  GPU peak memory during indexing: {peak:.0f} MB")

        # Search: test vectors against subsampled index
        fetch_k = min(max_k, index.ntotal)
        print(f"  Searching {len(test_vecs)} test queries against {index.ntotal}-vector index (k={fetch_k})...")
        t0 = time.time()
        distances, indices = index.search(test_vecs, fetch_k)
        print(f"  Search done in {time.time() - t0:.1f}s")

        # Write neighbor CSVs
        for k in ks:
            actual_k = min(k, fetch_k)
            out_path = os.path.join(sub_dir, f"neighbors_k{k}.csv")
            write_neighbors_csv(test_df, sub_train, indices, distances, actual_k, out_path)

    print("\nAll subsample sizes processed.")


if __name__ == "__main__":
    main()

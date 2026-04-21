#!/usr/bin/env python3
"""
build_11k_index.py
==================
Build a FAISS index from pre-split train/test CSVs and retrieve neighbors.
Supports optional --repo_filter to restrict to a single project.

Designed for the 11-project experiment pipeline. Reuses clean_text and
build_faiss_index from build_and_query_index.py.

Usage:
  # Agnostic (all 3300 train issues):
  python build_11k_index.py \
      --train_csv issues11k_train.csv \
      --test_csv issues11k_test.csv \
      --top_ks 3,9,30 \
      --output_dir results/issues11k/agnostic/neighbors

  # Project-specific:
  python build_11k_index.py \
      --train_csv issues11k_train.csv \
      --test_csv issues11k_test.csv \
      --top_ks 3,9,30 \
      --repo_filter "facebook/react" \
      --output_dir results/issues11k/project_specific/facebook_react/neighbors
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from build_and_query_index import clean_text, build_faiss_index


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
        description="Build FAISS index from pre-split CSVs + retrieve neighbors."
    )
    parser.add_argument("--train_csv", required=True, help="Path to train split CSV")
    parser.add_argument("--test_csv", required=True, help="Path to test split CSV")
    parser.add_argument("--top_ks", default="3,9,30",
                        help="Comma-separated k values for neighbor retrieval")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--repo_filter", default=None,
                        help="If given, filter both CSVs to this repo only")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model_cache_dir", default=None, help="HuggingFace model cache dir")
    args = parser.parse_args()

    ks = sorted(int(x) for x in args.top_ks.split(","))
    max_k = max(ks)

    # --- Load and optionally filter ---
    print(f"Loading train: {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    print(f"Loading test:  {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)

    # Normalize columns
    for df in [train_df, test_df]:
        df["body"] = df["body"].fillna("")
        df["title"] = df["title"].fillna("")
        if "label" in df.columns and "labels" not in df.columns:
            df.rename(columns={"label": "labels"}, inplace=True)
        df["labels"] = df["labels"].astype(str).str.lower().str.strip()

    if args.repo_filter:
        print(f"Filtering to repo: {args.repo_filter}")
        train_df = train_df[train_df["repo"] == args.repo_filter].reset_index(drop=True)
        test_df = test_df[test_df["repo"] == args.repo_filter].reset_index(drop=True)

    print(f"Train: {len(train_df)} issues, Test: {len(test_df)} issues")
    for lab in sorted(train_df["labels"].unique()):
        print(f"  {lab}: train={int((train_df['labels'] == lab).sum())}, "
              f"test={int((test_df['labels'] == lab).sum())}")

    # --- Skip check ---
    skip_file = os.path.join(args.output_dir, f"neighbors_k{max_k}.csv")
    if os.path.exists(skip_file):
        print(f"\nSKIP: {skip_file} already exists")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Save splits (preserve all columns including repo) ---
    save_cols = [c for c in ["repo", "created_at", "labels", "title", "body"]
                 if c in test_df.columns]

    train_path = os.path.join(args.output_dir, "train_split.csv")
    train_df[save_cols].to_csv(train_path, index=False)
    print(f"Wrote {train_path} ({len(train_df)} rows)")

    test_path = os.path.join(args.output_dir, "test_split.csv")
    test_df[save_cols].to_csv(test_path, index=False)
    print(f"Wrote {test_path} ({len(test_df)} rows)")

    # --- Build FAISS index ---
    import faiss
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print(f"\nLoading embedding model: {args.embedding_model}")
    embed_kwargs = {"model_name": args.embedding_model, "model_kwargs": {"device": "cuda"}}
    if args.model_cache_dir:
        embed_kwargs["cache_folder"] = args.model_cache_dir
    embed_model = HuggingFaceEmbeddings(**embed_kwargs)

    # Embed test texts
    test_texts = (test_df["title"] + " " + test_df["body"]).apply(clean_text).tolist()
    print(f"Embedding {len(test_texts)} test texts...")
    t0 = time.time()
    test_vecs = np.array(embed_model.embed_documents(test_texts), dtype="float32")
    faiss.normalize_L2(test_vecs)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Build index from train texts
    train_texts = (train_df["title"] + " " + train_df["body"]).apply(clean_text).tolist()
    print(f"Building FAISS index from {len(train_texts)} train texts...")
    t0 = time.time()
    index, _ = build_faiss_index(train_texts, embed_model)
    print(f"  Done in {time.time() - t0:.1f}s  (index size={index.ntotal})")

    # --- Search ---
    fetch_k = min(max_k, index.ntotal)
    print(f"Searching {len(test_vecs)} test queries (k={fetch_k})...")
    t0 = time.time()
    distances, indices = index.search(test_vecs, fetch_k)
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- Write neighbor CSVs ---
    for k in ks:
        k_actual = min(k, fetch_k)
        out_path = os.path.join(args.output_dir, f"neighbors_k{k}.csv")
        write_neighbors_csv(test_df, train_df, indices, distances, k_actual, out_path)

    print("\nDone!")


if __name__ == "__main__":
    main()

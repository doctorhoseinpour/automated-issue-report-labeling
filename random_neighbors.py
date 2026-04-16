#!/usr/bin/env python3
"""
random_neighbors.py
===================
Generates neighbor CSVs with RANDOM training examples instead of FAISS-retrieved
ones. Drop-in replacement for build_and_query_index.py output — same CSV schema,
so llm_labeler.py and evaluate.py work unchanged.

Used as an ablation: if RAGTAG with similar neighbors >> RAGTAG with random neighbors,
then retrieval quality (not just few-shotting) drives the performance gain.

Usage:
  python random_neighbors.py \
    --train_csv results/run_XXX/neighbors/train_split.csv \
    --test_csv results/run_XXX/neighbors/test_split.csv \
    --top_ks 3 \
    --seeds 1,2,3 \
    --output_dir results/ablation_random/
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_random_neighbors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    has_created = "created_at" in test_df.columns

    rows = []
    for qi in range(len(test_df)):
        test_row = test_df.iloc[qi]
        idxs = rng.choice(len(train_df), size=k, replace=False)

        for rank, ti in enumerate(idxs):
            train_row = train_df.iloc[ti]
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
                "neighbor_similarity": 0.0,
                "neighbor_title": train_row["title"],
                "neighbor_body": train_row["body"],
                "neighbor_label": train_row["labels"],
            })
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate random neighbor CSVs for ablation study"
    )
    parser.add_argument("--train_csv", required=True,
                        help="Path to train_split.csv from build_and_query_index.py")
    parser.add_argument("--test_csv", required=True,
                        help="Path to test_split.csv from build_and_query_index.py")
    parser.add_argument("--top_ks", required=True,
                        help="Comma-separated k values (e.g. '3' or '1,3,9')")
    parser.add_argument("--seeds", default="1,2,3",
                        help="Comma-separated random seeds (default: 1,2,3)")
    parser.add_argument("--output_dir", required=True,
                        help="Root output dir (subdirs created per seed)")

    args = parser.parse_args()
    ks = [int(x) for x in args.top_ks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    output_root = Path(args.output_dir)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    print(f"k values: {ks}, seeds: {seeds}")

    # Copy train/test splits to output root so downstream scripts can find them
    output_root.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_root / "train_split.csv", index=False)
    test_df.to_csv(output_root / "test_split.csv", index=False)
    print(f"Copied splits to {output_root}/")

    for seed in seeds:
        seed_dir = output_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for k in ks:
            df = generate_random_neighbors(train_df, test_df, k, seed)
            out_path = seed_dir / f"neighbors_k{k}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Wrote {out_path} ({len(df)} rows, {len(test_df)} issues × k={k})")

    print(f"\nDone. {len(seeds)} seeds × {len(ks)} k values = {len(seeds) * len(ks)} files")
    print(f"Output: {output_root}/")


if __name__ == "__main__":
    main()

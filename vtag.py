#!/usr/bin/env python3
"""
vtag.py — Voting-based RAG baseline (non-LLM)
=============================================
Classifies test issues by VOTING over the labels of their top-k retrieved
nearest-neighbor training issues. No LLM is used at inference.

Serves as a principled baseline that isolates the contribution of retrieval
alone vs. retrieval + LLM reasoning (RAGTAG). If VTAG already achieves strong
performance, it means the embedding space is doing most of the work.

Voting strategies (select via --voting):

  similarity    (default) Distance-weighted k-NN (Dudani 1976): each neighbor
                casts a vote of weight = cosine similarity to the query.
                score(c) = Σ sim_i over neighbors with label c
                This is the standard weighted k-NN rule and typically the
                strongest voting scheme for retrieval-based classification.

  shepard       Shepard's method: weights are squared similarities, which
                sharpens the influence of the closest neighbors.
                score(c) = Σ sim_i² over neighbors with label c

  majority      Plain one-vote-per-neighbor (uniform weights). Reported as
                an ablation to isolate the contribution of weighting.
                score(c) = |{i : label_i = c}|

All strategies are DETERMINISTIC. Ties are broken in favour of the label of
the single highest-similarity neighbor among the tied candidates (stable,
reproducible, no seeds).

Usage:
  # Retrieve neighbors with similarity scores first (one-time):
  python build_and_query_index.py --dataset issues3k.csv --top_ks "30" \\
      --test_size 0.5 --output_dir results/vtag/neighbors

  # Run VTAG:
  python vtag.py \\
      --neighbors_csv results/vtag/neighbors/neighbors_k30.csv \\
      --output_dir results/vtag/predictions/similarity \\
      --eval_dir results/vtag/evaluations/similarity \\
      --ks "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30" \\
      --voting similarity
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

VALID_LABELS = ["bug", "feature", "question"]
VALID_LABELS_SET = set(VALID_LABELS)

# Canonicalize common label synonyms to the three target classes.
# Kept in sync with llm_labeler.py's _CANON_MAP so RAGTAG and VTAG treat
# any edge-case training labels the same way.
_CANON_MAP = {
    "enhancement": "feature", "feature-request": "feature",
    "feature_request": "feature", "feat": "feature", "request": "feature",
    "bugfix": "bug", "defect": "bug", "issue": "bug", "fix": "bug",
    "support": "question", "howto": "question", "help": "question",
}


def canonicalize_label(raw: str) -> str:
    """Lowercase + map synonyms. Unknown labels fall through unchanged."""
    x = str(raw).strip().lower()
    if x in VALID_LABELS_SET:
        return x
    return _CANON_MAP.get(x, x)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_neighbors_grouped(path: str) -> List[dict]:
    """
    Load a neighbors CSV (as produced by build_and_query_index.py) and group
    by test_idx. Returns a list of dicts, one per test issue, with neighbors
    sorted by ascending rank (= descending similarity).
    """
    df = pd.read_csv(path)

    required_cols = {"test_idx", "neighbor_rank", "neighbor_label", "neighbor_similarity"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"neighbors CSV missing required columns: {missing}. "
            f"Re-run build_and_query_index.py to regenerate with similarity scores."
        )

    df = df.sort_values(["test_idx", "neighbor_rank"]).reset_index(drop=True)

    groups: Dict[int, dict] = {}
    for _, row in df.iterrows():
        ti = int(row["test_idx"])
        if ti not in groups:
            groups[ti] = {
                "test_idx": ti,
                "title": str(row.get("test_title", "")),
                "body": str(row.get("test_body", "")),
                "ground_truth": canonicalize_label(row.get("test_label", "")),
                "neighbors": [],
            }
        groups[ti]["neighbors"].append({
            "rank": int(row["neighbor_rank"]),
            "similarity": float(row["neighbor_similarity"]),
            "label": canonicalize_label(row["neighbor_label"]),
        })

    return [groups[i] for i in sorted(groups.keys())]


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def _weight_for(nb: dict, voting: str) -> float:
    sim = nb["similarity"]
    if voting == "similarity":
        return sim
    if voting == "shepard":
        return sim * sim
    if voting == "majority":
        return 1.0
    raise ValueError(f"Unknown voting scheme: {voting}")


def vote(neighbors_sorted: List[dict], k: int, voting: str) -> str:
    """
    Vote on a label using the top-k neighbors (already sorted by rank).
    Deterministic tie-break: label of the highest-similarity neighbor
    among tied candidates.
    """
    top_k = neighbors_sorted[:k]
    scores: Dict[str, float] = defaultdict(float)
    for nb in top_k:
        scores[nb["label"]] += _weight_for(nb, voting)

    max_score = max(scores.values())
    winners = {lab for lab, s in scores.items() if s == max_score}

    if len(winners) == 1:
        return next(iter(winners))

    # Tie-break: first (highest-similarity) neighbor whose label is a winner.
    for nb in top_k:
        if nb["label"] in winners:
            return nb["label"]
    return next(iter(winners))  # unreachable in practice


# ---------------------------------------------------------------------------
# Per-k run
# ---------------------------------------------------------------------------

def run_one_k(
    neighbors_data: List[dict],
    k: int,
    voting: str,
    output_csv: str,
) -> float:
    """Run VTAG at a given k, write preds CSV, return elapsed wall-time."""
    t0 = time.time()
    results = []
    for issue in neighbors_data:
        pred = vote(issue["neighbors"], k, voting)
        results.append({
            "test_idx": issue["test_idx"],
            "title": issue["title"],
            "body": issue["body"],
            "ground_truth": issue["ground_truth"],
            "predicted_label": pred,
            "raw_output": f"<label>{pred}</label>",
            "truncated": False,
            "neighbors_truncated": False,
            "query_truncated": False,
            "tokens_removed": 0,
            "parsed_via": f"vtag-{voting}",
            "prompt_tokens": 0,
            "generated_tokens": 0,
        })
    elapsed = time.time() - t0

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VTAG: voting-based RAG baseline (no LLM required)."
    )
    parser.add_argument("--neighbors_csv", required=True,
                        help="Path to neighbors CSV with >= max(ks) neighbors per test "
                             "issue and a 'neighbor_similarity' column.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for prediction CSVs + cost_metrics.csv.")
    parser.add_argument("--eval_dir", default=None,
                        help="Optional directory for evaluation CSVs. If given, "
                             "evaluate.py is invoked after each k.")
    parser.add_argument("--ks", default=",".join(str(i) for i in range(1, 31)),
                        help="Comma-separated k values (default: 1..30).")
    parser.add_argument("--voting", choices=["similarity", "shepard", "majority"],
                        default="similarity",
                        help="Voting strategy (default: similarity).")
    parser.add_argument("--model_name_for_eval", default=None,
                        help="Label used in evaluation CSVs "
                             "(default: VTAG-<voting>).")
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    if not ks:
        parser.error("--ks must contain at least one value")

    model_name = args.model_name_for_eval or f"VTAG-{args.voting}"

    print(f"{'=' * 60}")
    print(f"  VTAG — voting-based RAG baseline")
    print(f"{'=' * 60}")
    print(f"  Voting:       {args.voting}")
    print(f"  Neighbors:    {args.neighbors_csv}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Eval dir:     {args.eval_dir or '(skipped)'}")
    print(f"  K values:     {ks[0]}..{ks[-1]} ({len(ks)} values)")
    print(f"{'=' * 60}")

    # --- Load neighbor data ---
    load_t0 = time.time()
    neighbors_data = load_neighbors_grouped(args.neighbors_csv)
    load_elapsed = time.time() - load_t0
    max_available_k = min(len(issue["neighbors"]) for issue in neighbors_data)
    print(f"\nLoaded {len(neighbors_data)} test issues with up to "
          f"{max_available_k} neighbors each  ({load_elapsed:.1f}s)")

    invalid_ks = [k for k in ks if k < 1 or k > max_available_k]
    if invalid_ks:
        print(f"  WARNING: skipping k={invalid_ks} (out of available range 1..{max_available_k})")
        ks = [k for k in ks if 1 <= k <= max_available_k]

    os.makedirs(args.output_dir, exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, "evaluate.py")

    all_cost_stats = []

    # --- Run each k ---
    for k in ks:
        output_csv = os.path.join(args.output_dir, f"preds_k{k}.csv")

        if os.path.exists(output_csv):
            print(f"  [k={k:>2}]  predictions already exist, skipping")
            df_existing = pd.read_csv(output_csv)
            acc = (df_existing["predicted_label"] == df_existing["ground_truth"]).mean()
            print(f"           cached accuracy = {100 * acc:.2f}%")
            continue

        elapsed = run_one_k(neighbors_data, k, args.voting, output_csv)

        df = pd.read_csv(output_csv)
        acc = (df["predicted_label"] == df["ground_truth"]).mean()
        n_total = len(df)
        print(f"  [k={k:>2}]  accuracy = {100 * acc:.2f}%   "
              f"({elapsed * 1000:.1f} ms total, {1000 * elapsed / n_total:.2f} ms/issue)")

        all_cost_stats.append({
            "model": model_name,
            "top_k": k,
            "k_label": f"k{k}",
            "voting": args.voting,
            "wall_time_s": round(elapsed, 4),
            "issues_per_second": round(n_total / elapsed, 1) if elapsed > 0 else 0.0,
            "total_issues": n_total,
            "total_prompt_tokens": 0,
            "total_generated_tokens": 0,
            "avg_prompt_tokens": 0.0,
            "avg_generated_tokens": 0.0,
            "min_prompt_tokens": 0,
            "max_prompt_tokens": 0,
            "gpu_peak_memory_mb": 0.0,
            "model_load_time_s": 0.0,
            "gpu_device": "N/A (CPU voting)",
            "gpu_total_memory_mb": 0,
            "max_seq_length": 0,
            "max_new_tokens": 0,
            "load_in_4bit": False,
        })

        # --- Evaluate this k ---
        if args.eval_dir and os.path.exists(eval_script):
            os.makedirs(args.eval_dir, exist_ok=True)
            eval_csv = os.path.join(args.eval_dir, f"eval_k{k}.csv")
            subprocess.run(
                [
                    sys.executable, eval_script,
                    "--preds_csv", output_csv,
                    "--top_k", str(k),
                    "--output_csv", eval_csv,
                    "--model_name", model_name,
                ],
                check=False,
            )

    # --- Write cost metrics ---
    if all_cost_stats:
        cost_csv = os.path.join(args.output_dir, "cost_metrics.csv")
        pd.DataFrame(all_cost_stats).to_csv(cost_csv, index=False)
        print(f"\n  Cost metrics written to: {cost_csv}")

    # --- Aggregate eval summary if eval_dir is set ---
    if args.eval_dir and os.path.exists(args.eval_dir):
        eval_files = sorted(Path(args.eval_dir).glob("eval_k*.csv"))
        if eval_files:
            dfs = [pd.read_csv(f) for f in eval_files]
            agg = pd.concat(dfs, ignore_index=True).sort_values("top_k")
            agg_path = os.path.join(args.eval_dir, "all_results.csv")
            agg.to_csv(agg_path, index=False)

            # Compact summary
            summary_cols = ["top_k", "accuracy", "f1_macro", "f1_weighted",
                            "f1_bug", "f1_feature", "f1_question", "invalid_rate"]
            summary_cols = [c for c in summary_cols if c in agg.columns]
            print(f"\n{'=' * 60}")
            print(f"  VTAG-{args.voting} summary across k:")
            print(f"{'=' * 60}")
            print(agg[summary_cols].to_string(index=False))
            print(f"\n  Aggregated evaluation: {agg_path}")

            if "f1_macro" in agg.columns:
                best = agg.loc[agg["f1_macro"].idxmax()]
                print(f"\n  BEST macro-F1: {best['f1_macro']:.4f} @ k={int(best['top_k'])}")

    print("\nDone.")


if __name__ == "__main__":
    main()

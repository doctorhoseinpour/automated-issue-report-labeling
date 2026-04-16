#!/usr/bin/env python3
"""
analyze_prompt_tokens.py
========================
Analyzes the distribution of prompt token counts for RAGTAG few-shot prompts
across k values WITHOUT loading an LLM — only downloads/uses the tokenizer.

Replicates llm_labeler.py's exact prompt construction (same SYSTEM_PROMPT,
same chat template application, same "<label>" prefill) so results are
directly representative of what the real pipeline produces.

Neighbor strategy:
  - If --neighbors_dir is given and neighbors_k{k}.csv exists → real FAISS neighbors
  - Otherwise → random sample from train set (fast, good approximation)

Token counts are measured WITHOUT the truncation logic so you see the raw
distribution and can decide which context window to use.

Usage:
  python analyze_prompt_tokens.py \\
    --dataset issues3k.csv \\
    --tokenizer unsloth/Llama-3.2-3B-Instruct \\
    [--neighbors_dir results/run_<stamp>/neighbors] \\
    [--top_ks "0,1,3,9,15"] \\
    [--test_size 0.5] \\
    [--seed 42] \\
    [--cache_dir ./hf_cache]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Import prompt-building and split logic directly from the project so
# the analysis exactly matches what the real pipeline produces.
sys.path.insert(0, str(Path(__file__).parent))
from llm_labeler import build_chat_messages  # noqa: E402
from build_and_query_index import (  # noqa: E402
    deduplicate,
    ensure_labels,
    parse_test_size,
    remove_test_duplicates_from_train,
    split_train_test,
)

CONTEXT_WINDOWS = [2048, 4096, 8192, 16384]


# ---------------------------------------------------------------------------
# Neighbor loading
# ---------------------------------------------------------------------------

def load_real_neighbors(neighbors_dir: str, k: int) -> Optional[List[dict]]:
    """
    Load test issues + their top-k neighbors from a neighbor CSV produced by
    build_and_query_index.py. Returns None if the file doesn't exist.
    """
    path = os.path.join(neighbors_dir, f"neighbors_k{k}.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    issues: Dict[int, dict] = {}
    for _, row in df.iterrows():
        ti = int(row["test_idx"])
        if ti not in issues:
            issues[ti] = {
                "title": str(row.get("test_title", "")),
                "body": str(row.get("test_body", "")),
                "label": str(row.get("test_label", "")),
                "neighbors": [],
            }
        if len(issues[ti]["neighbors"]) < k:
            issues[ti]["neighbors"].append({
                "title": str(row.get("neighbor_title", "")),
                "body": str(row.get("neighbor_body", "")),
                "label": str(row.get("neighbor_label", "")),
            })
    return [issues[i] for i in sorted(issues.keys())]


# ---------------------------------------------------------------------------
# Prompt building + token counting
# ---------------------------------------------------------------------------

def count_tokens_for_k(
    test_issues: List[dict],
    train_df: pd.DataFrame,
    k: int,
    tokenizer,
    seed: int,
) -> List[int]:
    """
    Build prompts for every test issue at the given k and return a list of
    token counts. Passes max_prompt_tokens=999_999 so truncation never fires —
    we want the raw sizes.
    """
    rng = random.Random(seed)
    train_records = train_df.to_dict("records")
    token_counts = []

    for issue in test_issues:
        if k == 0:
            neighbors = []
        elif "neighbors" in issue and issue["neighbors"]:
            neighbors = issue["neighbors"][:k]
        else:
            # Random fallback: sample k issues from train set
            sampled = rng.sample(train_records, min(k, len(train_records)))
            neighbors = [
                {
                    "title": str(r.get("title", "")),
                    "body": str(r.get("body", "")),
                    "label": str(r.get("labels", "bug")),
                }
                for r in sampled
            ]

        messages, _ = build_chat_messages(
            test_title=issue["title"],
            test_body=issue["body"],
            neighbors=neighbors,
            k=k,
            is_thinking_model=False,
            max_prompt_tokens=999_999,  # disable truncation
            tokenizer=tokenizer,
        )

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt += "<label>"  # assistant prefill, exactly as in llm_labeler.py

        token_counts.append(len(tokenizer.encode(prompt)))

    return token_counts


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_stats_table(results: Dict[str, List[int]]) -> None:
    PCTS = [50, 75, 90, 95, 99]
    col_headers = ["k", "n", "min", "mean", "p50", "p75", "p90", "p95", "p99", "max"]
    for ctx in CONTEXT_WINDOWS:
        col_headers.append(f"≥{ctx // 1024}k")

    # Compute rows first so we can size columns
    rows = []
    for k_label, counts in results.items():
        arr = np.array(counts)
        n = len(arr)
        row = [
            k_label,
            str(n),
            str(int(arr.min())),
            f"{arr.mean():.0f}",
        ]
        for p in PCTS:
            row.append(str(int(np.percentile(arr, p))))
        row.append(str(int(arr.max())))
        for ctx in CONTEXT_WINDOWS:
            pct = 100.0 * (arr >= ctx).sum() / n
            row.append(f"{pct:.1f}%")
        rows.append(row)

    col_widths = [max(len(h), max(len(r[i]) for r in rows))
                  for i, h in enumerate(col_headers)]

    sep = "  ".join("-" * w for w in col_widths)
    header = "  ".join(h.rjust(w) for h, w in zip(col_headers, col_widths))
    total_w = len(header)

    print(f"\n{'=' * total_w}")
    print("PROMPT TOKEN STATS  (raw, no truncation applied)")
    print(f"  Columns ≥Nk = % of prompts that meet or exceed that token count")
    print(f"{'=' * total_w}")
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(v.rjust(w) for v, w in zip(row, col_widths)))
    print()


def print_coverage_table(results: Dict[str, List[int]]) -> None:
    print("CONTEXT WINDOW FIT RATE")
    print("  (% of prompts that fit WITHIN each window — i.e. < N tokens)")
    print()

    # Column widths
    k_col_w = max(len(k) for k in results) + 4
    ctx_col_w = 10

    header = f"  {'k':<{k_col_w}}" + "".join(
        f"  {'<' + str(ctx):>{ctx_col_w}}" for ctx in CONTEXT_WINDOWS
    )
    print(header)
    print("  " + "-" * (k_col_w + (ctx_col_w + 2) * len(CONTEXT_WINDOWS)))

    for k_label, counts in results.items():
        arr = np.array(counts)
        n = len(arr)
        line = f"  {k_label:<{k_col_w}}"
        for ctx in CONTEXT_WINDOWS:
            fit_pct = 100.0 * (arr < ctx).sum() / n
            line += f"  {fit_pct:>{ctx_col_w - 1}.1f}%"
        print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze RAGTAG prompt token distributions (tokenizer-only, no GPU)."
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to dataset CSV (e.g. issues3k.csv)")
    parser.add_argument("--tokenizer", required=True,
                        help="HuggingFace model/tokenizer ID "
                             "(e.g. unsloth/Llama-3.2-3B-Instruct)")
    parser.add_argument("--top_ks", default="0,1,3,9,15",
                        help="Comma-separated k values to analyze (default: 0,1,3,9,15)")
    parser.add_argument("--test_size", default="0.5",
                        help="Test set size — fraction (0,1) or absolute count (default: 0.5)")
    parser.add_argument("--neighbors_dir", default=None,
                        help="Optional: directory with real FAISS neighbor CSVs. "
                             "If omitted (or file missing), random train samples are used.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for neighbor sampling fallback (default: 42)")
    parser.add_argument("--cache_dir", default=None,
                        help="HuggingFace cache directory (e.g. ./hf_cache on NRP)")
    args = parser.parse_args()

    ks = [int(x) for x in args.top_ks.split(",")]
    test_size = parse_test_size(args.test_size)

    # ------------------------------------------------------------------
    # Load tokenizer only (no model weights)
    # ------------------------------------------------------------------
    print(f"\nLoading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer  # noqa: E402 (heavy import, keep local)

    tok_kwargs = {}
    if args.cache_dir:
        tok_kwargs["cache_dir"] = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    # ------------------------------------------------------------------
    # Load + split dataset (same logic as build_and_query_index.py)
    # ------------------------------------------------------------------
    print(f"\nLoading dataset: {args.dataset}")
    df = ensure_labels(pd.read_csv(args.dataset))
    df["body"] = df["body"].fillna("")
    df["title"] = df["title"].fillna("")
    df["labels"] = df["labels"].astype(str).str.lower().str.strip()
    df = deduplicate(df, "full")

    train_df, test_df = split_train_test(df, test_size)
    train_df = remove_test_duplicates_from_train(train_df, test_df)
    print(f"  Test: {len(test_df)} issues   Train: {len(train_df)} issues")

    # Flatten test_df into list-of-dicts (no neighbors yet)
    test_as_dicts = [
        {"title": str(r["title"]), "body": str(r["body"]), "label": str(r["labels"])}
        for r in test_df.to_dict("records")
    ]

    # ----------------------------------------------------------model--------
    # Build prompts and count tokens for each k
    # ------------------------------------------------------------------
    results: Dict[str, List[int]] = {}

    for k in ks:
        k_label = f"{k} (zero-shot)" if k == 0 else str(k)
        print(f"\nk={k} — building {len(test_as_dicts)} prompts...")

        # Prefer real neighbors when available
        test_issues = None
        if args.neighbors_dir and k > 0:
            test_issues = load_real_neighbors(args.neighbors_dir, k)
            if test_issues:
                print(f"  Using real FAISS neighbors ({args.neighbors_dir}/neighbors_k{k}.csv)")
            else:
                print(f"  neighbors_k{k}.csv not found → falling back to random train samples")

        if test_issues is None:
            test_issues = test_as_dicts

        counts = count_tokens_for_k(test_issues, train_df, k, tokenizer, args.seed)
        results[k_label] = counts

        arr = np.array(counts)
        print(f"  mean={arr.mean():.0f}  median={np.median(arr):.0f}  "
              f"p95={np.percentile(arr, 95):.0f}  max={arr.max()}")

    # ------------------------------------------------------------------
    # Print summary tables
    # ------------------------------------------------------------------
    print_stats_table(results)
    print_coverage_table(results)


if __name__ == "__main__":
    main()

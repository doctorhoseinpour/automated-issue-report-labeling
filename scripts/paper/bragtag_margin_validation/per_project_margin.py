#!/usr/bin/env python3
"""
Per-project diagnostic for the BRAGTAG margin m.

For each project, compute mean Youden's J = TPR_question - FPR_bug averaged
over k=1..15 for each m in {1,2,3,4,5}, twice:
  (A) on the held-out validation slice (10/10/10 per project from
      issues11k_train.csv, 270 retrieval pool per project) -- the same data
      the paper's calibration used, but grouped by project instead of
      pooled.
  (B) on the test set (300 PS test queries per project against the full
      300-issue PS training pool), reading the already-computed
      neighbors_k30.csv files. This peeks at the test set and is therefore
      a diagnostic only -- not a margin-selection procedure.

No LLM inference. Reads existing neighbor CSVs and writes:
  results/bragtag_margin_validation/analysis/per_project_margin_val.csv
  results/bragtag_margin_validation/analysis/per_project_margin_test.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
VAL_NEIGH = REPO_ROOT / "results" / "bragtag_margin_validation" / "neighbors" / "neighbors_k15.csv"
PS_BASE = REPO_ROOT / "results" / "issues11k" / "project_specific"
OUT_DIR = REPO_ROOT / "results" / "bragtag_margin_validation" / "analysis"

K_MAX = 15
M_GRID = [1, 2, 3, 4, 5]
LABELS = ("bug", "feature", "question")


def fire_rates(neigh_df: pd.DataFrame) -> pd.DataFrame:
    """Per (repo, k, m, true_label) fire rate of (N_bug - N_q) <= m in top-k."""
    neigh_df = neigh_df.sort_values(["repo", "qidx", "neighbor_rank"])
    queries = []
    for (repo, qidx), g in neigh_df.groupby(["repo", "qidx"], sort=False):
        labels = g["neighbor_label"].to_numpy()
        queries.append({
            "repo": repo,
            "q_label": g["q_label"].iloc[0],
            "cum_bug": np.cumsum(labels == "bug"),
            "cum_q": np.cumsum(labels == "question"),
        })

    rows = []
    repos = sorted({q["repo"] for q in queries})
    for repo in repos:
        for k in range(1, K_MAX + 1):
            for m in M_GRID:
                for true_lab in LABELS:
                    qs = [q for q in queries if q["repo"] == repo and q["q_label"] == true_lab]
                    n = len(qs)
                    fired = sum(
                        1 for q in qs
                        if int(q["cum_bug"][k - 1] - q["cum_q"][k - 1]) <= m
                    )
                    rows.append({
                        "repo": repo, "k": k, "m": m, "true_label": true_lab,
                        "n": n, "fired": fired,
                        "fire_rate": (fired / n) if n else 0.0,
                    })
    return pd.DataFrame(rows)


def mean_j_table(fire_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        fire_df.pivot_table(
            index=["repo", "k", "m"], columns="true_label", values="fire_rate"
        )
        .reset_index()
    )
    pivot["J"] = pivot["question"] - pivot["bug"]
    table = pivot.groupby(["repo", "m"])["J"].mean().unstack("m")
    table.columns = [f"m={m}" for m in table.columns]
    return table.reset_index()


def load_val_neighbors() -> pd.DataFrame:
    df = pd.read_csv(VAL_NEIGH)
    return df.rename(columns={
        "query_orig_idx": "qidx",
        "query_label": "q_label",
    })[["repo", "qidx", "q_label", "neighbor_rank", "neighbor_label"]]


def load_test_neighbors() -> pd.DataFrame:
    parts = []
    for proj_dir in sorted(PS_BASE.iterdir()):
        if not proj_dir.is_dir():
            continue
        f = proj_dir / "neighbors" / "neighbors_k30.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=["test_idx", "test_label", "neighbor_rank", "neighbor_label"])
        df = df[df["neighbor_rank"] < K_MAX].copy()
        df["repo"] = proj_dir.name
        df = df.rename(columns={"test_idx": "qidx", "test_label": "q_label"})
        parts.append(df[["repo", "qidx", "q_label", "neighbor_rank", "neighbor_label"]])
    return pd.concat(parts, ignore_index=True)


def pooled_row(fire_df: pd.DataFrame, label: str) -> dict:
    """Repeat the paper's pooled calculation across all projects."""
    pooled = (
        fire_df.groupby(["k", "m", "true_label"])
        .agg(n=("n", "sum"), fired=("fired", "sum"))
        .reset_index()
    )
    pooled["fire_rate"] = pooled["fired"] / pooled["n"].replace(0, np.nan)
    piv = pooled.pivot_table(index=["k", "m"], columns="true_label", values="fire_rate").reset_index()
    piv["J"] = piv["question"] - piv["bug"]
    by_m = piv.groupby("m")["J"].mean()
    out = {"repo": label}
    for m in M_GRID:
        out[f"m={m}"] = float(by_m.loc[m])
    return out


def report(label: str, fire_df: pd.DataFrame) -> pd.DataFrame:
    per_proj = mean_j_table(fire_df)
    pooled = pd.DataFrame([pooled_row(fire_df, "POOLED")])
    table = pd.concat([per_proj, pooled], ignore_index=True)
    print(f"\n=== {label}: mean Youden's J = TPR_question - FPR_bug, averaged k=1..15 ===")
    with pd.option_context("display.float_format", lambda v: f"{v:+.3f}"):
        print(table.to_string(index=False))
    # Annotate argmax per row (no aggregation, just for readability):
    m_cols = [f"m={m}" for m in M_GRID]
    table["argmax_m"] = table[m_cols].idxmax(axis=1).str.removeprefix("m=").astype(int)
    return table


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    val_neigh = load_val_neighbors()
    val_fire = fire_rates(val_neigh)
    val_table = report("VALIDATION (10 queries/label/project, 270 retrieval pool)", val_fire)
    val_table.to_csv(OUT_DIR / "per_project_margin_val.csv", index=False)

    test_neigh = load_test_neighbors()
    test_fire = fire_rates(test_neigh)
    test_table = report("TEST (300 queries/project, 300 retrieval pool)", test_fire)
    test_table.to_csv(OUT_DIR / "per_project_margin_test.csv", index=False)

    print(f"\nWrote: {OUT_DIR / 'per_project_margin_val.csv'}")
    print(f"Wrote: {OUT_DIR / 'per_project_margin_test.csv'}")


if __name__ == "__main__":
    main()

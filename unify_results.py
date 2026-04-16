#!/usr/bin/env python3
"""
unify_results.py
================
Merges per-context-window result CSVs into single unified files,
deduplicates, and writes to results/.

Outputs:
  results/unified_performance.csv  — all_results across ctx windows
  results/unified_cost.csv         — all_cost_metrics across ctx windows
"""

import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
CTX_DIRS = sorted(RESULTS_DIR.glob("issues3k_ctx*"))


def load_and_tag(csv_name: str, ctx_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for d in ctx_dirs:
        f = d / csv_name
        if not f.exists():
            continue
        ctx = int(d.name.split("_ctx")[1])
        df = pd.read_csv(f)
        df["context_window"] = ctx
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No {csv_name} found in {ctx_dirs}")
    return pd.concat(frames, ignore_index=True)


def dedup(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"  Removed {before - after} duplicate rows ({before} → {after})")
    else:
        print(f"  No duplicates ({after} rows)")
    return df


def main():
    print(f"Found context dirs: {[d.name for d in CTX_DIRS]}")

    # --- Performance ---
    print("\nPerformance metrics:")
    perf = load_and_tag("all_results.csv", CTX_DIRS)
    perf = dedup(perf)
    perf = perf.sort_values(["context_window", "approach", "model", "top_k"]).reset_index(drop=True)

    out = RESULTS_DIR / "unified_performance.csv"
    perf.to_csv(out, index=False)
    print(f"  Written to {out}")
    print(f"  Shape: {perf.shape}")
    print(f"  Context windows: {sorted(perf['context_window'].unique())}")
    print(f"  Approaches: {sorted(perf['approach'].unique())}")
    print(f"  Models: {perf['model'].nunique()}")

    # --- Cost ---
    print("\nCost metrics:")
    cost = load_and_tag("all_cost_metrics.csv", CTX_DIRS)
    cost = dedup(cost)
    cost = cost.sort_values(["context_window", "approach", "model", "top_k"]).reset_index(drop=True)

    out = RESULTS_DIR / "unified_cost.csv"
    cost.to_csv(out, index=False)
    print(f"  Written to {out}")
    print(f"  Shape: {cost.shape}")
    print(f"  Context windows: {sorted(cost['context_window'].unique())}")

    # --- Quick sanity check ---
    print("\n--- Sanity check ---")
    ragtag = perf[perf["approach"] == "ragtag"]
    print(f"RAGTAG rows: {len(ragtag)}")
    for ctx in sorted(ragtag["context_window"].unique()):
        sub = ragtag[ragtag["context_window"] == ctx]
        combos = sub.groupby(["model", "top_k"]).size()
        dupes = combos[combos > 1]
        if len(dupes):
            print(f"  ctx={ctx}: WARNING — {len(dupes)} duplicate model×k combos remain!")
        else:
            print(f"  ctx={ctx}: {len(sub)} rows, {sub['model'].nunique()} models × {sub['top_k'].nunique()} k values — clean")

    ft = perf[perf["approach"] != "ragtag"]
    if len(ft):
        print(f"Fine-tune rows: {len(ft)} ({sorted(ft['approach'].unique())})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Conditional-J margin analysis on Qwen-3B PS using existing RAGTAG predictions.

For each (k, m), compute:
  TPR_q_err(m, k)     = P(fire | true=q AND RAGTAG_pred=bug)
  FPR_b_correct(m, k) = P(fire | true=b AND RAGTAG_pred=bug)
  J_cond(m, k)        = TPR_q_err - FPR_b_correct

Aggregates pooled over the 11 PS projects.

NOTE: this uses the test split (peek). It is a diagnostic to see if the
conditional refinement gives a cleaner argmax than retrieval-only J. If it
does, the paper-grade version should re-run on the held-out validation
split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PS_BASE = REPO_ROOT / "results" / "issues11k" / "project_specific"
MODEL_TAG = "unsloth_Qwen2_5_3B_Instruct_bnb_4bit"

KS = [1, 3, 6, 9, 12, 15]
M_GRID = [1, 2, 3, 4, 5]


def cum_counts(neigh_df: pd.DataFrame) -> dict[tuple[str, int], dict]:
    """For each (repo, test_idx) return per-rank cumulative bug/question counts."""
    out = {}
    neigh_df = neigh_df.sort_values(["repo", "test_idx", "neighbor_rank"])
    for (repo, qidx), g in neigh_df.groupby(["repo", "test_idx"], sort=False):
        labels = g["neighbor_label"].to_numpy()
        out[(repo, qidx)] = {
            "true_label": g["test_label"].iloc[0],
            "cum_bug": np.cumsum(labels == "bug"),
            "cum_q": np.cumsum(labels == "question"),
        }
    return out


def main() -> None:
    # Load neighbors (k≤15) and RAGTAG preds for each project
    neigh_parts = []
    ragtag_parts = {}
    for proj_dir in sorted(PS_BASE.iterdir()):
        if not proj_dir.is_dir():
            continue
        repo = proj_dir.name
        # neighbors
        nf = proj_dir / "neighbors" / "neighbors_k30.csv"
        n = pd.read_csv(nf, usecols=["test_idx", "test_label", "neighbor_rank", "neighbor_label"])
        n = n[n["neighbor_rank"] < 15].copy()
        n["repo"] = repo
        neigh_parts.append(n)
        # RAGTAG predictions for each k
        for k in KS:
            f = proj_dir / MODEL_TAG / "ragtag" / "predictions" / f"preds_k{k}.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f, usecols=["test_idx", "ground_truth", "predicted_label"])
            df["repo"] = repo
            ragtag_parts.setdefault(k, []).append(df)

    neigh = pd.concat(neigh_parts, ignore_index=True)
    cums = cum_counts(neigh)

    rows = []
    for k in KS:
        if k not in ragtag_parts:
            continue
        rag = pd.concat(ragtag_parts[k], ignore_index=True)
        rag["true_label"] = rag["ground_truth"].astype(str).str.lower().str.strip()
        rag["pred_label"] = rag["predicted_label"].astype(str).str.lower().str.strip()
        # Subset queries
        target_q_err = rag[(rag["true_label"] == "question") & (rag["pred_label"] == "bug")]
        at_risk_b = rag[(rag["true_label"] == "bug") & (rag["pred_label"] == "bug")]

        # Also the unconditional versions for sanity
        all_q = rag[rag["true_label"] == "question"]
        all_b = rag[rag["true_label"] == "bug"]

        for m in M_GRID:
            def fire_rate(subset):
                fired = 0; n = 0
                for _, r in subset.iterrows():
                    rec = cums.get((r["repo"], int(r["test_idx"])))
                    if rec is None:
                        continue
                    if int(rec["cum_bug"][k - 1] - rec["cum_q"][k - 1]) <= m:
                        fired += 1
                    n += 1
                return (fired / n) if n else 0.0, n

            tpr_err, n_err = fire_rate(target_q_err)
            fpr_corr, n_corr = fire_rate(at_risk_b)
            tpr_all, n_all_q = fire_rate(all_q)
            fpr_all, n_all_b = fire_rate(all_b)

            rows.append({
                "k": k, "m": m,
                "TPR_q_err": tpr_err, "n_q_err": n_err,
                "FPR_b_correct": fpr_corr, "n_b_correct": n_corr,
                "J_cond": tpr_err - fpr_corr,
                "TPR_q_all": tpr_all,
                "FPR_b_all": fpr_all,
                "J_uncond": tpr_all - fpr_all,
            })

    out = pd.DataFrame(rows)
    out_path = REPO_ROOT / "results" / "bragtag_margin_validation" / "analysis" / "qwen3b_cond_j.csv"
    out.to_csv(out_path, index=False)

    # Print per-k table and the k-averaged table
    print("=== Per-(k, m): conditional vs unconditional J on Qwen-3B PS ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    avg = (
        out.groupby("m")
        .agg(
            mean_TPR_q_err=("TPR_q_err", "mean"),
            mean_FPR_b_correct=("FPR_b_correct", "mean"),
            mean_J_cond=("J_cond", "mean"),
            mean_TPR_q_all=("TPR_q_all", "mean"),
            mean_FPR_b_all=("FPR_b_all", "mean"),
            mean_J_uncond=("J_uncond", "mean"),
        )
        .reset_index()
    )
    print("\n=== Averaged across k=1..15 ===")
    print(avg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

"""Significance tests: BRAGTAG vs RAGTAG at each model's best k (PS, raw).

Reports for each Qwen size:
  1. Paired bootstrap 95% CI on macro F1 difference (BRAGTAG - RAGTAG)
  2. McNemar's test on accuracy disagreements
  3. Per-project Wilcoxon signed-rank test (robustness check)

Convention: pooled raw, see paper/sections/04_setup.tex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "issues11k"

LABELS = ["bug", "feature", "question"]
KS = [1, 3, 6, 9, 12, 15]
MODELS = [
    ("unsloth_Qwen2_5_3B_Instruct_bnb_4bit",  "Qwen-3B"),
    ("unsloth_Qwen2_5_7B_Instruct_bnb_4bit",  "Qwen-7B"),
    ("unsloth_Qwen2_5_14B_Instruct_bnb_4bit", "Qwen-14B"),
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "Qwen-32B"),
]
N_BOOTSTRAP = 1000
RNG_SEED = 42


def _macro(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)


def _load_with_proj(model: str, k: int, sub: str) -> pd.DataFrame:
    projects = sorted(p.name for p in (RESULTS / "project_specific").iterdir() if p.is_dir())
    parts = []
    for proj in projects:
        d = pd.read_csv(RESULTS / f"project_specific/{proj}/{model}/{sub}/predictions/preds_k{k}.csv",
                        usecols=["test_idx", "ground_truth", "predicted_label"])
        d["project"] = proj
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _best_k(model: str, sub: str) -> int:
    return max(KS, key=lambda k: _macro(
        _load_with_proj(model, k, sub)["ground_truth"],
        _load_with_proj(model, k, sub)["predicted_label"]))


def _bootstrap_diff_ci(y_true, yp_v, yp_b, n=N_BOOTSTRAP, seed=RNG_SEED):
    """Paired bootstrap CI on macro F1 difference (BRAGTAG - RAGTAG)."""
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        diffs[i] = (_macro(y_true[idx], yp_b[idx])
                    - _macro(y_true[idx], yp_v[idx]))
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def _mcnemar_test(y_true, yp_v, yp_b):
    """McNemar's test on accuracy disagreements."""
    v_correct = (yp_v == y_true)
    b_correct = (yp_b == y_true)
    table = [
        [int((v_correct & b_correct).sum()), int((v_correct & ~b_correct).sum())],
        [int((~v_correct & b_correct).sum()), int((~v_correct & ~b_correct).sum())],
    ]
    res = mcnemar(table, exact=False, correction=True)
    return table, res.statistic, res.pvalue


def _per_project_wilcoxon(merged: pd.DataFrame):
    """Wilcoxon signed-rank on per-project macro F1 differences."""
    project_diffs = []
    for proj, g in merged.groupby("project"):
        v = _macro(g["ground_truth"].values, g["predicted_label_v"].values)
        b = _macro(g["ground_truth"].values, g["predicted_label_b"].values)
        project_diffs.append(b - v)
    project_diffs = np.array(project_diffs)
    res = wilcoxon(project_diffs, alternative="greater")
    return project_diffs, res.statistic, res.pvalue


def main():
    print("=" * 110)
    print(f"{'Model':<10}  {'V k*':<5}  {'B k*':<5}  "
          f"{'mean diff':>10}  {'95% CI':<22}  {'McNemar p':>11}  "
          f"{'Wilcoxon p':>11}  {'V/B/=':<14}")
    print("=" * 110)
    for model, lbl in MODELS:
        # Best k for each method
        v_k = _best_k(model, "ragtag")
        b_k = _best_k(model, "ragtag_debias_m3")

        # Load and align
        v = _load_with_proj(model, v_k, "ragtag")
        b = _load_with_proj(model, b_k, "ragtag_debias_m3")
        merged = v.merge(b, on=["project", "test_idx", "ground_truth"], suffixes=("_v", "_b"))
        assert len(merged) == 3300

        y_true = merged["ground_truth"].values
        yp_v = merged["predicted_label_v"].values
        yp_b = merged["predicted_label_b"].values

        # Bootstrap CI
        mean_d, lo, hi = _bootstrap_diff_ci(y_true, yp_v, yp_b)

        # McNemar's
        table, stat, pval = _mcnemar_test(y_true, yp_v, yp_b)
        v_only = table[0][1]
        b_only = table[1][0]
        both_correct = table[0][0]

        # Per-project Wilcoxon (one-sided: BRAGTAG > RAGTAG)
        proj_diffs, w_stat, w_p = _per_project_wilcoxon(merged)
        n_pos = int((proj_diffs > 0).sum())
        n_neg = int((proj_diffs < 0).sum())

        ci_str = f"[+{lo:.4f}, +{hi:.4f}]" if lo > 0 else f"[{lo:+.4f}, {hi:+.4f}]"
        sig_marker = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        print(f"{lbl:<10}  {v_k:<5}  {b_k:<5}  "
              f"{mean_d:>+10.4f}  {ci_str:<22}  {pval:>10.2e}{sig_marker}  "
              f"{w_p:>10.2e}  V={v_only}/B={b_only}/=={both_correct}")

    print()
    print("Bootstrap: paired, 1000 resamples, percentile 95% CI on (BRAGTAG - RAGTAG) macro F1.")
    print("McNemar: continuity-corrected, on V-only-correct vs B-only-correct counts.")
    print("Wilcoxon: per-project paired (n=11), one-sided (BRAGTAG > RAGTAG).")
    print("Significance markers: * p<0.05, ** p<0.01, *** p<0.001.")


if __name__ == "__main__":
    main()

"""Significance tests: \\bragtag-PS vs Fine-Tune-PA at each model's best config.

Tests whether the §5.5 alternation pattern is statistically significant.
Both methods are evaluated with their scope-matched \\votag\\ fallback applied
(\\votag-PS for \\bragtag, \\votag-PA for fine-tune), consistent with
[`tables/method_comparison.tex`](paper/tables/method_comparison.tex).

Pairing methodology:
  - Fine-Tune-PA predictions are stored in a single agnostic file with global
    test_idx 0..3299.
  - \\bragtag-PS predictions are stored per-project with local test_idx 0..299.
  - We use [`results/issues11k/agnostic/neighbors/test_split.csv`](results/issues11k/agnostic/neighbors/test_split.csv)
    `repo` column as the canonical agnostic ordering, then for each global
    index we look up the corresponding (project, local_test_idx) by counting
    project occurrences as we walk.
  - Paired bootstrap and McNemar both use this aligned 3300-row pairing.

Reports per Qwen size:
  1. Paired bootstrap 95% CI on (BRAGTAG - Fine-Tune) macro F1 difference (1000 resamples)
  2. McNemar's test (continuity-corrected) on accuracy disagreements
  3. TOST equivalence at delta=0.02 and delta=0.05 (practical-equivalence margins)

And one aggregate test across all 4 models (4 x 3300 = 13,200 paired
predictions concatenated): aggregate paired bootstrap 95% CI on (BRAGTAG -
Fine-Tune) macro F1 difference, plus aggregate TOST.

Convention: pooled. \\votag-rescue is §5.5-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rescue import RESULTS, VTAG_BEST_K_PA, VTAG_BEST_K_PS, _project_list  # noqa: E402

LABELS = ["bug", "feature", "question"]
LABEL_SET = set(LABELS)
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


def _per_project_bragtag_rescued(model: str, k: int) -> dict[str, pd.DataFrame]:
    """For each project, load BRAGTAG-PS predictions at k and rescue invalid with
    \\votag-PS at its best k. Returns dict[project] -> df indexed by test_idx."""
    out = {}
    for proj in _project_list():
        rag_path = (RESULTS / "project_specific" / proj / model / "ragtag_debias_m3"
                    / "predictions" / f"preds_k{k}.csv")
        df = pd.read_csv(rag_path, usecols=["test_idx", "ground_truth", "predicted_label"])
        vtag_path = (RESULTS / "project_specific" / proj / "vtag" / "predictions"
                     / f"preds_k{VTAG_BEST_K_PS}.csv")
        vtg = pd.read_csv(vtag_path, usecols=["test_idx", "predicted_label"]).set_index("test_idx")
        inv = df["predicted_label"] == "invalid"
        if inv.any():
            df.loc[inv, "predicted_label"] = df.loc[inv, "test_idx"].map(vtg["predicted_label"])
        out[proj] = df.set_index("test_idx")
    return out


def _ft_pa_rescued(model: str) -> pd.DataFrame:
    """Fine-Tune-PA rescued with \\votag-PA. Single file, global test_idx 0..3299."""
    f = RESULTS / "agnostic" / model / "finetune_fixed" / "preds_finetune_fixed.csv"
    ft = pd.read_csv(f, usecols=["test_idx", "ground_truth", "predicted_label"])
    inv = ~ft["predicted_label"].isin(LABEL_SET)
    if inv.any():
        vtg = pd.read_csv(
            RESULTS / "agnostic" / "vtag" / "predictions" / f"preds_k{VTAG_BEST_K_PA}.csv",
            usecols=["test_idx", "predicted_label"],
        ).set_index("test_idx")
        ft.loc[inv, "predicted_label"] = ft.loc[inv, "test_idx"].map(vtg["predicted_label"])
    return ft.sort_values("test_idx").reset_index(drop=True)


def _aligned_pair(model: str, bragtag_k: int):
    """Return (y_true, yp_bragtag, yp_ft) arrays of length 3300, aligned per
    issue via the agnostic test ordering."""
    test_split = pd.read_csv(
        RESULTS / "agnostic" / "neighbors" / "test_split.csv",
        usecols=["repo"],
    )
    proj_tags = test_split["repo"].str.replace("/", "_", n=1).tolist()
    if len(proj_tags) != 3300:
        raise RuntimeError(f"unexpected agnostic test split length: {len(proj_tags)}")

    bragtag_per_proj = _per_project_bragtag_rescued(model, bragtag_k)
    ft = _ft_pa_rescued(model)
    if len(ft) != 3300:
        raise RuntimeError(f"unexpected FT-PA length: {len(ft)}")

    # Walk agnostic order; for each global_idx find (proj, local_idx).
    local_counter = {p: 0 for p in bragtag_per_proj}
    bragtag_labels = []
    bragtag_gt = []
    for g_idx, proj in enumerate(proj_tags):
        local_idx = local_counter[proj]
        local_counter[proj] += 1
        row = bragtag_per_proj[proj].loc[local_idx]
        bragtag_labels.append(row["predicted_label"])
        bragtag_gt.append(row["ground_truth"])

    bragtag_gt = np.array(bragtag_gt)
    yp_b = np.array(bragtag_labels)
    yp_ft = ft["predicted_label"].values
    y_true_ft = ft["ground_truth"].values

    # Sanity: ground truth must be identical between the two views, per row.
    if not (bragtag_gt == y_true_ft).all():
        raise RuntimeError("ground truth misalignment between BRAGTAG-PS and FT-PA pairings")

    return y_true_ft, yp_b, yp_ft


def _bootstrap_diff_ci(y_true, yp_b, yp_ft, n=N_BOOTSTRAP, seed=RNG_SEED):
    """Paired bootstrap CI on macro F1 difference (BRAGTAG - Fine-Tune)."""
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        diffs[i] = (_macro(y_true[idx], yp_b[idx])
                    - _macro(y_true[idx], yp_ft[idx]))
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def _mcnemar_test(y_true, yp_b, yp_ft):
    """McNemar's test on accuracy disagreements."""
    b_correct = (yp_b == y_true)
    ft_correct = (yp_ft == y_true)
    table = [
        [int((b_correct & ft_correct).sum()), int((b_correct & ~ft_correct).sum())],
        [int((~b_correct & ft_correct).sum()), int((~b_correct & ~ft_correct).sum())],
    ]
    res = mcnemar(table, exact=False, correction=True)
    return table, res.statistic, res.pvalue


def _best_k_bragtag_raw(model: str) -> int:
    """Best k for BRAGTAG-PS on raw (no-rescue) predictions, consistent with
    table generation."""
    best_k, best_macro = None, -1.0
    for k in KS:
        parts = []
        for proj in _project_list():
            df = pd.read_csv(
                RESULTS / "project_specific" / proj / model / "ragtag_debias_m3"
                / "predictions" / f"preds_k{k}.csv",
                usecols=["ground_truth", "predicted_label"])
            parts.append(df)
        pooled = pd.concat(parts, ignore_index=True)
        m = _macro(pooled["ground_truth"], pooled["predicted_label"])
        if m > best_macro:
            best_macro, best_k = m, k
    return best_k


def _tost(lo: float, hi: float, delta: float) -> str:
    """Return 'PASS' if [lo, hi] is entirely within [-delta, +delta], else FAIL."""
    return "PASS" if (lo > -delta and hi < +delta) else "FAIL"


def main():
    print("=" * 130)
    print(f"{'Model':<10}  {'B k*':<5}  "
          f"{'mean diff':>10}  {'95% CI':<24}  {'McNemar p':>11}  "
          f"{'TOST 0.02':>10}  {'TOST 0.05':>10}")
    print("=" * 130)

    aggregate_y, aggregate_b, aggregate_ft = [], [], []
    per_model_diffs = []

    for model, lbl in MODELS:
        b_k = _best_k_bragtag_raw(model)
        y_true, yp_b, yp_ft = _aligned_pair(model, b_k)

        mean_d, lo, hi = _bootstrap_diff_ci(y_true, yp_b, yp_ft)
        table, stat, pval = _mcnemar_test(y_true, yp_b, yp_ft)

        ci_str = f"[{lo:+.4f}, {hi:+.4f}]"
        sig_marker = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        tost_2 = _tost(lo, hi, 0.02)
        tost_5 = _tost(lo, hi, 0.05)

        print(f"{lbl:<10}  {b_k:<5}  "
              f"{mean_d:>+10.4f}  {ci_str:<24}  {pval:>10.2e}{sig_marker}  "
              f"{tost_2:>10}  {tost_5:>10}")

        aggregate_y.append(y_true)
        aggregate_b.append(yp_b)
        aggregate_ft.append(yp_ft)
        per_model_diffs.append(mean_d)

    print("=" * 130)

    # Aggregate test: concatenate all 4 models' paired predictions.
    y_all = np.concatenate(aggregate_y)
    b_all = np.concatenate(aggregate_b)
    ft_all = np.concatenate(aggregate_ft)
    agg_mean, agg_lo, agg_hi = _bootstrap_diff_ci(y_all, b_all, ft_all)
    agg_table, _, agg_p = _mcnemar_test(y_all, b_all, ft_all)
    agg_tost_2 = _tost(agg_lo, agg_hi, 0.02)
    agg_tost_5 = _tost(agg_lo, agg_hi, 0.05)
    print(f"{'AGGREGATE':<10}  {'--':<5}  "
          f"{agg_mean:>+10.4f}  [{agg_lo:+.4f}, {agg_hi:+.4f}]  "
          f"{agg_p:>10.2e}      "
          f"{agg_tost_2:>10}  {agg_tost_5:>10}")
    print(f"             (n={len(y_all)} paired predictions across the 4 model sizes)")

    print()
    print("Bootstrap: paired, 1000 resamples on aligned per-model 3300-row pairing.")
    print("           Percentile 95% CI on (BRAGTAG-PS - Fine-Tune-PA) macro F1.")
    print("McNemar:   continuity-corrected, on accuracy disagreement counts.")
    print("TOST:      two one-sided tests for equivalence at margin delta.")
    print("           PASS = the 95% CI lies entirely within [-delta, +delta].")
    print("           delta=0.02: tight equivalence; delta=0.05: loose practical equivalence.")
    print("Both methods include the scope-matched VTAG fallback (PS for BRAGTAG, PA for Fine-Tune).")
    print(f"Mean of per-model diffs: {sum(per_model_diffs)/len(per_model_diffs):+.4f}")


if __name__ == "__main__":
    main()

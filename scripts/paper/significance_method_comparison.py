"""Paired bootstrap CIs: retrieval-based methods vs Fine-Tune-PA (§5.5, RQ4).

Comparisons (retrieval method minus fine-tuning, so negative values favour
fine-tuning), each retrieval method at its best k on raw pooled macro F1, as in
[`tables/method_comparison_ext.tex`](SANER2027/tables/method_comparison_ext.tex):
  1. \\ragtag-PS  - Fine-Tune-PA, raw predictions (invalid outputs count as wrong)
  2. \\bragtag-PS - Fine-Tune-PA, raw predictions
  3. \\ragtag-PS  - Fine-Tune-PA, both with their scope-matched \\votag\\ fallback
     (\\votag-PS for \\ragtag, \\votag-PA for fine-tune)
  4. \\bragtag-PS - Fine-Tune-PA, both with the fallback

Output: the console report below and paper/tables/method_comparison_ci.csv
(method, protocol, model, k, delta, ci_lo, ci_hi; "All" rows are the mean of the
per-model differences), which scripts/paper/tab_method_comparison_ci.py turns into
SANER2027/tables/method_comparison_ci.tex. The CSV is only written on a full
four-model run.

Pairing methodology:
  - Fine-Tune-PA predictions are stored in a single agnostic file with global
    test_idx 0..3299.
  - PS predictions are stored per-project with local test_idx 0..299.
  - We use [`results/issues11k/agnostic/neighbors/test_split.csv`](results/issues11k/agnostic/neighbors/test_split.csv)
    `repo` column as the canonical agnostic ordering, then for each global
    index we look up the corresponding (project, local_test_idx) by counting
    project occurrences as we walk.

Resampling unit = issue. Every model size labels the same 3,300 test issues, so
the model sizes' predictions for one issue are repeated observations of that
issue, not independent rows. Each of the 1,000 resamples draws one set of 3,300
issue indices and applies it to every model:
  - per model: percentile 95% CI on that model's macro F1 difference;
  - aggregate: the mean of the per-model macro F1 differences, with its CI taken
    from the same resamples, so an issue's predictions stay together.

Scope: the CIs describe sampling variability over issues from these eleven
projects. They do not quantify generalization to other projects.

Usage (lab machine; RESULTS_DIR overrides the results path, see _rescue.py):
  venv/bin/python scripts/paper/significance_method_comparison.py
  venv/bin/python scripts/paper/significance_method_comparison.py --models Qwen-32B

Convention: pooled. \\votag-rescue is §5.5-only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rescue import RESULTS, VTAG_BEST_K_PA, VTAG_BEST_K_PS, _project_list  # noqa: E402

LABELS = ["bug", "feature", "question"]
LABEL_SET = set(LABELS)
KS = [1, 3, 6, 9, 12, 15]
N_TEST = 3300

MODELS = [
    ("unsloth_Qwen2_5_3B_Instruct_bnb_4bit",  "Qwen-3B"),
    ("unsloth_Qwen2_5_7B_Instruct_bnb_4bit",  "Qwen-7B"),
    ("unsloth_Qwen2_5_14B_Instruct_bnb_4bit", "Qwen-14B"),
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "Qwen-32B"),
]
# (label, retrieval prediction set, fine-tune prediction set, best-k source)
COMPARISONS = [
    ("RAGTAG-PS - FT-PA (raw)",             "ragtag",    "ft",   "ragtag"),
    ("BRAGTAG-PS - FT-PA (raw)",            "bragtag",   "ft",   "bragtag"),
    ("RAGTAG-PS - FT-PA (+VTAG fallback)",  "ragtag_v",  "ft_v", "ragtag"),
    ("BRAGTAG-PS - FT-PA (+VTAG fallback)", "bragtag_v", "ft_v", "bragtag"),
]
CSV_OUT = Path(__file__).resolve().parents[2] / "paper" / "tables" / "method_comparison_ci.csv"
APPROACH_DIR = {"ragtag": "ragtag", "bragtag": "ragtag_debias_m3"}
N_BOOTSTRAP = 1000
RNG_SEED = 42


def _macro(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)


def _ps_preds(model: str, approach: str, k: int, rescue: bool) -> dict[str, pd.DataFrame]:
    """Per-project PS predictions at k, indexed by local test_idx. With rescue,
    invalid outputs take \\votag-PS's label at its best k."""
    out = {}
    for proj in _project_list():
        df = pd.read_csv(RESULTS / "project_specific" / proj / model / approach
                         / "predictions" / f"preds_k{k}.csv",
                         usecols=["test_idx", "ground_truth", "predicted_label"])
        if rescue:
            vtg = pd.read_csv(RESULTS / "project_specific" / proj / "vtag" / "predictions"
                              / f"preds_k{VTAG_BEST_K_PS}.csv",
                              usecols=["test_idx", "predicted_label"]).set_index("test_idx")
            inv = df["predicted_label"] == "invalid"
            if inv.any():
                df.loc[inv, "predicted_label"] = df.loc[inv, "test_idx"].map(vtg["predicted_label"])
        out[proj] = df.set_index("test_idx")
    return out


def _ft_pa(model: str, rescue: bool) -> pd.DataFrame:
    """Fine-Tune-PA, optionally rescued with \\votag-PA. Single file, global test_idx 0..3299."""
    f = RESULTS / "agnostic" / model / "finetune_fixed" / "preds_finetune_fixed.csv"
    ft = pd.read_csv(f, usecols=["test_idx", "ground_truth", "predicted_label"])
    inv = ~ft["predicted_label"].isin(LABEL_SET)
    if rescue and inv.any():
        vtg = pd.read_csv(
            RESULTS / "agnostic" / "vtag" / "predictions" / f"preds_k{VTAG_BEST_K_PA}.csv",
            usecols=["test_idx", "predicted_label"],
        ).set_index("test_idx")
        ft.loc[inv, "predicted_label"] = ft.loc[inv, "test_idx"].map(vtg["predicted_label"])
    return ft.sort_values("test_idx").reset_index(drop=True)


def _best_k_raw(model: str, approach: str) -> int:
    """Best k for a PS retrieval method on raw (no-rescue) pooled macro F1,
    consistent with table generation."""
    best_k, best_macro = None, -1.0
    for k in KS:
        pooled = pd.concat([
            pd.read_csv(RESULTS / "project_specific" / proj / model / approach
                        / "predictions" / f"preds_k{k}.csv",
                        usecols=["ground_truth", "predicted_label"])
            for proj in _project_list()], ignore_index=True)
        m = _macro(pooled["ground_truth"], pooled["predicted_label"])
        if m > best_macro:
            best_macro, best_k = m, k
    return best_k


def _agnostic_order(per_proj: dict[str, pd.DataFrame], proj_tags: list[str]):
    """Walk the agnostic order; return (ground_truth, predicted_label) arrays."""
    local_counter = {p: 0 for p in per_proj}
    gt, pred = [], []
    for proj in proj_tags:
        row = per_proj[proj].loc[local_counter[proj]]
        local_counter[proj] += 1
        gt.append(row["ground_truth"])
        pred.append(row["predicted_label"])
    return np.array(gt), np.array(pred)


def _aligned(model: str):
    """Return (y_true, {prediction set: array}, {approach: best k}) for one
    model, every array of length 3300 in agnostic order."""
    test_split = pd.read_csv(RESULTS / "agnostic" / "neighbors" / "test_split.csv",
                             usecols=["repo"])
    proj_tags = test_split["repo"].str.replace("/", "_", n=1).tolist()
    if len(proj_tags) != N_TEST:
        raise RuntimeError(f"unexpected agnostic test split length: {len(proj_tags)}")

    best_k = {a: _best_k_raw(model, APPROACH_DIR[a]) for a in APPROACH_DIR}
    preds, truths = {}, {}
    for name, approach, rescue in [("ragtag", "ragtag", False),
                                   ("bragtag", "bragtag", False),
                                   ("ragtag_v", "ragtag", True),
                                   ("bragtag_v", "bragtag", True)]:
        per_proj = _ps_preds(model, APPROACH_DIR[approach], best_k[approach], rescue)
        truths[name], preds[name] = _agnostic_order(per_proj, proj_tags)
    for name, rescue in [("ft", False), ("ft_v", True)]:
        ft = _ft_pa(model, rescue)
        if len(ft) != N_TEST:
            raise RuntimeError(f"unexpected FT-PA length: {len(ft)}")
        truths[name], preds[name] = ft["ground_truth"].values, ft["predicted_label"].values

    # Sanity: ground truth must be identical across all views, per row.
    y_true = truths["ft"]
    for name, gt in truths.items():
        if not (gt == y_true).all():
            raise RuntimeError(f"ground truth misalignment between {name} and FT-PA pairings")
    return y_true, preds, best_k


def _bootstrap(aligned: dict, n=N_BOOTSTRAP, seed=RNG_SEED) -> dict[str, np.ndarray]:
    """Issue-level paired bootstrap. Returns {comparison: (n, n_models) array of
    resampled macro F1 differences}; column j is the j-th model in `aligned`."""
    rng = np.random.default_rng(seed)
    sets = sorted({s for _, a, b, _ in COMPARISONS for s in (a, b)})
    out = {c: np.empty((n, len(aligned))) for c, _, _, _ in COMPARISONS}
    for i in range(n):
        idx = rng.integers(0, N_TEST, N_TEST)  # one draw of issues, shared by every model
        for j, (y_true, preds, _) in enumerate(aligned.values()):
            f1 = {s: _macro(y_true[idx], preds[s][idx]) for s in sets}
            for c, a, b, _ in COMPARISONS:
                out[c][i, j] = f1[a] - f1[b]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--models", nargs="+", default=[lbl for _, lbl in MODELS],
                        help="Model labels to include (default: all four).")
    parser.add_argument("--csv", type=Path, default=CSV_OUT,
                        help="CSV for tab_method_comparison_ci.py (written only on a full run).")
    args = parser.parse_args()
    models = [(tag, lbl) for tag, lbl in MODELS if lbl in args.models]
    if len(models) != len(args.models):
        parser.error(f"unknown model label in {args.models}")

    aligned = {lbl: _aligned(tag) for tag, lbl in models}
    boot = _bootstrap(aligned)

    rows = []  # (method, protocol, model, k, delta, lo, hi) for the CSV
    print("=" * 84)
    print(f"{'Comparison':<38}  {'Model':<10}  {'k*':>3}  {'diff':>8}  95% CI")
    print("=" * 84)
    for c, a, b, k_src in COMPARISONS:
        protocol = "fallback" if a.endswith("_v") else "raw"
        diffs = []
        for j, (lbl, (y_true, preds, best_k)) in enumerate(aligned.items()):
            d = _macro(y_true, preds[a]) - _macro(y_true, preds[b])
            diffs.append(d)
            lo, hi = np.percentile(boot[c][:, j], [2.5, 97.5])
            rows.append((k_src, protocol, lbl, best_k[k_src], d, lo, hi))
            print(f"{c if j == 0 else '':<38}  {lbl:<10}  {best_k[k_src]:>3}  "
                  f"{d:>+8.4f}  [{lo:+.4f}, {hi:+.4f}]")
        if len(aligned) > 1:
            lo, hi = np.percentile(boot[c].mean(axis=1), [2.5, 97.5])
            rows.append((k_src, protocol, "All", "-", float(np.mean(diffs)), lo, hi))
            print(f"{'':<38}  {f'mean ({len(aligned)})':<10}  {'':>3}  "
                  f"{np.mean(diffs):>+8.4f}  [{lo:+.4f}, {hi:+.4f}]")
        print("-" * 84)

    if len(models) == len(MODELS):
        with open(args.csv, "w") as fh:
            fh.write("# Paired bootstrap 95% CIs on macro-F1 differences (retrieval method minus PA fine-tuning),\n"
                     f"# {N_BOOTSTRAP} issue-level resamples shared across model sizes (seed {RNG_SEED}).\n"
                     "# protocol raw: invalid outputs count as incorrect; fallback: VOTAG's vote replaces them\n"
                     "# (VOTAG-PS k=15 for RAGTAG/BRAGTAG, VOTAG-PA k=16 for fine-tuning). All = mean of the four\n"
                     "# per-model differences. Written by scripts/paper/significance_method_comparison.py.\n"
                     "method,protocol,model,k,delta,ci_lo,ci_hi\n")
            for method, protocol, lbl, k, d, lo, hi in rows:
                fh.write(f"{method},{protocol},{lbl},{k},{d:.5f},{lo:.5f},{hi:.5f}\n")
        print(f"wrote {args.csv}")
    else:
        print(f"(partial run: {args.csv} not written)")

    print()
    print(f"Bootstrap: paired, {N_BOOTSTRAP} resamples of the {N_TEST:,} test issues (seed {RNG_SEED});")
    print("           each resample is shared by every model, so the aggregate (mean of the")
    print("           per-model differences) keeps an issue's predictions together.")
    print("           Percentile 95% CIs on (retrieval method - Fine-Tune-PA) macro F1.")
    print("Scope:     issues from these eleven projects; not generalization to other projects.")
    print("Fallback:  scope-matched VTAG (PS for RAGTAG/BRAGTAG, PA for Fine-Tune), comparisons 3-4.")


if __name__ == "__main__":
    main()

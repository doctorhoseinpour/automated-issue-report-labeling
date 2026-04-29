"""RQ2.6 + RQ2.7: Confusion matrices, bug-bias quantification, and
retrieval-side bug skew.

RQ2.6 — for each (model, approach, k, setting) cell, compute over-prediction
ratio per class (predicted_count / true_count). Show that bug is consistently
over-predicted for plain RAGTAG and zero-shot.

RQ2.7 — from neighbors CSVs, compute fraction of top-k neighbors that are
bugs, sliced by ground-truth class. Establishes the retrieval-side mechanism
for bug-bias.

Outputs:
  docs/analysis/rq2_overprediction.csv
  docs/analysis/rq2_retrieval_bug_skew.csv
  docs/analysis/figures/rq2_confusion_qwen32b.png
  docs/analysis/figures/rq2_bug_overprediction.png
  docs/analysis/figures/rq2_retrieval_bug_skew.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _utils import (  # noqa: E402
    DOCS_ANALYSIS,
    FIGURES,
    MODEL_ORDER,
    PROJECTS,
    REPO_ROOT,
    RESULTS_DIR,
    VALID_LABELS,
    ensure_dirs,
    load_predictions,
    rel,
)


def _confusion(y_true: list[str], y_pred: list[str]) -> np.ndarray:
    """3x3 confusion matrix in [bug, feature, question] order; counts include
    'invalid' as a 4th column for diagnostics."""
    labels = VALID_LABELS
    yt = [str(x).lower().strip() for x in y_true]
    yp = [str(x).lower().strip() for x in y_pred]
    cm = np.zeros((len(labels), len(labels) + 1), dtype=int)  # extra col for invalid
    for t, p in zip(yt, yp):
        if t not in labels:
            continue
        ti = labels.index(t)
        if p == "invalid" or p not in labels:
            cm[ti, len(labels)] += 1
        else:
            pi = labels.index(p)
            cm[ti, pi] += 1
    return cm


def _overpred_ratios(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """For each class: predicted_count / true_count. Excludes invalid from numerator."""
    yt = [str(x).lower().strip() for x in y_true]
    yp = [str(x).lower().strip() for x in y_pred]
    out: dict[str, float] = {}
    for cls in VALID_LABELS:
        true_n = sum(1 for x in yt if x == cls)
        pred_n = sum(1 for x in yp if x == cls)
        out[f"true_{cls}"] = true_n
        out[f"pred_{cls}"] = pred_n
        out[f"overpred_{cls}"] = pred_n / true_n if true_n else float("nan")
    out["n_invalid"] = sum(1 for x in yp if x == "invalid")
    return out


def _build_overpred(preds_index: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in preds_index.iterrows():
        if not r.get("model", ""):
            # VTAG rows. Still useful — include them.
            pass
        path = REPO_ROOT / r["predictions_path"]
        if not path.is_file():
            continue
        try:
            df = load_predictions(path)
        except Exception:
            continue
        d = _overpred_ratios(df["ground_truth"].tolist(), df["predicted_label"].tolist())
        rows.append({
            "model": r.get("model") or "(no LLM)",
            "approach": r["approach"],
            "setting": r["setting"],
            "project": r["project"],
            "k_label": r["k_label"],
            **d,
            "predictions_path": r["predictions_path"],
        })
    return pd.DataFrame(rows)


def _plot_confusion_qwen32b(preds_index: pd.DataFrame) -> None:
    """4-panel confusion matrix for Qwen-32B agnostic: ZS, RAGTAG-k9, debias-k9-ps, FT."""
    sel = [
        ("ragtag", "zero_shot", "agnostic", "_overall", "Zero-shot"),
        ("ragtag", "k9", "agnostic", "_overall", "RAGTAG k=9 (ag)"),
        ("ft", "finetune_fixed", "agnostic", "_overall", "FT (ag)"),
        ("ragtag_debias", "k9", "project_specific", None, "Debias k=9 (ps avg)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (approach, k_label, setting, project, title) in zip(axes.flat, sel):
        if approach == "ragtag_debias":
            # Aggregate confusion matrices across 11 projects
            cm_total = np.zeros((3, 4), dtype=int)
            for proj in PROJECTS:
                rows = preds_index[
                    (preds_index["model"] == "Qwen-32B")
                    & (preds_index["approach"] == approach)
                    & (preds_index["setting"] == setting)
                    & (preds_index["k_label"] == k_label)
                    & (preds_index["project"] == proj)
                ]
                if rows.empty:
                    continue
                df = load_predictions(REPO_ROOT / rows.iloc[0]["predictions_path"])
                cm_total += _confusion(df["ground_truth"].tolist(), df["predicted_label"].tolist())
            cm = cm_total
        else:
            rows = preds_index[
                (preds_index["model"] == "Qwen-32B")
                & (preds_index["approach"] == approach)
                & (preds_index["setting"] == setting)
                & (preds_index["k_label"] == k_label)
                & (preds_index["project"] == project)
            ]
            if rows.empty:
                ax.set_title(f"{title} (no data)")
                ax.axis("off")
                continue
            df = load_predictions(REPO_ROOT / rows.iloc[0]["predictions_path"])
            cm = _confusion(df["ground_truth"].tolist(), df["predicted_label"].tolist())
        # Normalize by row (true label) for percentages
        row_totals = cm[:, :3].sum(axis=1, keepdims=True) + cm[:, 3:4]
        norm = cm / np.where(row_totals == 0, 1, row_totals)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        cols = VALID_LABELS + ["invalid"]
        ax.set_xticks(range(4))
        ax.set_xticklabels(cols)
        ax.set_yticks(range(3))
        ax.set_yticklabels(VALID_LABELS)
        for i in range(3):
            for j in range(4):
                v = norm[i, j]
                ax.text(j, i, f"{cm[i,j]}\n({v:.2f})", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
    fig.suptitle("Qwen-32B confusion matrices (counts and row-normalized)")
    fig.tight_layout()
    out = FIGURES / "rq2_confusion_qwen32b.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {rel(out)}")


def _plot_overpred(over: pd.DataFrame) -> None:
    """Bar plot: overpred_bug per (model, approach) at the typical config."""
    # For each model, aggregate plain RAGTAG (mean across ag k=1,3,6,9) and
    # zero-shot (ag), and debias (ps mean k=1,3,6,9), and FT.
    agg_rows: list[dict] = []
    for model in MODEL_ORDER:
        for approach, k_filter, setting, label in (
            ("ragtag", ["zero_shot"], "agnostic", "Zero-shot ag"),
            ("ragtag", ["k1", "k3", "k6", "k9"], "agnostic", "RAGTAG ag (k1-9 mean)"),
            ("ragtag", ["k1", "k3", "k6", "k9"], "project_specific", "RAGTAG ps (k1-9 mean)"),
            ("ragtag_debias", ["k1", "k3", "k6", "k9"], "project_specific", "Debias ps (k1-9 mean)"),
            ("ft", ["finetune_fixed"], "agnostic", "FT ag"),
        ):
            sub = over[
                (over["model"] == model)
                & (over["approach"] == approach)
                & (over["setting"] == setting)
                & (over["k_label"].isin(k_filter))
            ]
            if sub.empty:
                continue
            agg_rows.append({
                "model": model,
                "approach_label": label,
                "overpred_bug_mean": sub["overpred_bug"].mean(),
                "overpred_feature_mean": sub["overpred_feature"].mean(),
                "overpred_question_mean": sub["overpred_question"].mean(),
            })
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(DOCS_ANALYSIS / "rq2_overpred_summary.csv", index=False)

    # Bar plot.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    approaches = [
        "Zero-shot ag", "RAGTAG ag (k1-9 mean)", "RAGTAG ps (k1-9 mean)",
        "Debias ps (k1-9 mean)", "FT ag",
    ]
    x = np.arange(len(MODEL_ORDER))
    width = 0.16
    for i, ap in enumerate(approaches):
        vals = []
        for m in MODEL_ORDER:
            row = agg[(agg["model"] == m) & (agg["approach_label"] == ap)]
            vals.append(row["overpred_bug_mean"].iloc[0] if not row.empty else np.nan)
        ax.bar(x + i * width, vals, width, label=ap)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label="Calibrated (= 1.0)")
    ax.set_xticks(x + 2 * width)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("Bug over-prediction ratio (predicted_bug / true_bug)")
    ax.set_title("Bug over-prediction across approaches and model sizes")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIGURES / "rq2_bug_overprediction.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {rel(out)}")


def _retrieval_bug_skew(k_set: list[int]) -> pd.DataFrame:
    """For each setting and each k, compute fraction of top-k neighbors
    that are bugs, sliced by ground-truth class.

    Reads:
      results/issues11k/agnostic/neighbors/neighbors_k{3,9,30}.csv
      results/issues11k/project_specific/<proj>/neighbors/neighbors_k{3,9,30}.csv
    """
    rows: list[dict] = []
    for setting in ("agnostic", "project_specific"):
        if setting == "agnostic":
            paths = [(RESULTS_DIR / "agnostic" / "neighbors" / f"neighbors_k{k}.csv", "_overall", k)
                     for k in k_set]
        else:
            paths = []
            for proj in PROJECTS:
                for k in k_set:
                    paths.append((
                        RESULTS_DIR / "project_specific" / proj / "neighbors" / f"neighbors_k{k}.csv",
                        proj, k,
                    ))
        for path, proj, k in paths:
            if not path.is_file():
                continue
            df = pd.read_csv(path)
            grouped = df.groupby("test_idx").agg(
                test_label=("test_label", "first"),
                n_bug=("neighbor_label", lambda s: (s.str.lower() == "bug").sum()),
                n_total=("neighbor_label", "size"),
            ).reset_index()
            grouped["frac_bug"] = grouped["n_bug"] / grouped["n_total"]
            for cls in VALID_LABELS:
                sub = grouped[grouped["test_label"].str.lower() == cls]
                rows.append({
                    "setting": setting,
                    "project": proj,
                    "top_k": k,
                    "ground_truth": cls,
                    "n_test_issues": len(sub),
                    "frac_bug_mean": float(sub["frac_bug"].mean()) if len(sub) else float("nan"),
                    "frac_bug_majority": float((sub["frac_bug"] > 0.5).mean()) if len(sub) else float("nan"),
                    "frac_bug_p25": float(sub["frac_bug"].quantile(0.25)) if len(sub) else float("nan"),
                    "frac_bug_p75": float(sub["frac_bug"].quantile(0.75)) if len(sub) else float("nan"),
                })
    return pd.DataFrame(rows)


def _plot_retrieval_skew(df: pd.DataFrame) -> None:
    """Three sub-bars per setting per k, one per ground-truth class.

    Y-axis: mean frac_bug_in_topk.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, setting in zip(axes, ("agnostic", "project_specific")):
        sub = df[df["setting"] == setting]
        if setting == "project_specific":
            sub = sub.groupby(["top_k", "ground_truth"], as_index=False)[
                ["frac_bug_mean", "frac_bug_majority"]
            ].mean()
        ks = sorted(sub["top_k"].unique())
        x = np.arange(len(ks))
        width = 0.27
        for i, cls in enumerate(VALID_LABELS):
            vals = []
            for k in ks:
                row = sub[(sub["top_k"] == k) & (sub["ground_truth"] == cls)]
                vals.append(row["frac_bug_mean"].iloc[0] if not row.empty else np.nan)
            ax.bar(x + i * width, vals, width, label=f"GT={cls}")
        ax.axhline(0.333, color="black", linestyle="--", linewidth=1, alpha=0.5,
                   label="Balanced (= 1/3)")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_ylabel("Mean fraction of top-k neighbors that are 'bug'")
        title = "Agnostic" if setting == "agnostic" else "Project-specific (mean across 11)"
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Retrieval-side bug skew: top-k bug fraction by ground truth")
    fig.tight_layout()
    out = FIGURES / "rq2_retrieval_bug_skew.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {rel(out)}")


def main() -> None:
    ensure_dirs()
    preds_index = pd.read_csv(DOCS_ANALYSIS / "preds_index.csv")

    print("Computing over-prediction ratios from prediction CSVs...")
    over = _build_overpred(preds_index)
    over.to_csv(DOCS_ANALYSIS / "rq2_overprediction.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq2_overprediction.csv')} ({len(over)} rows)")

    _plot_confusion_qwen32b(preds_index)
    _plot_overpred(over)

    print("Computing retrieval-side bug skew from neighbors CSVs...")
    skew = _retrieval_bug_skew(k_set=[3, 9, 30])
    skew.to_csv(DOCS_ANALYSIS / "rq2_retrieval_bug_skew.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq2_retrieval_bug_skew.csv')} ({len(skew)} rows)")
    _plot_retrieval_skew(skew)

    # Summary print: overpred_bug for ZS and RAGTAG (both settings).
    print("\n--- Over-prediction summary (mean across models and k) ---")
    for label, ap, k_filter, setting in (
        ("Zero-shot ag", "ragtag", ["zero_shot"], "agnostic"),
        ("RAGTAG ag", "ragtag", ["k1", "k3", "k6", "k9"], "agnostic"),
        ("RAGTAG ps", "ragtag", ["k1", "k3", "k6", "k9"], "project_specific"),
        ("Debias ps", "ragtag_debias", ["k1", "k3", "k6", "k9"], "project_specific"),
        ("FT ag", "ft", ["finetune_fixed"], "agnostic"),
        ("FT ps", "ft", ["finetune_fixed"], "project_specific"),
    ):
        sub = over[
            (over["approach"] == ap)
            & (over["setting"] == setting)
            & (over["k_label"].isin(k_filter))
        ]
        if sub.empty:
            continue
        bug = sub["overpred_bug"].mean()
        feat = sub["overpred_feature"].mean()
        q = sub["overpred_question"].mean()
        print(f"  {label:14s}  bug={bug:.3f}  feat={feat:.3f}  q={q:.3f}")

    print("\n--- Retrieval bug skew (agnostic) ---")
    ag = skew[skew["setting"] == "agnostic"]
    print(ag.to_string(index=False))


if __name__ == "__main__":
    main()

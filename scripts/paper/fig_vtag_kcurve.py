"""Generate paper/figures/vtag_kcurve.{pdf,png} with POOLED PS aggregation.

Convention: see paper/sections/04_setup.tex (Evaluation Metrics paragraph) and
the project memory `project_aggregation_convention.md`. PS macro F1 is computed
by concatenating the 11 per-project prediction CSVs into a single 3,300-issue
test set and evaluating once. Do NOT average per-project F1 numbers.

Outputs a single 2-panel figure used by Section 5 (RQ1):
  (left)  VOTAG macro F1 vs k for PS and PA, with peak markers
  (right) Per-class F1 (bug/feature/question) at each setting's best k
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "issues11k"
FIG_DIR = REPO_ROOT / "paper" / "figures"

LABELS = ["bug", "feature", "question"]
KS = list(range(1, 21)) + [25, 30]
RAGTAG_KS = (1, 3, 6, 9)


def _macro_f1(df: pd.DataFrame) -> float:
    return f1_score(
        df["ground_truth"], df["predicted_label"],
        average="macro", labels=LABELS, zero_division=0,
    )


def _per_class_f1(df: pd.DataFrame) -> dict[str, float]:
    _, _, f1, _ = precision_recall_fscore_support(
        df["ground_truth"], df["predicted_label"],
        labels=LABELS, zero_division=0,
    )
    return dict(zip(LABELS, f1))


def _per_class_pr(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-class precision and recall (metric-triangulation panel, layout B)."""
    pr, rc, _, _ = precision_recall_fscore_support(
        df["ground_truth"], df["predicted_label"],
        labels=LABELS, zero_division=0,
    )
    return {"P": dict(zip(LABELS, pr)), "R": dict(zip(LABELS, rc))}


def _plot_single(curve: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    """Layout A: k-curve only, column width (per-class numbers live in the
    triangulation table)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.plot(curve["k"], curve["pa_f1"], marker="o", markersize=4,
            linewidth=1.6, color="C0", label="PA (project-agnostic)")
    ax.plot(curve["k"], curve["ps_f1"], marker="s", markersize=4,
            linewidth=1.6, color="C1", label="PS (project-specific)")
    for k in RAGTAG_KS:
        ax.axvline(k, color="black", linestyle="--", alpha=0.18, linewidth=0.9)
    pa_best = curve.loc[curve["pa_f1"].idxmax()]
    ps_best = curve.loc[curve["ps_f1"].idxmax()]
    ax.scatter([pa_best["k"]], [pa_best["pa_f1"]], s=110, facecolors="none",
               edgecolors="C0", linewidths=1.8, zorder=5)
    ax.scatter([ps_best["k"]], [ps_best["ps_f1"]], s=110, facecolors="none",
               edgecolors="C1", linewidths=1.8, zorder=5)
    ax.set_xlabel(r"$k$ (number of retrieved neighbors)")
    ax.set_ylabel("Macro $F_1$")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 30.5)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_pr(curve: pd.DataFrame, pr: dict, out_pdf: Path, out_png: Path) -> None:
    """Layout B: left k-curve, right per-class precision (hatched) and recall
    (filled) at each setting's best k."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.0, 4.2),
                                      gridspec_kw={"width_ratios": [1.4, 1.0]})
    ax_l.plot(curve["k"], curve["pa_f1"], marker="o", markersize=4,
              linewidth=1.6, color="C0", label="PA (project-agnostic)")
    ax_l.plot(curve["k"], curve["ps_f1"], marker="s", markersize=4,
              linewidth=1.6, color="C1", label="PS (project-specific)")
    for k in RAGTAG_KS:
        ax_l.axvline(k, color="black", linestyle="--", alpha=0.18, linewidth=0.9)
    pa_best = curve.loc[curve["pa_f1"].idxmax()]
    ps_best = curve.loc[curve["ps_f1"].idxmax()]
    ax_l.scatter([pa_best["k"]], [pa_best["pa_f1"]], s=110, facecolors="none",
                 edgecolors="C0", linewidths=1.8, zorder=5)
    ax_l.scatter([ps_best["k"]], [ps_best["ps_f1"]], s=110, facecolors="none",
                 edgecolors="C1", linewidths=1.8, zorder=5)
    ax_l.set_xlabel(r"$k$ (number of retrieved neighbors)")
    ax_l.set_ylabel("Macro $F_1$")
    ax_l.legend(loc="lower right", fontsize=9, frameon=False)
    ax_l.grid(True, alpha=0.3)
    ax_l.set_xlim(0.5, 30.5)

    x = np.arange(len(LABELS))
    w = 0.19
    spec = [("pa", "P", -1.5, "C0", "//"), ("pa", "R", -0.5, "C0", None),
            ("ps", "P", 0.5, "C1", "//"), ("ps", "R", 1.5, "C1", None)]
    for setting, metric, off, color, hatch in spec:
        vals = [pr[setting][metric][lab] for lab in LABELS]
        k_best = int(pa_best["k"]) if setting == "pa" else int(ps_best["k"])
        bars = ax_r.bar(x + off * w, vals, w, color=color if hatch is None else "white",
                        edgecolor=color, hatch=hatch, linewidth=1.2,
                        label=f"{setting.upper()} (k={k_best}) {'precision' if metric == 'P' else 'recall'}")
        ax_r.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5)
    ax_r.axhline(1 / 3, color="0.5", linestyle=":", linewidth=0.9)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([lab.capitalize() for lab in LABELS])
    ax_r.set_ylabel("Per-class precision / recall")
    ax_r.set_ylim(0.40, 0.85)
    ax_r.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    ax_r.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _pa_preds(k: int) -> pd.DataFrame:
    return pd.read_csv(
        RESULTS / "agnostic" / "vtag" / "predictions" / f"preds_k{k}.csv",
        usecols=["ground_truth", "predicted_label"],
    )


def _ps_preds_pooled(projects: list[str], k: int) -> pd.DataFrame:
    """Concat per-project PS predictions into one 3,300-issue frame."""
    parts = []
    for proj in projects:
        parts.append(pd.read_csv(
            RESULTS / "project_specific" / proj / "vtag" / "predictions"
            / f"preds_k{k}.csv",
            usecols=["ground_truth", "predicted_label"],
        ))
    return pd.concat(parts, ignore_index=True)


def _build_curves() -> pd.DataFrame:
    projects = sorted(p.name for p in (RESULTS / "project_specific").iterdir() if p.is_dir())
    rows = []
    for k in KS:
        rows.append({
            "k": k,
            "pa_f1": _macro_f1(_pa_preds(k)),
            "ps_f1": _macro_f1(_ps_preds_pooled(projects, k)),
        })
    return pd.DataFrame(rows)


def _plot(curve: pd.DataFrame, per_class: dict, out_pdf: Path, out_png: Path) -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.0, 4.2),
                                      gridspec_kw={"width_ratios": [1.4, 1.0]})

    # ---- Left: kcurve (unchanged) ----
    ax_l.plot(curve["k"], curve["pa_f1"], marker="o", markersize=4,
              linewidth=1.6, color="C0", label="PA (project-agnostic)")
    ax_l.plot(curve["k"], curve["ps_f1"], marker="s", markersize=4,
              linewidth=1.6, color="C1", label="PS (project-specific)")
    for k in RAGTAG_KS:
        ax_l.axvline(k, color="black", linestyle="--", alpha=0.18, linewidth=0.9)
    pa_best = curve.loc[curve["pa_f1"].idxmax()]
    ps_best = curve.loc[curve["ps_f1"].idxmax()]
    ax_l.scatter([pa_best["k"]], [pa_best["pa_f1"]], s=110, facecolors="none",
                 edgecolors="C0", linewidths=1.8, zorder=5)
    ax_l.scatter([ps_best["k"]], [ps_best["ps_f1"]], s=110, facecolors="none",
                 edgecolors="C1", linewidths=1.8, zorder=5)
    ax_l.set_xlabel(r"$k$ (number of retrieved neighbors)")
    ax_l.set_ylabel("Macro $F_1$")
    ax_l.legend(loc="lower right", fontsize=9, frameon=False)
    ax_l.grid(True, alpha=0.3)
    ax_l.set_xlim(0.5, 30.5)

    # ---- Right: grouped per-class F1 bar chart at each setting's best k ----
    x = np.arange(len(LABELS))
    width = 0.36
    pa_vals = [per_class["pa"][lab] for lab in LABELS]
    ps_vals = [per_class["ps"][lab] for lab in LABELS]
    bars_pa = ax_r.bar(x - width / 2, pa_vals, width,
                       color="C0", label=f"PA (k={int(pa_best['k'])})")
    bars_ps = ax_r.bar(x + width / 2, ps_vals, width,
                       color="C1", label=f"PS (k={int(ps_best['k'])})")
    ax_r.bar_label(bars_pa, fmt="%.3f", padding=2, fontsize=8.5)
    ax_r.bar_label(bars_ps, fmt="%.3f", padding=2, fontsize=8.5)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([lab.capitalize() for lab in LABELS])
    ax_r.set_ylabel("Per-class $F_1$")
    ax_r.set_ylim(0.50, max(max(pa_vals), max(ps_vals)) * 1.08)
    ax_r.legend(loc="upper right", fontsize=9, frameon=False)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    projects = sorted(p.name for p in (RESULTS / "project_specific").iterdir() if p.is_dir())

    curve = _build_curves()
    print(curve.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    pa_best_k = int(curve.loc[curve["pa_f1"].idxmax(), "k"])
    ps_best_k = int(curve.loc[curve["ps_f1"].idxmax(), "k"])
    per_class = {
        "pa": _per_class_f1(_pa_preds(pa_best_k)),
        "ps": _per_class_f1(_ps_preds_pooled(projects, ps_best_k)),
    }

    out_pdf = FIG_DIR / "vtag_kcurve.pdf"
    out_png = FIG_DIR / "vtag_kcurve.png"
    _plot(curve, per_class, out_pdf, out_png)
    print(f"\nwrote {out_pdf.relative_to(REPO_ROOT)}")
    print(f"wrote {out_png.relative_to(REPO_ROOT)}")

    if "--single-panel" in sys.argv:  # layout A
        _plot_single(curve, FIG_DIR / "vtag_kcurve_single.pdf", FIG_DIR / "vtag_kcurve_single.png")
        print("wrote paper/figures/vtag_kcurve_single.{pdf,png}")
    if "--pr-panel" in sys.argv:  # layout B
        pr = {"pa": _per_class_pr(_pa_preds(pa_best_k)),
              "ps": _per_class_pr(_ps_preds_pooled(projects, ps_best_k))}
        _plot_pr(curve, pr, FIG_DIR / "vtag_kcurve_pr.pdf", FIG_DIR / "vtag_kcurve_pr.png")
        print("wrote paper/figures/vtag_kcurve_pr.{pdf,png}")
        for s_ in ("pa", "ps"):
            for lab in LABELS:
                print(f"  {s_.upper()} {lab:<9} P={pr[s_]['P'][lab]:.4f} R={pr[s_]['R'][lab]:.4f}")

    pa_best = curve.loc[curve["pa_f1"].idxmax()]
    ps_best = curve.loc[curve["ps_f1"].idxmax()]
    gap_peak = pa_best["pa_f1"] - ps_best["ps_f1"]
    abs_gap_mean = (curve["pa_f1"] - curve["ps_f1"]).abs().mean()
    print("\n--- headline (pooled) ---")
    print(f"PA best: {pa_best['pa_f1']:.4f} at k={int(pa_best['k'])}")
    print(f"PS best: {ps_best['ps_f1']:.4f} at k={int(ps_best['k'])}")
    print(f"gap at peaks: {gap_peak:+.4f}")
    print(f"mean |PA-PS| across k=[1,30]: {abs_gap_mean:.4f}")
    print(f"\n--- per-class F1 at best-k ---")
    for lab in LABELS:
        print(f"  {lab:9s}  PA={per_class['pa'][lab]:.4f}  PS={per_class['ps'][lab]:.4f}")


if __name__ == "__main__":
    main()

"""RQ1: VTAG as the retrieval-only floor; plateau motivates RAGTAG k grid.

Tasks:
  1.1 Full VTAG k-curve (k=1..20, 25, 30) for both settings, plot + table
  1.2 VTAG cost (wall-clock seconds, tokens per query)
  1.3 Per-project variance
  1.4 k-grid justification panel

Outputs:
  docs/analysis/rq1_vtag_table.csv
  docs/analysis/rq1_vtag_per_project.csv
  docs/analysis/rq1_vtag_cost.csv
  docs/analysis/figures/rq1_vtag_curve_agnostic.png
  docs/analysis/figures/rq1_vtag_curve_project_specific.png
  docs/analysis/figures/rq1_vtag_kgrid_justification.png
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
    PROJECTS,
    REPO_ROOT,
    ensure_dirs,
    rel,
)


PLATEAU_DELTA = 0.005


def _vtag_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Agnostic VTAG: one row per k. Project-specific: per-project, one per k."""
    return df[(df["approach"] == "vtag")].copy()


def _build_overall_table(cells: pd.DataFrame) -> pd.DataFrame:
    """For each (setting, k), produce overall macro F1.

    Agnostic _overall: read directly.
    Project-specific: macro-F1 averaged across the 11 projects (each cell
    has 100 issues per class, equally weighted average is fine).
    """
    rows = []
    ag = cells[(cells["setting"] == "agnostic") & (cells["project"] == "_overall")]
    for _, r in ag.iterrows():
        rows.append({
            "setting": "agnostic",
            "k": int(r["k"]),
            "f1_macro": r["f1_macro"],
            "accuracy": r["accuracy"],
            "n_projects": 11,
        })
    ps = cells[cells["setting"] == "project_specific"]
    for k_val in sorted(ps["k"].dropna().unique()):
        sub = ps[ps["k"] == k_val]
        rows.append({
            "setting": "project_specific",
            "k": int(k_val),
            "f1_macro": sub["f1_macro"].mean(),
            "accuracy": sub["accuracy"].mean(),
            "n_projects": sub["project"].nunique(),
        })
    return pd.DataFrame(rows).sort_values(["setting", "k"]).reset_index(drop=True)


def _per_project_table(cells: pd.DataFrame) -> pd.DataFrame:
    """For project_specific only: one row per (project, k)."""
    ps = cells[(cells["approach"] == "vtag") & (cells["setting"] == "project_specific")]
    return ps[["project", "k", "f1_macro", "accuracy"]].sort_values(
        ["project", "k"]
    ).reset_index(drop=True)


def _identify_plateau(curve: pd.DataFrame, delta: float = PLATEAU_DELTA) -> int:
    """First k that is within `delta` of the peak F1 on the curve.

    This corresponds to "smallest k that captures (peak - delta) of the
    achievable F1." Reading from the figures, this is the k where the
    curve first enters its top band.
    """
    curve = curve.sort_values("k").reset_index(drop=True)
    peak = float(curve["f1_macro"].max())
    threshold = peak - delta
    for _, r in curve.iterrows():
        if r["f1_macro"] >= threshold:
            return int(r["k"])
    return int(curve["k"].iloc[-1])


def _plot_vtag_curve(curve: pd.DataFrame, per_proj: pd.DataFrame, setting: str,
                     plateau_k: int, out: Path) -> None:
    """Plot macro F1 vs k with per-project envelope for ps, single line for ag."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if setting == "project_specific":
        # Plot each project as a thin line, mean as bold.
        for proj in PROJECTS:
            pp = per_proj[per_proj["project"] == proj].sort_values("k")
            ax.plot(pp["k"], pp["f1_macro"], color="lightgray", linewidth=0.8, alpha=0.7)
        ax.plot(curve["k"], curve["f1_macro"], color="black", linewidth=2,
                label="Mean across 11 projects")
    else:
        ax.plot(curve["k"], curve["f1_macro"], color="black", linewidth=2,
                label="Macro F1 (3,300 issues)")
    # Mark RAGTAG k grid.
    for k in (1, 3, 6, 9):
        ax.axvline(k, color="C0", linestyle="--", alpha=0.35, linewidth=1)
    ax.axvline(plateau_k, color="C3", linestyle=":", linewidth=1.5,
               label=f"Plateau (k={plateau_k}, Δ < {PLATEAU_DELTA})")
    ax.set_xlabel("k (number of retrieved neighbors)")
    ax.set_ylabel("Macro F1")
    title_setting = "Agnostic" if setting == "agnostic" else "Project-specific"
    ax.set_title(f"VTAG plateau — {title_setting}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_kgrid_panel(curve_ag: pd.DataFrame, curve_ps: pd.DataFrame,
                      plateau_ag: int, plateau_ps: int, out: Path) -> None:
    """Side-by-side: both settings on one panel, with the chosen RAGTAG k grid highlighted."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(curve_ag["k"], curve_ag["f1_macro"], marker="o", markersize=4,
            label="VTAG agnostic", color="C0")
    ax.plot(curve_ps["k"], curve_ps["f1_macro"], marker="s", markersize=4,
            label="VTAG project-specific (mean)", color="C1")
    for k in (1, 3, 6, 9):
        ax.axvline(k, color="black", linestyle="--", alpha=0.25, linewidth=1)
    ax.text(1, ax.get_ylim()[0], "RAGTAG k grid", fontsize=8,
            color="black", alpha=0.6, va="bottom")
    ax.set_xlabel("k")
    ax.set_ylabel("Macro F1")
    ax.set_title("VTAG plateau motivates the RAGTAG k grid")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _vtag_cost(cells: pd.DataFrame) -> pd.DataFrame:
    """Read cost_metrics.csv for VTAG cells (one file per cell, dedup paths)."""
    cost_idx = pd.read_csv(DOCS_ANALYSIS / "cost_index.csv")
    vtag_costs = cost_idx[cost_idx["approach"] == "vtag"]
    seen: set[str] = set()
    rows = []
    for _, r in vtag_costs.iterrows():
        if r["cost_path"] in seen:
            continue
        seen.add(r["cost_path"])
        cm_path = REPO_ROOT / r["cost_path"]
        try:
            cm = pd.read_csv(cm_path)
        except Exception:
            continue
        for _, cmrow in cm.iterrows():
            rows.append({
                "setting": r["setting"],
                "project": r["project"],
                "k_label": cmrow.get("k_label", ""),
                "wall_time_s": cmrow.get("wall_time_s"),
                "issues_per_second": cmrow.get("issues_per_second"),
                "total_issues": cmrow.get("total_issues"),
                "total_prompt_tokens": cmrow.get("total_prompt_tokens"),
                "total_generated_tokens": cmrow.get("total_generated_tokens"),
                "gpu_peak_memory_mb": cmrow.get("gpu_peak_memory_mb"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    cells = pd.read_csv(DOCS_ANALYSIS / "all_cells.csv")
    vtag = _vtag_overall(cells)
    overall = _build_overall_table(vtag)
    overall.to_csv(DOCS_ANALYSIS / "rq1_vtag_table.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq1_vtag_table.csv')} ({len(overall)} rows)")

    per_proj = _per_project_table(vtag)
    per_proj.to_csv(DOCS_ANALYSIS / "rq1_vtag_per_project.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq1_vtag_per_project.csv')} ({len(per_proj)} rows)")

    curve_ag = overall[overall["setting"] == "agnostic"][["k", "f1_macro"]].sort_values("k").reset_index(drop=True)
    curve_ps = overall[overall["setting"] == "project_specific"][["k", "f1_macro"]].sort_values("k").reset_index(drop=True)
    plateau_ag = _identify_plateau(curve_ag)
    plateau_ps = _identify_plateau(curve_ps)
    print(f"plateau (agnostic): k={plateau_ag}")
    print(f"plateau (project-specific): k={plateau_ps}")

    _plot_vtag_curve(curve_ag, per_proj, "agnostic", plateau_ag,
                     FIGURES / "rq1_vtag_curve_agnostic.png")
    _plot_vtag_curve(curve_ps, per_proj, "project_specific", plateau_ps,
                     FIGURES / "rq1_vtag_curve_project_specific.png")
    _plot_kgrid_panel(curve_ag, curve_ps, plateau_ag, plateau_ps,
                      FIGURES / "rq1_vtag_kgrid_justification.png")
    print("wrote 3 figures under docs/analysis/figures/")

    cost = _vtag_cost(cells)
    cost.to_csv(DOCS_ANALYSIS / "rq1_vtag_cost.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq1_vtag_cost.csv')} ({len(cost)} rows)")

    # Variance summary across projects per k.
    var = (
        per_proj.groupby("k")["f1_macro"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "f1_mean", "std": "f1_std",
                         "min": "f1_min", "max": "f1_max"})
    )
    var.to_csv(DOCS_ANALYSIS / "rq1_vtag_variance.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq1_vtag_variance.csv')} ({len(var)} rows)")

    # Print key headline numbers.
    print("\n--- RQ1 headline ---")
    for setting, curve in (("agnostic", curve_ag), ("project_specific", curve_ps)):
        for k in (1, 3, 6, 9, 15, 20, 30):
            row = curve[curve["k"] == k]
            if not row.empty:
                print(f"  {setting:18s} k={k:2d}  F1={row['f1_macro'].iloc[0]:.4f}")
    print()
    if not cost.empty:
        agg = cost.groupby("setting")["wall_time_s"].agg(["mean", "median", "max"])
        print("VTAG wall-clock seconds per cell (across all k):")
        print(agg.to_string())
        tt = cost.groupby("setting")["total_prompt_tokens"].sum()
        print(f"\nTotal prompt tokens (sanity 0?): {dict(tt)}")


if __name__ == "__main__":
    main()

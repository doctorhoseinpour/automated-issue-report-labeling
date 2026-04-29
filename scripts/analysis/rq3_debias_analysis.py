"""RQ3.1–3.4: Debiased RAGTAG analysis.

3.1 Plain RAGTAG vs debias-RAGTAG, all (model, k) ps cells. Macro F1 and
    per-class precision/recall deltas.
3.2 Debias-best vs FT, per Qwen size (CIs are already in rq2_significance).
3.3 Per-project debias-vs-FT win rate.
3.4 Scaling effect: gain over plain RAGTAG vs model size; gap to FT vs size.

Outputs:
  docs/analysis/rq3_debias_vs_ragtag.csv
  docs/analysis/rq3_per_project_winrate.csv
  docs/analysis/figures/rq3_debias_gain_by_size.png
  docs/analysis/figures/rq3_per_project_winrate.png
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
    ensure_dirs,
    rel,
)


K_LABELS = ["k1", "k3", "k6", "k9"]


def _ps_aggregate(cells: pd.DataFrame, model: str, approach: str, k_label: str) -> dict | None:
    """Mean of metrics across 11 projects, project_specific."""
    sub = cells[
        (cells["model"] == model)
        & (cells["approach"] == approach)
        & (cells["setting"] == "project_specific")
        & (cells["k_label"] == k_label)
    ]
    if sub.empty:
        return None
    cols = [
        "f1_macro", "accuracy", "invalid_rate",
        "f1_bug", "precision_bug", "recall_bug",
        "f1_feature", "precision_feature", "recall_feature",
        "f1_question", "precision_question", "recall_question",
    ]
    return {c: sub[c].mean() for c in cols}


def _build_debias_vs_ragtag(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model in MODEL_ORDER:
        for k_label in K_LABELS:
            plain = _ps_aggregate(cells, model, "ragtag", k_label)
            debias = _ps_aggregate(cells, model, "ragtag_debias", k_label)
            if plain is None or debias is None:
                continue
            r = {"model": model, "k_label": k_label}
            for col in plain.keys():
                r[f"plain_{col}"] = plain[col]
                r[f"debias_{col}"] = debias[col]
                r[f"delta_{col}"] = debias[col] - plain[col]
            rows.append(r)
    return pd.DataFrame(rows)


def _per_project_winrate(cells: pd.DataFrame) -> pd.DataFrame:
    """Per project: best debias-ps F1 vs FT-ag F1 vs FT-ps F1, per model."""
    rows: list[dict] = []
    for model in MODEL_ORDER:
        ft_ag = cells[
            (cells["model"] == model)
            & (cells["approach"] == "ft")
            & (cells["setting"] == "agnostic")
            & (cells["project"] == "_overall")
        ]
        ft_ag_f1 = ft_ag["f1_macro"].iloc[0] if not ft_ag.empty else float("nan")
        for proj in PROJECTS:
            deb = cells[
                (cells["model"] == model)
                & (cells["approach"] == "ragtag_debias")
                & (cells["setting"] == "project_specific")
                & (cells["project"] == proj)
                & (cells["k_label"].isin(K_LABELS))
            ]
            best_deb = deb["f1_macro"].max() if not deb.empty else float("nan")
            ft_ps = cells[
                (cells["model"] == model)
                & (cells["approach"] == "ft")
                & (cells["setting"] == "project_specific")
                & (cells["project"] == proj)
            ]
            ft_ps_f1 = ft_ps["f1_macro"].iloc[0] if not ft_ps.empty else float("nan")
            rows.append({
                "model": model,
                "project": proj,
                "debias_best_ps": best_deb,
                "ft_ag": ft_ag_f1,
                "ft_ps": ft_ps_f1,
                "deb_minus_ft_ag": best_deb - ft_ag_f1,
                "deb_minus_ft_ps": best_deb - ft_ps_f1,
            })
    return pd.DataFrame(rows)


def _plot_debias_gain_by_size(debias_df: pd.DataFrame, cells: pd.DataFrame, out: Path) -> None:
    """Two panels: (1) Δ(debias - plain RAGTAG) by k for each model; (2) best-debias - best-FT by model size."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Δ over plain RAGTAG by k, per model
    ax = axes[0]
    for i, model in enumerate(MODEL_ORDER):
        sub = debias_df[debias_df["model"] == model]
        ax.plot([1, 3, 6, 9], sub["delta_f1_macro"].values, marker="o",
                label=model, linewidth=1.6)
    ax.axhline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_xticks([1, 3, 6, 9])
    ax.set_xlabel("k")
    ax.set_ylabel("Macro F1: Debias − Plain RAGTAG (ps mean)")
    ax.set_title("Debias gain over plain RAGTAG by k")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Best debias vs best FT by model size
    ax = axes[1]
    sizes = [3, 7, 14, 32]
    deb_best, ft_ag, ft_ps = [], [], []
    for model in MODEL_ORDER:
        sub = debias_df[debias_df["model"] == model]
        deb_best.append(sub["debias_f1_macro"].max())
        ft_ag.append(cells[(cells["model"] == model) & (cells["approach"] == "ft")
                           & (cells["setting"] == "agnostic")]["f1_macro"].iloc[0])
        ft_ps_rows = cells[
            (cells["model"] == model) & (cells["approach"] == "ft")
            & (cells["setting"] == "project_specific")
        ]
        ft_ps.append(ft_ps_rows["f1_macro"].mean() if not ft_ps_rows.empty else np.nan)
    ax.plot(sizes, deb_best, marker="^", color="C3", label="Best debias (ps)", linewidth=2)
    ax.plot(sizes, ft_ag, marker="D", color="C2", label="FT agnostic", linewidth=2)
    ax.plot(sizes, ft_ps, marker="s", color="C4", label="FT project-specific", linewidth=2)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}B" for s in sizes])
    ax.set_xlabel("Model size")
    ax.set_ylabel("Macro F1")
    ax.set_title("Best debias-ps vs FT by model size")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_winrate(winrate: pd.DataFrame, out: Path) -> None:
    """Heatmap: rows=projects, cols=model. Cells: Δ(debias-ps best, FT-ag). Annotated with FT-ps comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, col, title in zip(axes, ("deb_minus_ft_ag", "deb_minus_ft_ps"),
                                ("Best debias-ps − FT-ag", "Best debias-ps − FT-ps")):
        pivot = winrate.pivot(index="project", columns="model", values=col)
        pivot = pivot.reindex(index=PROJECTS, columns=MODEL_ORDER)
        im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-0.2, vmax=0.2)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER)
        ax.set_yticks(range(len(PROJECTS)))
        ax.set_yticklabels(PROJECTS)
        for i, proj in enumerate(PROJECTS):
            for j, m in enumerate(MODEL_ORDER):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            color="black" if abs(v) < 0.13 else "white", fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Per-project debias-vs-FT margin (positive = debias wins)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    cells = pd.read_csv(DOCS_ANALYSIS / "all_cells.csv")
    debias_df = _build_debias_vs_ragtag(cells)
    debias_df.to_csv(DOCS_ANALYSIS / "rq3_debias_vs_ragtag.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq3_debias_vs_ragtag.csv')} ({len(debias_df)} rows)")

    winrate = _per_project_winrate(cells)
    winrate.to_csv(DOCS_ANALYSIS / "rq3_per_project_winrate.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq3_per_project_winrate.csv')} ({len(winrate)} rows)")

    _plot_debias_gain_by_size(debias_df, cells, FIGURES / "rq3_debias_gain_by_size.png")
    _plot_winrate(winrate, FIGURES / "rq3_per_project_winrate.png")
    print("wrote 2 figures")

    # Print compact summary.
    print("\n--- Δ Macro F1 (debias - plain RAGTAG, ps mean) ---")
    pivot = debias_df.pivot(index="model", columns="k_label", values="delta_f1_macro")
    pivot = pivot[K_LABELS].reindex(MODEL_ORDER)
    print(pivot.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\n--- Per-project win counts (debias-ps best vs FT-ag) ---")
    for model in MODEL_ORDER:
        sub = winrate[winrate["model"] == model]
        wins_ag = (sub["deb_minus_ft_ag"] > 0).sum()
        wins_ps = (sub["deb_minus_ft_ps"] > 0).sum()
        avg_ag = sub["deb_minus_ft_ag"].mean()
        avg_ps = sub["deb_minus_ft_ps"].mean()
        print(f"  {model:10s}  vs FT-ag: {wins_ag}/11 wins, mean Δ={avg_ag:+.4f}; "
              f"vs FT-ps: {wins_ps}/11 wins, mean Δ={avg_ps:+.4f}")


if __name__ == "__main__":
    main()

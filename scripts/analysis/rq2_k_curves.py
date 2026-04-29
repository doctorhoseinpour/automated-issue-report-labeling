"""RQ2.3: RAGTAG k-curves per Qwen model.

For each model size, plot F1 vs k for RAGTAG (agnostic + project-specific
mean) and the VTAG/zero-shot anchors. Plus a 2x2 panel that puts all four
sizes side-by-side for the scaling-law preview.

Outputs:
  docs/analysis/figures/rq2_kcurves_panel.png   # 2x2, all four models
  docs/analysis/rq2_k_curves.csv                # tidy long form
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _utils import (  # noqa: E402
    DOCS_ANALYSIS,
    FIGURES,
    MODEL_ORDER,
    ensure_dirs,
    rel,
)


K_INTS = [0, 1, 3, 6, 9]
K_LABELS = ["zero_shot", "k1", "k3", "k6", "k9"]


def _agnostic_curve(cells: pd.DataFrame, model: str, approach: str) -> pd.DataFrame:
    sub = cells[
        (cells["model"] == model)
        & (cells["approach"] == approach)
        & (cells["setting"] == "agnostic")
        & (cells["project"] == "_overall")
    ]
    return sub[["k", "k_label", "f1_macro"]].sort_values("k").reset_index(drop=True)


def _ps_mean_curve(cells: pd.DataFrame, model: str, approach: str) -> pd.DataFrame:
    sub = cells[
        (cells["model"] == model)
        & (cells["approach"] == approach)
        & (cells["setting"] == "project_specific")
    ]
    if sub.empty:
        return pd.DataFrame(columns=["k", "k_label", "f1_macro"])
    grouped = (
        sub.groupby(["k_label", "k"])["f1_macro"]
        .mean()
        .reset_index()
        .sort_values("k")
    )
    return grouped


def _vtag_curve(cells: pd.DataFrame, setting: str, ks: list[int]) -> pd.DataFrame:
    if setting == "agnostic":
        sub = cells[
            (cells["approach"] == "vtag")
            & (cells["setting"] == "agnostic")
            & (cells["project"] == "_overall")
        ]
        return sub[sub["k"].isin(ks)][["k", "f1_macro"]].sort_values("k").reset_index(drop=True)
    sub = cells[(cells["approach"] == "vtag") & (cells["setting"] == "project_specific")]
    if sub.empty:
        return pd.DataFrame(columns=["k", "f1_macro"])
    grouped = (
        sub[sub["k"].isin(ks)]
        .groupby("k")["f1_macro"]
        .mean()
        .reset_index()
        .sort_values("k")
    )
    return grouped


def _plot_one(ax, cells: pd.DataFrame, model: str) -> None:
    ag = _agnostic_curve(cells, model, "ragtag")
    ps = _ps_mean_curve(cells, model, "ragtag")
    deb_ps = _ps_mean_curve(cells, model, "ragtag_debias")
    vtag_ag = _vtag_curve(cells, "agnostic", K_INTS[1:])  # exclude k=0
    vtag_ps = _vtag_curve(cells, "project_specific", K_INTS[1:])

    if not ag.empty:
        ax.plot(ag["k"], ag["f1_macro"], marker="o", color="C0",
                label="RAGTAG agnostic", linewidth=1.6)
    if not ps.empty:
        ax.plot(ps["k"], ps["f1_macro"], marker="s", color="C1",
                label="RAGTAG project-spec (mean)", linewidth=1.6)
    if not deb_ps.empty:
        ax.plot(deb_ps["k"], deb_ps["f1_macro"], marker="^", color="C3",
                label="Debias project-spec (mean)", linewidth=1.6)
    if not vtag_ag.empty:
        ax.plot(vtag_ag["k"], vtag_ag["f1_macro"], marker="x", color="gray",
                label="VTAG agnostic", linestyle="--", linewidth=1.2)
    if not vtag_ps.empty:
        ax.plot(vtag_ps["k"], vtag_ps["f1_macro"], marker="+", color="lightgray",
                label="VTAG project-spec (mean)", linestyle="--", linewidth=1.2)

    # FT horizontal line(s)
    ft_ag = cells[
        (cells["model"] == model)
        & (cells["approach"] == "ft")
        & (cells["setting"] == "agnostic")
    ]["f1_macro"].mean()
    ft_ps_rows = cells[
        (cells["model"] == model)
        & (cells["approach"] == "ft")
        & (cells["setting"] == "project_specific")
    ]
    ft_ps = ft_ps_rows["f1_macro"].mean() if not ft_ps_rows.empty else None

    if pd.notna(ft_ag):
        ax.axhline(ft_ag, color="C2", linestyle=":", linewidth=1.4,
                   label=f"FT agnostic = {ft_ag:.3f}")
    if ft_ps is not None and pd.notna(ft_ps):
        ax.axhline(ft_ps, color="C4", linestyle=":", linewidth=1.4,
                   label=f"FT project-spec = {ft_ps:.3f}")

    ax.set_xlabel("k")
    ax.set_ylabel("Macro F1")
    ax.set_title(model)
    ax.set_xticks(K_INTS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")


def main() -> None:
    ensure_dirs()
    cells = pd.read_csv(DOCS_ANALYSIS / "all_cells.csv")

    # Build the tidy long-form table.
    rows: list[dict] = []
    for model in MODEL_ORDER:
        for ap_label, approach in (("ragtag_ag", "ragtag"), ("ragtag_ps", "ragtag"),
                                   ("debias_ps", "ragtag_debias")):
            if ap_label.endswith("_ag"):
                df = _agnostic_curve(cells, model, approach)
            else:
                df = _ps_mean_curve(cells, model, approach)
            for _, r in df.iterrows():
                rows.append({
                    "model": model,
                    "approach": ap_label,
                    "k_label": r.get("k_label", ""),
                    "k": r["k"],
                    "f1_macro": r["f1_macro"],
                })
    df_curves = pd.DataFrame(rows)
    df_curves.to_csv(DOCS_ANALYSIS / "rq2_k_curves.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq2_k_curves.csv')} ({len(df_curves)} rows)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        _plot_one(ax, cells, model)
    fig.suptitle("RAGTAG k-curves per Qwen size — agnostic vs project-specific, with FT and VTAG anchors")
    fig.tight_layout()
    out = FIGURES / "rq2_kcurves_panel.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {rel(out)}")


if __name__ == "__main__":
    main()

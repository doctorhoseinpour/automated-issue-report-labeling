"""RQ2.4: Scaling-law axis — F1 vs model size per approach.

For each approach, plot best macro F1 vs model size (3B → 32B). One plot
per setting (agnostic, project-specific). VTAG drawn as a model-independent
horizontal anchor.

Outputs:
  docs/analysis/rq2_scaling_table.csv
  docs/analysis/figures/rq2_scaling_agnostic.png
  docs/analysis/figures/rq2_scaling_project_specific.png
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

MODEL_PARAMS = {"Qwen-3B": 3.0, "Qwen-7B": 7.0, "Qwen-14B": 14.0, "Qwen-32B": 32.0}

K_FILTER = ["k1", "k3", "k6", "k9"]


def _best_for(cells: pd.DataFrame, model: str, approach: str, setting: str,
              k_filter: list[str] | None = None) -> tuple[float, str | None]:
    sub = cells[
        (cells["model"] == model)
        & (cells["approach"] == approach)
        & (cells["setting"] == setting)
    ]
    if k_filter is not None:
        sub = sub[sub["k_label"].isin(k_filter)]
    if sub.empty:
        return float("nan"), None
    if setting == "agnostic":
        sub = sub[sub["project"] == "_overall"]
        if sub.empty:
            return float("nan"), None
        best = sub.loc[sub["f1_macro"].idxmax()]
        return float(best["f1_macro"]), str(best["k_label"])
    # project_specific: aggregate by k_label, mean across projects
    agg = sub.groupby("k_label")["f1_macro"].mean().reset_index()
    if agg.empty:
        return float("nan"), None
    best = agg.loc[agg["f1_macro"].idxmax()]
    return float(best["f1_macro"]), str(best["k_label"])


def _build_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for model in MODEL_ORDER:
        for setting in ("agnostic", "project_specific"):
            zs_f1, _ = _best_for(cells, model, "ragtag", "agnostic", k_filter=["zero_shot"])
            ragtag_f1, ragtag_k = _best_for(cells, model, "ragtag", setting, k_filter=K_FILTER)
            ft_f1, _ = _best_for(cells, model, "ft", setting)
            row = {
                "model": model,
                "params_b": MODEL_PARAMS[model],
                "setting": setting,
                "zero_shot_ag": zs_f1,
                "ragtag_best": ragtag_f1,
                "ragtag_best_k": ragtag_k,
                "ft": ft_f1,
            }
            if setting == "project_specific":
                deb_f1, deb_k = _best_for(cells, model, "ragtag_debias", setting, k_filter=K_FILTER)
                row["debias_best"] = deb_f1
                row["debias_best_k"] = deb_k
            rows.append(row)
    return pd.DataFrame(rows)


def _vtag_anchor(cells: pd.DataFrame, setting: str) -> float:
    if setting == "agnostic":
        sub = cells[
            (cells["approach"] == "vtag")
            & (cells["setting"] == "agnostic")
            & (cells["project"] == "_overall")
        ]
    else:
        sub = cells[(cells["approach"] == "vtag") & (cells["setting"] == "project_specific")]
        if sub.empty:
            return float("nan")
        agg = sub.groupby("k")["f1_macro"].mean().reset_index()
        return float(agg["f1_macro"].max())
    if sub.empty:
        return float("nan")
    return float(sub["f1_macro"].max())


def _plot(table: pd.DataFrame, setting: str, vtag: float, out: Path) -> None:
    sub = table[table["setting"] == setting].sort_values("params_b").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = sub["params_b"]
    ax.plot(x, sub["zero_shot_ag"], marker="o", color="C7",
            label="Zero-shot (agnostic, model-only)", linewidth=1.4, linestyle="--")
    ax.plot(x, sub["ragtag_best"], marker="s", color="C0",
            label="RAGTAG best", linewidth=2)
    if setting == "project_specific":
        ax.plot(x, sub["debias_best"], marker="^", color="C3",
                label="Debias best", linewidth=2)
    ax.plot(x, sub["ft"], marker="D", color="C2",
            label="Fine-tune", linewidth=2)
    ax.axhline(vtag, color="gray", linestyle=":", linewidth=1.4,
               label=f"VTAG floor (best k) = {vtag:.3f}")
    ax.set_xscale("log")
    ax.set_xticks(list(MODEL_PARAMS.values()))
    ax.set_xticklabels([f"{int(p)}B" for p in MODEL_PARAMS.values()])
    ax.set_xlabel("Model size (Qwen2.5-Instruct, log scale)")
    ax.set_ylabel("Macro F1")
    title_setting = "Agnostic" if setting == "agnostic" else "Project-specific"
    ax.set_title(f"Scaling-law axis — {title_setting}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    cells = pd.read_csv(DOCS_ANALYSIS / "all_cells.csv")
    table = _build_table(cells)
    table.to_csv(DOCS_ANALYSIS / "rq2_scaling_table.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq2_scaling_table.csv')} ({len(table)} rows)")

    for setting in ("agnostic", "project_specific"):
        vtag = _vtag_anchor(cells, setting)
        out = FIGURES / f"rq2_scaling_{setting}.png"
        _plot(table, setting, vtag, out)
        print(f"wrote {rel(out)}")

    print("\n--- Scaling table ---")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

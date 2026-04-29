"""RQ2.5: Pareto frontier — cost (GPU-seconds, tokens) vs macro F1.

For each (model, approach, setting, k) cell, read the cost CSV to get
wall-clock and training time (excluding model load time per the project
convention). Plot F1 vs cost and identify the Pareto frontier.

Two cost views:
  - GPU-seconds (training_time_s + wall_time_s for FT; wall_time_s only otherwise)
  - average prompt tokens per query

One plot per setting (agnostic, project-specific).

Outputs:
  docs/analysis/rq2_pareto_table.csv
  docs/analysis/figures/rq2_pareto_gpu_agnostic.png
  docs/analysis/figures/rq2_pareto_gpu_project_specific.png
  docs/analysis/figures/rq2_pareto_tokens_agnostic.png
  docs/analysis/figures/rq2_pareto_tokens_project_specific.png
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
    REPO_ROOT,
    ensure_dirs,
    rel,
)


def _load_cost(cost_index: pd.DataFrame) -> pd.DataFrame:
    """Read every cost CSV; return one row per (cost_path, k_label)."""
    seen: set[str] = set()
    rows: list[dict] = []
    for _, r in cost_index.iterrows():
        cp = r["cost_path"]
        if cp in seen:
            continue
        seen.add(cp)
        path = REPO_ROOT / cp
        try:
            cm = pd.read_csv(path)
        except Exception:
            continue
        for _, cmrow in cm.iterrows():
            klab = cmrow.get("k_label")
            if pd.isna(klab) or klab == "" or klab == "N/A":
                klab = r["k_label"]
            rows.append({
                "cost_path": cp,
                "model": r["model"],
                "model_tag": r.get("model_tag", ""),
                "setting": r["setting"],
                "project": r["project"],
                "approach": r["approach"],
                "k_label": klab,
                "wall_time_s": float(cmrow.get("wall_time_s") or 0.0),
                "training_time_s": float(cmrow.get("training_time_s") or 0.0)
                    if "training_time_s" in cmrow else 0.0,
                "model_load_time_s": float(cmrow.get("model_load_time_s") or 0.0)
                    if "model_load_time_s" in cmrow else 0.0,
                "total_prompt_tokens": float(cmrow.get("total_prompt_tokens") or 0.0),
                "total_generated_tokens": float(cmrow.get("total_generated_tokens") or 0.0),
                "avg_prompt_tokens": float(cmrow.get("avg_prompt_tokens") or 0.0),
                "total_issues": float(cmrow.get("total_issues") or 0.0),
                "gpu_peak_memory_mb": float(cmrow.get("gpu_peak_memory_mb") or 0.0),
            })
    return pd.DataFrame(rows)


def _aggregate(cost: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Combine cost rows with F1 from cells; aggregate ps across projects."""
    # Compute total compute seconds: wall + training (FT only). Exclude load.
    cost = cost.copy()
    cost["compute_s"] = cost["wall_time_s"] + cost["training_time_s"]

    # Build per-cell (model, approach, setting, k_label, project) table.
    # For project_specific, sum compute_s across 11 projects (each project is
    # one independent run); for agnostic, single number per cell.
    rows: list[dict] = []
    for (model, approach, setting, k_label), g in cost.groupby(
        ["model", "approach", "setting", "k_label"], dropna=False,
    ):
        if setting == "project_specific":
            compute_s = g["compute_s"].sum()
            avg_prompt = g["avg_prompt_tokens"].mean()
            issues = g["total_issues"].sum()
        else:
            compute_s = g["compute_s"].sum()
            avg_prompt = g["avg_prompt_tokens"].mean()
            issues = g["total_issues"].sum()

        # Look up F1 from all_cells.
        if setting == "project_specific":
            sub = cells[
                (cells["model"] == model)
                & (cells["approach"] == approach)
                & (cells["setting"] == setting)
                & (cells["k_label"] == k_label)
            ]
            f1 = sub["f1_macro"].mean() if not sub.empty else float("nan")
        else:
            sub = cells[
                (cells["model"] == model)
                & (cells["approach"] == approach)
                & (cells["setting"] == setting)
                & (cells["k_label"] == k_label)
                & (cells["project"] == "_overall")
            ]
            f1 = sub["f1_macro"].iloc[0] if not sub.empty else float("nan")

        rows.append({
            "model": model if pd.notna(model) and model else "(no LLM)",
            "approach": approach,
            "setting": setting,
            "k_label": k_label,
            "compute_s": compute_s,
            "avg_prompt_tokens": avg_prompt,
            "total_issues": issues,
            "f1_macro": f1,
        })
    return pd.DataFrame(rows)


def _pareto_frontier(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Return rows that lie on the upper-left Pareto frontier (low x, high y)."""
    sub = df[[x, y, "model", "approach", "setting", "k_label"]].dropna()
    sub = sub.sort_values(x, ascending=True).reset_index(drop=True)
    frontier_rows = []
    best_y = -float("inf")
    for _, r in sub.iterrows():
        if r[y] > best_y:
            frontier_rows.append(r)
            best_y = r[y]
    return pd.DataFrame(frontier_rows)


def _plot(df: pd.DataFrame, setting: str, x: str, x_label: str, out: Path,
          x_log: bool = True) -> None:
    sub = df[df["setting"] == setting].dropna(subset=[x, "f1_macro"]).copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    color_map = {
        "ragtag": "C0",
        "ragtag_debias": "C3",
        "ft": "C2",
        "vtag": "gray",
        "vtag_debias": "lightgray",
    }
    for approach, grp in sub.groupby("approach"):
        # Distinguish zero_shot from RAGTAG-with-k.
        zs_part = grp[grp["k_label"] == "zero_shot"]
        rest = grp[grp["k_label"] != "zero_shot"]
        c = color_map.get(approach, "black")
        for _, r in rest.iterrows():
            label = r.get("model", "")
            ax.scatter(r[x], r["f1_macro"], color=c, s=55, alpha=0.85,
                       edgecolor="black", linewidth=0.4)
            tag = f"{r['model'].replace('Qwen-', '')[:3]}/{r['k_label'].replace('finetune_fixed', 'FT').replace('k', 'k')}"
            ax.annotate(tag, (r[x], r["f1_macro"]), fontsize=6.5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        for _, r in zs_part.iterrows():
            ax.scatter(r[x], r["f1_macro"], color=c, s=55, alpha=0.5,
                       marker="P", edgecolor="black", linewidth=0.4)

    # Compute Pareto frontier (low cost, high F1)
    front = _pareto_frontier(sub, x, "f1_macro")
    if not front.empty:
        front = front.sort_values(x)
        ax.plot(front[x], front["f1_macro"], color="black", linestyle="-",
                linewidth=1.4, alpha=0.6, label="Pareto frontier")
    # Approach legend (one entry per approach)
    handles = []
    for approach, c in color_map.items():
        if approach not in sub["approach"].unique():
            continue
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                                   markersize=8, markerfacecolor=c,
                                   markeredgecolor="black", label=approach))
    ax.legend(handles=handles + [plt.Line2D([0], [0], color="black",
              linestyle="-", linewidth=1.4, label="Pareto frontier")],
              loc="lower right", fontsize=9)

    if x_log:
        ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Macro F1")
    title_setting = "Agnostic" if setting == "agnostic" else "Project-specific"
    ax.set_title(f"Pareto: {x_label} vs Macro F1 — {title_setting}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    cost_index = pd.read_csv(DOCS_ANALYSIS / "cost_index.csv")
    cells = pd.read_csv(DOCS_ANALYSIS / "all_cells.csv")
    raw = _load_cost(cost_index)
    print(f"loaded {len(raw)} raw cost rows")
    table = _aggregate(raw, cells)
    table.to_csv(DOCS_ANALYSIS / "rq2_pareto_table.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq2_pareto_table.csv')} ({len(table)} rows)")

    for setting in ("agnostic", "project_specific"):
        _plot(table, setting, "compute_s", "GPU-seconds (excl. model load)",
              FIGURES / f"rq2_pareto_gpu_{setting}.png")
        _plot(table, setting, "avg_prompt_tokens", "Avg prompt tokens / query",
              FIGURES / f"rq2_pareto_tokens_{setting}.png", x_log=False)
        print(f"wrote pareto figures for {setting}")

    print("\n--- Pareto frontier (project_specific, GPU-s vs F1) ---")
    sub = table[table["setting"] == "project_specific"].dropna(subset=["compute_s", "f1_macro"])
    front = _pareto_frontier(sub, "compute_s", "f1_macro")
    print(front[["model", "approach", "k_label", "compute_s", "f1_macro"]].to_string(index=False))


if __name__ == "__main__":
    main()

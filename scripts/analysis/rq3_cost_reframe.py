"""RQ3.6: Cost reframe — debias-ps vs FT-ag.

Per Qwen size:
  - debias-ps best k: total inference seconds (sum across 11 projects, no training).
  - FT-ag: training time + inference time on 3,300 issues.
  - Examples used: debias=300 ps retrieval examples per project; FT=3,300 ag training examples.

Outputs:
  docs/analysis/rq3_cost_reframe.csv
  docs/analysis/figures/rq3_cost_reframe.png
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
    REPO_ROOT,
    ensure_dirs,
    rel,
)


DEBIAS_K = {"Qwen-3B": "k6", "Qwen-7B": "k6", "Qwen-14B": "k9", "Qwen-32B": "k9"}


def _read_costs(cost_index: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cost per (model, approach, setting, k_label) across 11 projects (sum) for ps;
    or single value for agnostic. Returns total compute_s, n_examples_used."""
    rows = []
    seen: set[str] = set()
    for _, r in cost_index.iterrows():
        if r["cost_path"] in seen:
            continue
        seen.add(r["cost_path"])
        path = REPO_ROOT / r["cost_path"]
        try:
            cm = pd.read_csv(path)
        except Exception:
            continue
        for _, cmrow in cm.iterrows():
            klab = cmrow.get("k_label")
            if pd.isna(klab) or klab == "" or klab == "N/A":
                klab = r["k_label"]
            wall = float(cmrow.get("wall_time_s") or 0.0)
            train = float(cmrow.get("training_time_s") or 0.0) if "training_time_s" in cmrow else 0.0
            rows.append({
                "model": r["model"],
                "approach": r["approach"],
                "setting": r["setting"],
                "project": r["project"],
                "k_label": klab,
                "wall_time_s": wall,
                "training_time_s": train,
                "compute_s": wall + train,
            })
    return pd.DataFrame(rows)


def _build_table(cost: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        # debias-ps best k: sum compute across 11 projects
        kdb = DEBIAS_K[model]
        deb_cells = cost[
            (cost["model"] == model)
            & (cost["approach"] == "ragtag_debias")
            & (cost["setting"] == "project_specific")
            & (cost["k_label"] == kdb)
        ]
        deb_compute = deb_cells["compute_s"].sum()
        deb_train = deb_cells["training_time_s"].sum()  # should be 0
        deb_f1_rows = cells[
            (cells["model"] == model)
            & (cells["approach"] == "ragtag_debias")
            & (cells["setting"] == "project_specific")
            & (cells["k_label"] == kdb)
        ]
        deb_f1 = deb_f1_rows["f1_macro"].mean()

        # FT-ag
        ft_cells = cost[
            (cost["model"] == model)
            & (cost["approach"] == "ft")
            & (cost["setting"] == "agnostic")
        ]
        ft_compute = ft_cells["compute_s"].sum()
        ft_train = ft_cells["training_time_s"].sum()
        ft_inf = ft_cells["wall_time_s"].sum()
        ft_f1_rows = cells[
            (cells["model"] == model)
            & (cells["approach"] == "ft")
            & (cells["setting"] == "agnostic")
            & (cells["project"] == "_overall")
        ]
        ft_f1 = ft_f1_rows["f1_macro"].iloc[0] if not ft_f1_rows.empty else float("nan")

        rows.append({
            "model": model,
            "debias_k": kdb,
            "debias_compute_s": deb_compute,
            "debias_training_s": deb_train,
            "debias_f1": deb_f1,
            "debias_examples_per_project": 300,
            "debias_total_examples_seen": 300 * 11,  # 11 projects, 300 each
            "ft_compute_s": ft_compute,
            "ft_training_s": ft_train,
            "ft_inference_s": ft_inf,
            "ft_f1": ft_f1,
            "ft_examples_trained_on": 3300,
            "delta_f1": deb_f1 - ft_f1,
            "compute_ratio_deb_over_ft": deb_compute / ft_compute if ft_compute else float("nan"),
        })
    return pd.DataFrame(rows)


def _plot(table: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sizes = [3, 7, 14, 32]

    ax = axes[0]
    ax.bar([s - 0.6 for s in sizes], table["ft_training_s"], 1.2,
           label="FT training", color="C2", alpha=0.85)
    ax.bar([s - 0.6 for s in sizes], table["ft_inference_s"],
           1.2, bottom=table["ft_training_s"],
           label="FT inference", color="C2", alpha=0.45)
    ax.bar([s + 0.6 for s in sizes], table["debias_compute_s"], 1.2,
           label="Debias inference (sum 11 projects)", color="C3", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}B" for s in sizes])
    ax.set_xlabel("Model size")
    ax.set_ylabel("GPU-seconds")
    ax.set_title("Compute cost: FT-ag vs Debias-ps")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.plot(sizes, table["ft_f1"], marker="D", color="C2", label="FT agnostic", linewidth=2)
    ax.plot(sizes, table["debias_f1"], marker="^", color="C3",
            label="Debias project-specific (best k)", linewidth=2)
    for x, dbf, ftf in zip(sizes, table["debias_f1"], table["ft_f1"]):
        ax.annotate(f"Δ={dbf-ftf:+.3f}", (x, max(dbf, ftf) + 0.005),
                    ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}B" for s in sizes])
    ax.set_xlabel("Model size")
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1: FT-ag vs Debias-ps")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("RQ3.6 — Debias-ps reaches FT-ag-or-better F1 with no training")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    cost_index = pd.read_csv(DOCS_ANALYSIS / "cost_index.csv")
    cells = pd.read_csv(DOCS_ANALYSIS / "all_cells.csv")
    cost = _read_costs(cost_index)
    table = _build_table(cost, cells)
    table.to_csv(DOCS_ANALYSIS / "rq3_cost_reframe.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq3_cost_reframe.csv')}")

    _plot(table, FIGURES / "rq3_cost_reframe.png")
    print("wrote rq3_cost_reframe.png")

    print("\n--- Cost reframe summary ---")
    show = table[[
        "model", "debias_k", "debias_compute_s", "debias_f1",
        "ft_training_s", "ft_inference_s", "ft_compute_s", "ft_f1",
        "delta_f1", "compute_ratio_deb_over_ft",
    ]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()

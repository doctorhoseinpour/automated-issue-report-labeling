"""RQ2.2: per-project leaderboard heatmap and table.

Computes per-project macro F1 for each (model, approach, k, project).
Agnostic cells are sliced from predictions CSVs; project-specific cells
read directly from eval CSVs. VTAG agnostic is sliced; VTAG project-specific
is direct.

Outputs:
  docs/analysis/per_project_metrics.csv    # one row per (model, approach, k, project)
  docs/analysis/rq2_per_project_table.csv  # best-k per (model, approach, project)
  docs/analysis/figures/rq2_per_project_heatmap_qwen32b.png
  docs/analysis/figures/rq2_per_project_heatmap_qwen{3B,7B,14B,32B}.png
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
    compute_metrics,
    ensure_dirs,
    load_predictions,
    project_for_test_idx,
    rel,
)


def _agnostic_per_project(pred_path: Path, idx_to_proj: dict[int, str]) -> dict[str, dict]:
    """Slice agnostic predictions per project and compute metrics."""
    df = load_predictions(pred_path)
    df = df.dropna(subset=["test_idx"])
    df["test_idx"] = df["test_idx"].astype(int)
    df["project"] = df["test_idx"].map(idx_to_proj)
    out: dict[str, dict] = {}
    for proj in PROJECTS:
        sub = df[df["project"] == proj]
        if len(sub) == 0:
            continue
        out[proj] = compute_metrics(
            sub["ground_truth"].tolist(), sub["predicted_label"].tolist(),
        )
    return out


def _build_per_project_metrics(cells: pd.DataFrame, preds_index: pd.DataFrame) -> pd.DataFrame:
    """Build the long-form table of per-project metrics for every cell."""
    idx_to_proj = project_for_test_idx()
    rows: list[dict] = []

    # Project-specific cells: take metrics from the eval CSV directly.
    ps = cells[(cells["setting"] == "project_specific") & (cells["project"].isin(PROJECTS))]
    for _, r in ps.iterrows():
        rows.append({
            "model": r["model"],
            "approach": r["approach"],
            "setting": r["setting"],
            "project": r["project"],
            "k_label": r["k_label"],
            "k": r.get("k"),
            "f1_macro": r["f1_macro"],
            "accuracy": r["accuracy"],
            "f1_bug": r["f1_bug"],
            "f1_feature": r["f1_feature"],
            "f1_question": r["f1_question"],
            "precision_bug": r["precision_bug"],
            "recall_bug": r["recall_bug"],
            "precision_feature": r["precision_feature"],
            "recall_feature": r["recall_feature"],
            "precision_question": r["precision_question"],
            "recall_question": r["recall_question"],
            "invalid_rate": r["invalid_rate"],
            "source": "eval_csv",
        })

    # Agnostic cells: slice predictions per project.
    ag_index = preds_index[preds_index["setting"] == "agnostic"]
    print(f"Slicing {len(ag_index)} agnostic predictions per project...")
    for i, (_, r) in enumerate(ag_index.iterrows(), 1):
        pred_path = REPO_ROOT / r["predictions_path"]
        if not pred_path.is_file():
            continue
        try:
            per_proj = _agnostic_per_project(pred_path, idx_to_proj)
        except Exception as e:
            print(f"  WARN: failed {r['predictions_path']}: {e}")
            continue
        for proj, m in per_proj.items():
            rows.append({
                "model": r["model"] or "(no LLM)",
                "approach": r["approach"],
                "setting": "agnostic",
                "project": proj,
                "k_label": r["k_label"],
                "k": r.get("k"),
                "f1_macro": m["f1_macro"],
                "accuracy": m["accuracy"],
                "f1_bug": m["f1_bug"],
                "f1_feature": m["f1_feature"],
                "f1_question": m["f1_question"],
                "precision_bug": m["precision_bug"],
                "recall_bug": m["recall_bug"],
                "precision_feature": m["precision_feature"],
                "recall_feature": m["recall_feature"],
                "precision_question": m["precision_question"],
                "recall_question": m["recall_question"],
                "invalid_rate": m["invalid_rate"],
                "source": "derived_from_predictions",
            })
        if i % 20 == 0:
            print(f"  ... {i}/{len(ag_index)}")
    return pd.DataFrame(rows)


def _best_k_per_cell(per_proj: pd.DataFrame, k_filter: list[str] | None = None) -> pd.DataFrame:
    """For each (model, approach, setting, project): pick row with highest f1_macro across allowed k.

    NaN model values (VTAG / VTAG-debias rows) are kept by filling with a sentinel.
    """
    df = per_proj.copy()
    if k_filter is not None:
        df = df[df["k_label"].isin(k_filter)]
    if df.empty:
        return df
    df = df.assign(_model_fill=df["model"].fillna("_no_model"))
    idx = df.groupby(["_model_fill", "approach", "setting", "project"])["f1_macro"].idxmax()
    return df.loc[idx].drop(columns="_model_fill").reset_index(drop=True)


def _heatmap(per_proj_best: pd.DataFrame, model: str, out: Path) -> None:
    """Heatmap: rows=projects, cols=approach×setting; one per Qwen size + VTAG floor row."""
    # Approach columns to show:
    cols: list[tuple[str, str, str]] = [
        ("vtag", "agnostic", "VTAG ag"),
        ("vtag", "project_specific", "VTAG ps"),
        ("zero_shot", "agnostic", "ZS ag"),
        ("ragtag", "agnostic", "RAGTAG ag"),
        ("ragtag", "project_specific", "RAGTAG ps"),
        ("ragtag_debias", "project_specific", "Debias ps"),
        ("ft", "agnostic", "FT ag"),
        ("ft", "project_specific", "FT ps"),
    ]
    rows: list[str] = list(PROJECTS)
    matrix = np.full((len(rows), len(cols)), np.nan)
    for j, (ap, setting, _) in enumerate(cols):
        if ap in ("vtag", "vtag_debias"):
            sub = per_proj_best[
                (per_proj_best["approach"] == ap)
                & (per_proj_best["setting"] == setting)
            ]
        elif ap == "zero_shot":
            sub = per_proj_best[
                (per_proj_best["model"] == model)
                & (per_proj_best["approach"] == "ragtag")
                & (per_proj_best["setting"] == "agnostic")
                & (per_proj_best["k_label"] == "zero_shot")
            ]
        else:
            sub = per_proj_best[
                (per_proj_best["model"] == model)
                & (per_proj_best["approach"] == ap)
                & (per_proj_best["setting"] == setting)
                & (per_proj_best["k_label"] != "zero_shot")
            ]
        for _, r in sub.iterrows():
            if r["project"] in rows:
                matrix[rows.index(r["project"]), j] = r["f1_macro"]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.3, vmax=0.85)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[2] for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=8)
    ax.set_title(f"Per-project macro F1 — {model}")
    fig.colorbar(im, ax=ax, label="Macro F1")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    cells = pd.read_csv(DOCS_ANALYSIS / "all_cells.csv")
    preds_index = pd.read_csv(DOCS_ANALYSIS / "preds_index.csv")

    cache = DOCS_ANALYSIS / "per_project_metrics.csv"
    if cache.is_file():
        print(f"Loading cached per-project metrics from {rel(cache)}")
        per_proj = pd.read_csv(cache)
    else:
        per_proj = _build_per_project_metrics(cells, preds_index)
        per_proj.to_csv(cache, index=False)
        print(f"wrote {rel(cache)} ({len(per_proj)} rows)")

    # Best-k slices for the leaderboard table.
    # Retrieval-based: use {k1,k3,k6,k9}. Zero-shot/ft: just one k_label each.
    parts: list[pd.DataFrame] = []
    parts.append(_best_k_per_cell(
        per_proj[per_proj["approach"].isin(["ragtag"])],
        k_filter=["k1", "k3", "k6", "k9"],
    ))
    parts.append(_best_k_per_cell(
        per_proj[per_proj["approach"] == "ragtag_debias"],
        k_filter=["k1", "k3", "k6", "k9"],
    ))
    parts.append(_best_k_per_cell(per_proj[per_proj["approach"] == "ft"]))
    # VTAG: all k values are valid; pick best k per project per setting.
    parts.append(_best_k_per_cell(per_proj[per_proj["approach"] == "vtag"]))
    parts.append(_best_k_per_cell(per_proj[per_proj["approach"] == "vtag_debias"]))
    # Zero-shot rows from ragtag k_label='zero_shot' (agnostic only)
    zs = per_proj[(per_proj["approach"] == "ragtag") & (per_proj["k_label"] == "zero_shot")]
    parts.append(zs)

    best = pd.concat(parts, ignore_index=True)
    best.to_csv(DOCS_ANALYSIS / "rq2_per_project_table.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq2_per_project_table.csv')} ({len(best)} rows)")

    for model in MODEL_ORDER:
        out = FIGURES / f"rq2_per_project_heatmap_{model.lower().replace('-', '')}.png"
        _heatmap(best, model, out)
        print(f"wrote {rel(out)}")

    # Project hardness summary: mean F1 across all (model, approach) for each project.
    hardness = best.groupby("project")["f1_macro"].agg(["mean", "min", "max", "std"]).reset_index()
    hardness = hardness.sort_values("mean")
    hardness.to_csv(DOCS_ANALYSIS / "rq2_project_hardness.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq2_project_hardness.csv')}")
    print("\nProject hardness (sorted easiest → hardest):")
    print(hardness.to_string(index=False))


if __name__ == "__main__":
    main()

"""RQ3.5: VTAG-debias mechanism ablation.

Side-by-side per-class precision/recall deltas:
  VTAG-debias − VTAG plain   (no LLM, mechanism reduces to retrieval rebalance)
  RAGTAG-debias − RAGTAG plain  (LLM consumes the rebalanced few-shots)

The contrast: VTAG-debias trades bug-recall for question-recall almost 1:1
(macro F1 gain is marginal because the bug-recall loss cancels question-recall
gains). RAGTAG-debias gives a clean macro F1 gain because the LLM rescues
true-bug cases from the rebalanced few-shots.

Quantify rescue: of issues where VTAG-debias flipped bug→question (true
bug), how often did RAGTAG-debias correctly predict bug?

Outputs:
  docs/analysis/rq3_vtag_debias_perclass.csv
  docs/analysis/rq3_llm_rescue.csv
  docs/analysis/figures/rq3_mechanism_ablation.png
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
    VALID_LABELS,
    ensure_dirs,
    load_predictions,
    rel,
)


K_LABELS = ["k1", "k3", "k6", "k9"]


def _ps_aggregate(cells: pd.DataFrame, approach: str, k_label: str, model: str = "(no LLM)") -> dict | None:
    if approach in ("vtag", "vtag_debias"):
        sub = cells[
            (cells["approach"] == approach)
            & (cells["setting"] == "project_specific")
            & (cells["k_label"] == k_label)
        ]
    else:
        sub = cells[
            (cells["model"] == model)
            & (cells["approach"] == approach)
            & (cells["setting"] == "project_specific")
            & (cells["k_label"] == k_label)
        ]
    if sub.empty:
        return None
    cols = ["f1_macro"]
    for cls in VALID_LABELS:
        cols += [f"precision_{cls}", f"recall_{cls}", f"f1_{cls}"]
    return {c: sub[c].mean() for c in cols}


def _ag_metrics(cells: pd.DataFrame, approach: str, k_label: str, model: str = "(no LLM)") -> dict | None:
    if approach in ("vtag", "vtag_debias"):
        sub = cells[
            (cells["approach"] == approach)
            & (cells["setting"] == "agnostic")
            & (cells["project"] == "_overall")
            & (cells["k_label"] == k_label)
        ]
    else:
        sub = cells[
            (cells["model"] == model)
            & (cells["approach"] == approach)
            & (cells["setting"] == "agnostic")
            & (cells["project"] == "_overall")
            & (cells["k_label"] == k_label)
        ]
    if sub.empty:
        return None
    cols = ["f1_macro"]
    for cls in VALID_LABELS:
        cols += [f"precision_{cls}", f"recall_{cls}", f"f1_{cls}"]
    return {c: float(sub[c].iloc[0]) for c in cols}


def _build_perclass_deltas(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    # VTAG: agnostic and ps
    for setting in ("agnostic", "project_specific"):
        for k_label in K_LABELS:
            if setting == "agnostic":
                plain = _ag_metrics(cells, "vtag", k_label)
                deb = _ag_metrics(cells, "vtag_debias", k_label)
            else:
                plain = _ps_aggregate(cells, "vtag", k_label)
                deb = _ps_aggregate(cells, "vtag_debias", k_label)
            if plain is None or deb is None:
                continue
            r = {"system": "VTAG", "setting": setting, "model": "(no LLM)",
                 "k_label": k_label}
            for col in plain.keys():
                r[f"delta_{col}"] = deb[col] - plain[col]
            rows.append(r)
    # RAGTAG: project_specific (debias only exists ps)
    for model in MODEL_ORDER:
        for k_label in K_LABELS:
            plain = _ps_aggregate(cells, "ragtag", k_label, model=model)
            deb = _ps_aggregate(cells, "ragtag_debias", k_label, model=model)
            if plain is None or deb is None:
                continue
            r = {"system": f"RAGTAG-{model}", "setting": "project_specific", "model": model,
                 "k_label": k_label}
            for col in plain.keys():
                r[f"delta_{col}"] = deb[col] - plain[col]
            rows.append(r)
    return pd.DataFrame(rows)


def _plot_mechanism(perclass: pd.DataFrame, out: Path) -> None:
    """Two-panel: per-class recall delta at k=9 for VTAG-debias and each RAGTAG-debias."""
    sub = perclass[perclass["k_label"] == "k9"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    metric_titles = ("delta_recall", "delta_precision")
    for ax, metric in zip(axes, metric_titles):
        # Build series for each system
        systems_order = ["VTAG (ag)", "VTAG (ps)"] + [f"RAGTAG-{m} (ps)" for m in MODEL_ORDER]
        x = np.arange(len(VALID_LABELS))
        width = 0.11
        labels: list[str] = []
        for i, system in enumerate(systems_order):
            if system == "VTAG (ag)":
                row = sub[(sub["system"] == "VTAG") & (sub["setting"] == "agnostic")]
            elif system == "VTAG (ps)":
                row = sub[(sub["system"] == "VTAG") & (sub["setting"] == "project_specific")]
            else:
                model = system.replace("RAGTAG-", "").replace(" (ps)", "")
                row = sub[(sub["system"] == f"RAGTAG-{model}") & (sub["setting"] == "project_specific")]
            if row.empty:
                continue
            r = row.iloc[0]
            vals = [r[f"{metric}_{cls}"] for cls in VALID_LABELS]
            ax.bar(x + i * width, vals, width, label=system)
            labels.append(system)
        ax.axhline(0, color="black", linestyle="--", alpha=0.4)
        ax.set_xticks(x + width * (len(systems_order) - 1) / 2)
        ax.set_xticklabels(VALID_LABELS)
        ax.set_ylabel(f"Δ {metric.replace('delta_', '')} (debias − plain) at k=9")
        ax.set_title(f"Per-class {metric.replace('delta_', '')} delta")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("RQ3.5 mechanism ablation — VTAG-debias trades, RAGTAG-debias rescues")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _llm_rescue_rate(preds_index: pd.DataFrame) -> pd.DataFrame:
    """For each Qwen size: of issues where VTAG-debias flipped a true-bug
    prediction from bug to question (vs plain VTAG), what fraction of those
    issues did the model's RAGTAG-debias correctly keep as bug?

    Both VTAG signals are agnostic; RAGTAG-debias is ps. Align by global test_idx.
    """
    # Load VTAG ag k9 plain and debias predictions.
    def load_vtag(approach: str) -> pd.DataFrame:
        rows = preds_index[
            (preds_index["approach"] == approach)
            & (preds_index["setting"] == "agnostic")
            & (preds_index["k_label"] == "k9")
        ]
        if rows.empty:
            return pd.DataFrame()
        return load_predictions(REPO_ROOT / rows.iloc[0]["predictions_path"])

    vtag_plain = load_vtag("vtag")
    vtag_deb = load_vtag("vtag_debias")
    if vtag_plain.empty or vtag_deb.empty:
        return pd.DataFrame()
    vtag_plain = vtag_plain.set_index("test_idx")[
        ["ground_truth", "predicted_label"]
    ].rename(columns={"predicted_label": "vtag_plain"})
    vtag_deb = vtag_deb.set_index("test_idx")[["predicted_label"]].rename(
        columns={"predicted_label": "vtag_debias"}
    )
    base = vtag_plain.join(vtag_deb, how="inner")

    def lower(s: pd.Series) -> pd.Series:
        return s.astype(str).str.lower().str.strip()

    base["gt"] = lower(base["ground_truth"])
    base["vp"] = lower(base["vtag_plain"])
    base["vd"] = lower(base["vtag_debias"])

    # The flip set: GT=bug AND VTAG plain predicted bug AND VTAG debias predicted question.
    flips = base[(base["gt"] == "bug") & (base["vp"] == "bug") & (base["vd"] == "question")]
    print(f"  flip set (GT=bug, VTAG plain=bug, VTAG debias=question): {len(flips)} issues")

    # For each Qwen model: load RAGTAG-debias ps k9 predictions, see how often it predicts bug on the flip set.
    # Map flip global_idx → ps local_idx via project membership.
    from _utils import project_for_test_idx
    idx_to_proj = project_for_test_idx()

    rows: list[dict] = []
    for model in MODEL_ORDER:
        flip_global = list(flips.index)
        # Find each flip's project.
        flip_by_proj: dict[str, list[int]] = {}
        for gi in flip_global:
            proj = idx_to_proj.get(int(gi))
            if proj:
                flip_by_proj.setdefault(proj, []).append(int(gi))

        n_total = 0
        n_rescued = 0
        for proj, gids in flip_by_proj.items():
            sub = preds_index[
                (preds_index["model"] == model)
                & (preds_index["approach"] == "ragtag_debias")
                & (preds_index["setting"] == "project_specific")
                & (preds_index["project"] == proj)
                & (preds_index["k_label"] == "k9")
            ]
            if sub.empty:
                continue
            df = load_predictions(REPO_ROOT / sub.iloc[0]["predictions_path"])
            df = df.dropna(subset=["test_idx"])
            df["test_idx"] = df["test_idx"].astype(int)
            df = df.set_index("test_idx")
            # For ps preds, test_idx is local 0..299; map global → local.
            from _utils import RESULTS_DIR  # local import to keep imports compact
            ts = pd.read_csv(RESULTS_DIR / "agnostic" / "neighbors" / "test_split.csv")
            repo = proj.replace("_", "/", 1)
            ag_idx_for_proj = ts.index[ts["repo"] == repo].tolist()  # list of global indices
            global_to_local = {g: i for i, g in enumerate(ag_idx_for_proj)}
            for gi in gids:
                li = global_to_local.get(gi)
                if li is None or li not in df.index:
                    continue
                pred = str(df.loc[li, "predicted_label"]).lower().strip()
                n_total += 1
                if pred == "bug":
                    n_rescued += 1
        rate = n_rescued / n_total if n_total else float("nan")
        rows.append({
            "model": model,
            "n_flip_set": n_total,
            "n_rescued_to_bug": n_rescued,
            "rescue_rate": rate,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    cells = pd.read_csv(DOCS_ANALYSIS / "all_cells.csv")
    preds_index = pd.read_csv(DOCS_ANALYSIS / "preds_index.csv")

    perclass = _build_perclass_deltas(cells)
    perclass.to_csv(DOCS_ANALYSIS / "rq3_vtag_debias_perclass.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq3_vtag_debias_perclass.csv')} ({len(perclass)} rows)")

    _plot_mechanism(perclass, FIGURES / "rq3_mechanism_ablation.png")
    print("wrote rq3_mechanism_ablation.png")

    print("\n--- VTAG vs RAGTAG debias mechanism comparison (k=9) ---")
    sub = perclass[perclass["k_label"] == "k9"].copy()
    cols_show = ["system", "setting",
                 "delta_recall_bug", "delta_recall_feature", "delta_recall_question",
                 "delta_precision_bug", "delta_precision_feature", "delta_precision_question",
                 "delta_f1_macro"]
    print(sub[cols_show].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\nComputing LLM rescue rate (this needs predictions joins)...")
    rescue = _llm_rescue_rate(preds_index)
    if not rescue.empty:
        rescue.to_csv(DOCS_ANALYSIS / "rq3_llm_rescue.csv", index=False)
        print(f"wrote {rel(DOCS_ANALYSIS / 'rq3_llm_rescue.csv')}")
        print("\n--- LLM rescue rate ---")
        print(rescue.to_string(index=False))


if __name__ == "__main__":
    main()

"""Generate paper/figures/finetune_comparison.{pdf,png} with POOLED PS aggregation.

Two-panel figure for the fine-tuning analysis:
  (left)  Macro F1 bar chart at each model: \\ragtag-PS (best k) vs FT-PS vs FT-PA
  (right) Per-class F1 (bug / feature / question) for FT-PS vs FT-PA, showing
          which classes benefit most from PA's larger training pool

Convention: pooled raw, no \\votag-rescue. See paper/sections/04_setup.tex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rescue import load_raw_preds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "issues11k"
FIG_DIR = REPO_ROOT / "paper" / "figures"

LABELS = ["bug", "feature", "question"]
KS_RAG = [1, 3, 6, 9, 12, 15]

MODELS = [
    ("unsloth_Qwen2_5_3B_Instruct_bnb_4bit",  "Qwen-3B"),
    ("unsloth_Qwen2_5_7B_Instruct_bnb_4bit",  "Qwen-7B"),
    ("unsloth_Qwen2_5_14B_Instruct_bnb_4bit", "Qwen-14B"),
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "Qwen-32B"),
]

# Color palette: RAGTAG-PS = blue (matches our two-category palette in §5.4),
# FT-PS = light orange, FT-PA = darker orange — orange family for fine-tuning.
RAG_COLOR    = "#4C72B0"  # blue
FT_PS_COLOR  = "#FBB04E"  # light amber
FT_PA_COLOR  = "#D17B0F"  # darker orange


def _macro(df) -> float:
    return f1_score(df["ground_truth"], df["predicted_label"],
                    labels=LABELS, average="macro", zero_division=0)


def _per_class(df) -> dict[str, float]:
    _, _, f1, _ = precision_recall_fscore_support(
        df["ground_truth"], df["predicted_label"],
        labels=LABELS, zero_division=0)
    return dict(zip(LABELS, f1))


def _projects() -> list[str]:
    return sorted(p.name for p in (RESULTS / "project_specific").iterdir() if p.is_dir())


def _ft_pa(model) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"agnostic/{model}/finetune_fixed/preds_finetune_fixed.csv",
                       usecols=["ground_truth", "predicted_label"])


def _ft_ps(model) -> pd.DataFrame:
    parts = [
        pd.read_csv(RESULTS / f"project_specific/{p}/{model}/finetune_fixed/preds_finetune_fixed.csv",
                    usecols=["ground_truth", "predicted_label"])
        for p in _projects()
    ]
    return pd.concat(parts, ignore_index=True)


def _build():
    rows = {}
    for model, lbl in MODELS:
        rag_k = max(KS_RAG, key=lambda k: _macro(load_raw_preds(model, "PS", k, "ragtag")))
        rag_df = load_raw_preds(model, "PS", rag_k, "ragtag")
        ftps_df = _ft_ps(model)
        ftpa_df = _ft_pa(model)
        rows[lbl] = {
            "rag_k": rag_k,
            "rag_macro": _macro(rag_df),
            "ftps_macro": _macro(ftps_df),
            "ftpa_macro": _macro(ftpa_df),
            "ftps_pc": _per_class(ftps_df),
            "ftpa_pc": _per_class(ftpa_df),
        }
    return rows


def _plot(rows, out_pdf, out_png):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 4.8),
                                      gridspec_kw={"width_ratios": [1.0, 1.4]})

    n_models = len(MODELS)
    x = np.arange(n_models)

    # ---- Left: macro F1 (3 bars per model: RAGTAG-PS, FT-PS, FT-PA) ----
    width = 0.27
    rag_vals = [rows[lbl]["rag_macro"] for _, lbl in MODELS]
    ftps_vals = [rows[lbl]["ftps_macro"] for _, lbl in MODELS]
    ftpa_vals = [rows[lbl]["ftpa_macro"] for _, lbl in MODELS]
    bars1 = ax_l.bar(x - width, rag_vals, width, color=RAG_COLOR,
                     edgecolor="0.3", linewidth=0.5, label="RAGTAG-PS (best $k$)")
    bars2 = ax_l.bar(x, ftps_vals, width, color=FT_PS_COLOR,
                     edgecolor="0.3", linewidth=0.5, label="Fine-Tune PS")
    bars3 = ax_l.bar(x + width, ftpa_vals, width, color=FT_PA_COLOR,
                     edgecolor="0.3", linewidth=0.5, label="Fine-Tune PA")
    ax_l.bar_label(bars1, fmt="%.3f", padding=2, fontsize=7.5)
    ax_l.bar_label(bars2, fmt="%.3f", padding=2, fontsize=7.5)
    ax_l.bar_label(bars3, fmt="%.3f", padding=2, fontsize=7.5)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels([lbl for _, lbl in MODELS], fontsize=9)
    ax_l.set_ylabel("Macro $F_1$ (pooled)")
    ax_l.set_title("(a) Macro $F_1$: RAGTAG-PS vs Fine-Tune", fontsize=10, fontweight="bold")
    all_vals = rag_vals + ftps_vals + ftpa_vals
    ax_l.set_ylim(min(all_vals) * 0.94, max(all_vals) * 1.06)
    ax_l.grid(True, axis="y", alpha=0.3)
    ax_l.legend(loc="upper left", fontsize=8.5, frameon=False)

    # ---- Right: per-class F1 DELTA (FT-PA - FT-PS), 3 bars per model ----
    # One bar per class per model, height = FT-PA - FT-PS. Sign-aware (Qwen-32B
    # question is negative). Class differentiated by color.
    CLASS_COLORS = {
        "bug":      "#C44E52",  # red
        "feature":  "#55A868",  # green
        "question": "#8172B2",  # purple
    }
    width2 = 0.26
    class_offsets = {"bug": -width2, "feature": 0.0, "question": +width2}

    for cls in LABELS:
        deltas = [rows[lbl]["ftpa_pc"][cls] - rows[lbl]["ftps_pc"][cls]
                  for _, lbl in MODELS]
        bars = ax_r.bar(x + class_offsets[cls], deltas, width2,
                        color=CLASS_COLORS[cls], edgecolor="0.3", linewidth=0.5,
                        label=cls)
        ax_r.bar_label(bars, fmt="%+.3f", padding=2, fontsize=7.5)

    ax_r.axhline(0, color="0.3", linewidth=0.8)
    ax_r.set_xticks(x)
    ax_r.set_xticklabels([lbl for _, lbl in MODELS], fontsize=9)
    ax_r.set_ylabel("$\\Delta F_1$ (PA $-$ PS)")
    ax_r.set_title("(b) Per-class $F_1$ gain from Fine-Tune PS $\\to$ PA",
                   fontsize=10, fontweight="bold")
    all_deltas = [rows[lbl]["ftpa_pc"][c] - rows[lbl]["ftps_pc"][c]
                  for _, lbl in MODELS for c in LABELS]
    pad = (max(all_deltas) - min(all_deltas)) * 0.15
    ax_r.set_ylim(min(all_deltas) - pad, max(all_deltas) + pad * 1.5)
    ax_r.grid(True, axis="y", alpha=0.3)
    ax_r.legend(loc="upper right", fontsize=8.5, frameon=False, ncol=3)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = _build()
    print(f"{'model':<10}  {'RAG-PS k*':<10}  {'RAG macro':>9}  {'FT-PS macro':>11}  {'FT-PA macro':>11}")
    for _, lbl in MODELS:
        r = rows[lbl]
        print(f"{lbl:<10}  k={r['rag_k']:<6}  {r['rag_macro']:>9.4f}  {r['ftps_macro']:>11.4f}  {r['ftpa_macro']:>11.4f}")
    print()
    print(f"{'model':<10}  {'class':<8}  {'FT-PS':>6}  {'FT-PA':>6}  delta")
    for _, lbl in MODELS:
        r = rows[lbl]
        for c in LABELS:
            d = r["ftpa_pc"][c] - r["ftps_pc"][c]
            print(f"{lbl:<10}  {c:<8}  {r['ftps_pc'][c]:>6.3f}  {r['ftpa_pc'][c]:>6.3f}  " + f"{d:+.3f}")
        print()

    out_pdf = FIG_DIR / "finetune_comparison.pdf"
    out_png = FIG_DIR / "finetune_comparison.png"
    _plot(rows, out_pdf, out_png)
    print(f"wrote {out_pdf.relative_to(REPO_ROOT)}")
    print(f"wrote {out_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

"""Generate paper/figures/ragtag_perclass.{pdf,png} with POOLED PS aggregation.

Per-class F1 (bug / feature / question) for each Qwen size at its best
RAGTAG-PS k. Drives home that question consistently lags bug and feature
across every model, even at each model's best configuration.

This figure reports RAW \\ragtag\\ per-class F1 (no \\votag-rescue applied).
The rescue is reserved for the Fine-Tune comparison subsection.

Convention: pooled per-class F1 over the 3,300-issue test set, see
paper/sections/04_setup.tex.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "issues11k"
FIG_DIR = REPO_ROOT / "paper" / "figures"

LABELS = ["bug", "feature", "question"]

# Best PS k per model under the RAW metric.
MODELS = [
    # (model_tag, label, color, best_k)
    ("unsloth_Qwen2_5_3B_Instruct_bnb_4bit",  "Qwen-3B",  "#0072B2",  3),
    ("unsloth_Qwen2_5_7B_Instruct_bnb_4bit",  "Qwen-7B",  "#E69F00",  6),
    ("unsloth_Qwen2_5_14B_Instruct_bnb_4bit", "Qwen-14B", "#8c564b", 12),
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "Qwen-32B", "#D55E00", 12),
]


def _per_class_f1(df: pd.DataFrame) -> dict[str, float]:
    _, _, f1, _ = precision_recall_fscore_support(
        df["ground_truth"], df["predicted_label"],
        labels=LABELS, zero_division=0,
    )
    return dict(zip(LABELS, f1))


def _ps_pooled(model: str, k: int) -> pd.DataFrame:
    projects = sorted(p.name for p in (RESULTS / "project_specific").iterdir() if p.is_dir())
    parts = []
    for proj in projects:
        p = (RESULTS / "project_specific" / proj / model / "ragtag"
             / "predictions" / f"preds_k{k}.csv")
        parts.append(pd.read_csv(p, usecols=["ground_truth", "predicted_label"]))
    return pd.concat(parts, ignore_index=True)


def _build() -> dict:
    out = {}
    for tag, label, color, best_k in MODELS:
        df = _ps_pooled(tag, best_k)
        out[label] = {
            "color": color,
            "best_k": best_k,
            "f1": _per_class_f1(df),
        }
    return out


def _plot(data: dict, out_pdf: Path, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.6))

    n_models = len(MODELS)
    n_classes = len(LABELS)
    x = np.arange(n_classes)
    width = 0.20

    for i, (_, model_label, color, _) in enumerate(MODELS):
        f1s = [data[model_label]["f1"][lab] for lab in LABELS]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, f1s, width,
                      color=color,
                      label=f"{model_label} (k={data[model_label]['best_k']})")
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([lab.capitalize() for lab in LABELS])
    ax.set_ylabel("Per-class $F_1$ (pooled)")

    # Y-axis range to give room for value labels above the tallest bars.
    all_vals = [data[m]["f1"][lab] for _, m, _, _ in MODELS for lab in LABELS]
    ax.set_ylim(min(all_vals) * 0.92, max(all_vals) * 1.06)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=4, fontsize=9, frameon=False, borderaxespad=0)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _build()

    # Print the table.
    print(f"{'Model':<10} {'k':<3}  " + "  ".join(f"{lab:<8}" for lab in LABELS))
    print("-" * 60)
    for tag, label, color, best_k in MODELS:
        row = data[label]["f1"]
        cells = "  ".join(f"{row[lab]:<8.4f}" for lab in LABELS)
        print(f"{label:<10} {best_k:<3}  {cells}")

    out_pdf = FIG_DIR / "ragtag_perclass.pdf"
    out_png = FIG_DIR / "ragtag_perclass.png"
    _plot(data, out_pdf, out_png)
    print(f"\nwrote {out_pdf.relative_to(REPO_ROOT)}")
    print(f"wrote {out_png.relative_to(REPO_ROOT)}")

    # Headline gaps.
    print("\n--- gap to bug / gap to feature, per model ---")
    for tag, label, color, best_k in MODELS:
        f1 = data[label]["f1"]
        print(f"  {label} (k={best_k}): "
              f"Q-vs-bug = {f1['bug']-f1['question']:+.3f}, "
              f"Q-vs-feature = {f1['feature']-f1['question']:+.3f}")


if __name__ == "__main__":
    main()

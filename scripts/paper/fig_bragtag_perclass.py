"""Generate paper/figures/bragtag_perclass.{pdf,png} with POOLED PS aggregation.

Three-panel figure (one per label: bug / feature / question). Each panel has
4 model groups on the x-axis, each with 2 bars: vanilla \\ragtag at its best
k, and \\bragtag at its best k. Drives home that \\bragtag's macro F1 gain
comes from a large boost on the question class with essentially no impact on
bug or feature.

Convention: pooled raw per-class F1 (no VTAG-rescue), see paper/sections/04_setup.tex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rescue import load_raw_preds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "paper" / "figures"

LABELS = ["bug", "feature", "question"]
KS = [1, 3, 6, 9, 12, 15]

MODELS = [
    ("unsloth_Qwen2_5_3B_Instruct_bnb_4bit",  "Qwen-3B"),
    ("unsloth_Qwen2_5_7B_Instruct_bnb_4bit",  "Qwen-7B"),
    ("unsloth_Qwen2_5_14B_Instruct_bnb_4bit", "Qwen-14B"),
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "Qwen-32B"),
]

# Match the C0/C1 palette used in fig:vtag-kcurve so two-category comparisons
# look consistent across the paper. Blue = baseline (vanilla), orange = the
# intervention (BRAGTAG).
VANILLA_COLOR = "C0"  # matplotlib default blue
BRAGTAG_COLOR = "C1"  # matplotlib default orange


def _macro_f1(df) -> float:
    return f1_score(df["ground_truth"], df["predicted_label"],
                    labels=LABELS, average="macro", zero_division=0)


def _per_class_f1(df) -> dict[str, float]:
    _, _, f1, _ = precision_recall_fscore_support(
        df["ground_truth"], df["predicted_label"],
        labels=LABELS, zero_division=0,
    )
    return dict(zip(LABELS, f1))


def _build() -> dict:
    """For each model, find best k for vanilla and bragtag, then pull per-class F1."""
    out = {}
    for tag, lbl in MODELS:
        v_k = max(KS, key=lambda k: _macro_f1(load_raw_preds(tag, "PS", k, "ragtag")))
        b_k = max(KS, key=lambda k: _macro_f1(load_raw_preds(tag, "PS", k, "ragtag_debias_m3")))
        v_pc = _per_class_f1(load_raw_preds(tag, "PS", v_k, "ragtag"))
        b_pc = _per_class_f1(load_raw_preds(tag, "PS", b_k, "ragtag_debias_m3"))
        out[lbl] = {"v_k": v_k, "b_k": b_k, "v": v_pc, "b": b_pc}
    return out


def _plot(data: dict, out_pdf: Path, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4), sharey=True)
    n_models = len(MODELS)
    x = np.arange(n_models)
    width = 0.35

    for ax, lab in zip(axes, LABELS):
        v_vals = [data[lbl]["v"][lab] for _, lbl in MODELS]
        b_vals = [data[lbl]["b"][lab] for _, lbl in MODELS]
        bars_v = ax.bar(x - width/2, v_vals, width, color=VANILLA_COLOR,
                        edgecolor="0.3", linewidth=0.5,
                        label="Vanilla \\ragtag")
        bars_b = ax.bar(x + width/2, b_vals, width, color=BRAGTAG_COLOR,
                        edgecolor="0.2", linewidth=0.5,
                        label="\\bragtag")
        ax.bar_label(bars_v, fmt="%.3f", padding=2, fontsize=7.5)
        ax.bar_label(bars_b, fmt="%.3f", padding=2, fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in MODELS], fontsize=9)
        ax.set_title(lab.capitalize(), fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel("Per-class $F_1$ (pooled)")

    # Shared y-limits with headroom for value labels
    all_vals = [data[lbl][m][c] for _, lbl in MODELS for m in ["v","b"] for c in LABELS]
    axes[0].set_ylim(min(all_vals) * 0.92, max(all_vals) * 1.06)

    # Single shared legend at the top
    handles = [
        plt.Rectangle((0,0), 1, 1, color=VANILLA_COLOR, ec="0.3", lw=0.5),
        plt.Rectangle((0,0), 1, 1, color=BRAGTAG_COLOR, ec="0.2", lw=0.5),
    ]
    fig.legend(handles, ["RAGTAG (best k)", "BRAGTAG (best k)"],
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, fontsize=10, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _build()

    print(f"{'model':<10}  {'method':<8}  {'best k':<6}  " + "  ".join(f"{c:<8}" for c in LABELS))
    for tag, lbl in MODELS:
        v = data[lbl]["v"]; b = data[lbl]["b"]
        print(f"{lbl:<10}  vanilla   k={data[lbl]['v_k']:<4}  " + "  ".join(f"{v[c]:<8.3f}" for c in LABELS))
        print(f"{lbl:<10}  bragtag   k={data[lbl]['b_k']:<4}  " + "  ".join(f"{b[c]:<8.3f}" for c in LABELS))

    out_pdf = FIG_DIR / "bragtag_perclass.pdf"
    out_png = FIG_DIR / "bragtag_perclass.png"
    _plot(data, out_pdf, out_png)
    print(f"\nwrote {out_pdf.relative_to(REPO_ROOT)}")
    print(f"wrote {out_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

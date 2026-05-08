"""Generate paper/figures/bragtag_kcurve.{pdf,png} with POOLED PS aggregation.

Single-panel figure: macro F1 vs k for all four Qwen sizes, with vanilla
\\ragtag (dashed thin line, hollow markers) overlaid against \\bragtag (solid
line, filled markers) in the same color per model. The two-line-per-model
overlay visually communicates "BRAGTAG outperforms vanilla at every (model,
k>=6)" while making the k=1, k=3 underperformance visible.

Convention: pooled raw macro F1 (no VTAG-rescue), see paper/sections/04_setup.tex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rescue import load_raw_preds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "paper" / "figures"

LABELS = ["bug", "feature", "question"]

MODELS = [
    # (model_tag, label, color, marker)
    ("unsloth_Qwen2_5_3B_Instruct_bnb_4bit",  "Qwen-3B",  "#0072B2", "o"),
    ("unsloth_Qwen2_5_7B_Instruct_bnb_4bit",  "Qwen-7B",  "#E69F00", "s"),
    ("unsloth_Qwen2_5_14B_Instruct_bnb_4bit", "Qwen-14B", "#8c564b", "^"),
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "Qwen-32B", "#D55E00", "D"),
]

K_VALUES = [1, 3, 6, 9, 12, 15]


def _macro_f1(df: pd.DataFrame) -> float:
    return f1_score(
        df["ground_truth"], df["predicted_label"],
        average="macro", labels=LABELS, zero_division=0,
    )


def _build_curves() -> dict:
    out = {"vanilla": {}, "bragtag": {}}
    for tag, _, _, _ in MODELS:
        out["vanilla"][tag] = {k: _macro_f1(load_raw_preds(tag, "PS", k, "ragtag"))
                                for k in K_VALUES}
        out["bragtag"][tag] = {k: _macro_f1(load_raw_preds(tag, "PS", k, "ragtag_debias_m3"))
                                for k in K_VALUES}
    return out


def _plot(curves: dict, out_pdf: Path, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for tag, label, color, marker in MODELS:
        # BRAGTAG (solid, filled marker)
        bragtag_ys = [curves["bragtag"][tag][k] for k in K_VALUES]
        ax.plot(K_VALUES, bragtag_ys, color=color, linestyle="-", linewidth=1.8,
                marker=marker, markersize=6.0,
                markerfacecolor=color, markeredgecolor=color,
                label=label)
        # Vanilla (dashed thin, hollow marker)
        vanilla_ys = [curves["vanilla"][tag][k] for k in K_VALUES]
        ax.plot(K_VALUES, vanilla_ys, color=color, linestyle="--", linewidth=1.0,
                marker=marker, markersize=6.0,
                markerfacecolor="white", markeredgecolor=color,
                alpha=0.55)

    ax.set_xlabel(r"$k$ (number of retrieved few-shot neighbors)")
    ax.set_ylabel("Macro $F_1$ (pooled)")
    ax.set_xticks(K_VALUES)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 15.5)

    # Legend: model colors + linestyle key
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=1.8,
               marker="o", markersize=6, markerfacecolor="black",
               label="BRAGTAG (solid, filled)"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0,
               marker="o", markersize=6, markerfacecolor="white",
               markeredgecolor="black", alpha=0.7,
               label="RAGTAG (dashed, hollow)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    handles = handles + style_handles
    labels = labels + ["BRAGTAG (solid, filled)", "RAGTAG (dashed, hollow)"]
    ax.legend(handles, labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=3,
              fontsize=9, frameon=False, borderaxespad=0)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    curves = _build_curves()

    print(f"{'model':<10}  {'method':<8}  " + "  ".join(f"k={k:<3}" for k in K_VALUES))
    for tag, label, _, _ in MODELS:
        for method in ("vanilla", "bragtag"):
            row = curves[method][tag]
            cells = "  ".join(f"{row[k]:.4f}" for k in K_VALUES)
            print(f"{label:<10}  {method:<8}  {cells}")

    out_pdf = FIG_DIR / "bragtag_kcurve.pdf"
    out_png = FIG_DIR / "bragtag_kcurve.png"
    _plot(curves, out_pdf, out_png)
    print(f"\nwrote {out_pdf.relative_to(REPO_ROOT)}")
    print(f"wrote {out_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

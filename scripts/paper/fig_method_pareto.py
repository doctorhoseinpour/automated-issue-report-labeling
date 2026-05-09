"""Generate paper/figures/method_pareto.{pdf,png} — accuracy vs cost.

Single figure with four sub-panels (one per Qwen size). Each panel plots
macro F1 (y) vs total GPU time in hours (x) for the three methods at their
best configuration with the \\votag\\ fallback applied:
  - \\ragtag-PS at best k        (blue circle)
  - \\bragtag-PS at best k        (green square)
  - Fine-Tune-PA                  (orange triangle)

Reads the same data the table reads via tab_method_comparison helpers, so
numbers stay in sync.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tab_method_comparison import (  # noqa: E402
    MODELS,
    _row_few_shot,
    _row_finetune,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "paper" / "figures"

METHOD_STYLE = {
    "ragtag":           {"color": "#4C72B0", "marker": "o", "label": r"\ragtag-PS"},
    "ragtag_debias_m3": {"color": "#55A868", "marker": "s", "label": r"\bragtag-PS"},
    "finetune":         {"color": "#D17B0F", "marker": "^", "label": "Fine-Tune-PA"},
}
DISPLAY_LABEL = {  # plain-text labels (matplotlib doesn't expand LaTeX macros)
    "ragtag":           "RAGTAG-PS",
    "ragtag_debias_m3": "BRAGTAG-PS",
    "finetune":         "Fine-Tune-PA",
}


def _build():
    out = {}
    for tag, lbl in MODELS:
        rows = [
            _row_few_shot(tag, lbl, "ragtag"),
            _row_few_shot(tag, lbl, "ragtag_debias_m3"),
            _row_finetune(tag, lbl),
        ]
        out[lbl] = rows
    return out


def _plot(rows, out_pdf, out_png):
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.4), sharey=False)
    for ax, (_, model_lbl) in zip(axes, MODELS):
        for r in rows[model_lbl]:
            style = METHOD_STYLE[r["method"]]
            ax.scatter(
                r["gpu_time_s"] / 3600.0,
                r["macro"],
                color=style["color"],
                marker=style["marker"],
                s=110,
                edgecolor="0.2",
                linewidth=0.6,
                zorder=3,
            )
        ax.set_title(model_lbl, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("GPU time (h)", fontsize=9)

        # Annotate each point with macro F1 to the right of the marker.
        # Side of label chosen so text doesn't run off the right edge.
        times = [r["gpu_time_s"] / 3600.0 for r in rows[model_lbl]]
        x_pad = (max(times) - min(times)) * 0.06
        for r in rows[model_lbl]:
            t_h = r["gpu_time_s"] / 3600.0
            on_right = t_h <= (min(times) + max(times)) / 2
            ha = "left" if on_right else "right"
            dx = x_pad if on_right else -x_pad
            ax.annotate(
                f"{r['macro']:.3f}",
                (t_h, r["macro"]),
                xytext=(dx, 0),
                textcoords="offset points",
                ha=ha, va="center",
                fontsize=8, color="0.25",
            )

        # Per-panel y-axis padding around the three points
        macros = [r["macro"] for r in rows[model_lbl]]
        y_lo, y_hi = min(macros), max(macros)
        y_pad = max(0.005, (y_hi - y_lo) * 0.6)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

        # Per-panel x-axis padding
        x_lo, x_hi = min(times), max(times)
        x_extra = max(0.05, (x_hi - x_lo) * 0.35)
        ax.set_xlim(x_lo - x_extra, x_hi + x_extra)

    axes[0].set_ylabel(r"Macro $F_1$ (pooled)", fontsize=9)

    # Single shared legend across the figure (manual, plain-text labels)
    handles = [
        plt.Line2D([0], [0], marker=METHOD_STYLE[m]["marker"],
                   color="w", markerfacecolor=METHOD_STYLE[m]["color"],
                   markeredgecolor="0.2", markersize=10, label=DISPLAY_LABEL[m])
        for m in ["ragtag", "ragtag_debias_m3", "finetune"]
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = _build()

    print(f"{'model':<10}  {'method':<10}  {'time (h)':>8}  {'macro':>7}")
    for _, lbl in MODELS:
        for r in rows[lbl]:
            print(f"{lbl:<10}  {DISPLAY_LABEL[r['method']]:<13}  "
                  f"{r['gpu_time_s']/3600:>8.2f}  {r['macro']:>7.4f}")
        print()

    out_pdf = FIG_DIR / "method_pareto.pdf"
    out_png = FIG_DIR / "method_pareto.png"
    _plot(rows, out_pdf, out_png)
    print(f"wrote {out_pdf.relative_to(REPO_ROOT)}")
    print(f"wrote {out_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

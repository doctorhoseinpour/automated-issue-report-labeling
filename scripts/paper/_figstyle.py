"""Shared matplotlib style for the SANER 2027 paper figures.

Figures are drawn at their final printed size (IEEEtran: column 3.5 in, text
width 7.16 in) so that the type is 6.5-8 pt on the page and nothing is scaled
in LaTeX. Model colours are the Okabe-Ito colour-blind-safe set (validated with
the dataviz palette checker, all pairs), in a fixed order that every figure
shares; markers double-encode the model so the lines stay identifiable in
greyscale print.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

MODELS = ["Qwen-3B", "Qwen-7B", "Qwen-14B", "Qwen-32B"]
COLORS = {"Qwen-3B": "#0072B2", "Qwen-7B": "#E69F00", "Qwen-14B": "#009E73", "Qwen-32B": "#D55E00"}
MARKERS = {"Qwen-3B": "o", "Qwen-7B": "s", "Qwen-14B": "^", "Qwen-32B": "D"}
GREY = "#6f6f6f"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.linewidth": 0.4,
        "grid.color": "#e3e3e3",
        "lines.linewidth": 1.1,
        "lines.markersize": 3.4,
        "lines.markeredgewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })


def ring(ax, x, y, color, size=7.5):
    """Ring a selected point (the configuration the paper reports)."""
    ax.plot([x], [y], marker="o", ms=size, mfc="none", mec=color, mew=0.9, ls="none", zorder=5)

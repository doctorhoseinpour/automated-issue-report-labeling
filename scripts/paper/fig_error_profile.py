"""SANER2027/figures/error_profile.pdf -- per-class precision/recall profile of every
method at its best configuration (column width, two stacked panels).

  (a) bug class, (b) question class; x = precision, y = recall; pooled over the
  3,300 test issues, raw predictions. One point per (method, model): colour = model
  size, marker = method. VOTAG has no LLM and is drawn once per scope in grey. Thin
  arrows trace zero-shot -> RAGTAG -> BRAGTAG for each model.

On this balanced test set the predicted share of a class equals R / (3 P), so the
grey diagonal P = R is where a class is predicted exactly as often as it occurs:
points above it over-predict the class, points below it under-predict it. Light
curves are iso-F1 lines. Feature is omitted: it moves by at most 0.012 F1 between
methods (see the results table).

Reads paper/tables/triangulation_all_cells.csv, so it runs on any machine.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from _figstyle import COLORS, GREY, MODELS, apply_style  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "paper" / "tables" / "triangulation_all_cells.csv"
OUT_DIR = REPO / "SANER2027" / "figures"

# key, setting, marker, filled, legend label
METHODS = [
    ("zero_shot", "PA", "x", True, "Zero-shot"),
    ("ragtag", "PS", "o", True, "RAGTAG (PS)"),
    ("bragtag", "PS", "^", True, "BRAGTAG (PS)"),
    ("finetune", "PS", "s", False, "Fine-Tune (PS)"),
    ("finetune", "PA", "s", True, "Fine-Tune (PA)"),
]
PANELS = [  # class, title, xlim, ylim
    ("bug", "(a) Bug class", (0.50, 0.80), (0.60, 0.95)),
    ("question", "(b) Question class", (0.55, 0.92), (0.25, 0.77)),
]
ISO_F1 = {"bug": [0.6, 0.7, 0.8], "question": [0.5, 0.6, 0.7]}


def best_configs(cells: pd.DataFrame) -> pd.DataFrame:
    """Best k on raw macro F1 per (method, setting, model); k-free methods pass through."""
    return (cells.sort_values("f1_macro", ascending=False)
                 .groupby(["method", "setting", "model"], as_index=False).head(1))


def _pt(best, method, setting, model, cls):
    r = best[(best.method == method) & (best.setting == setting) & (best.model == model)].iloc[0]
    return float(r[f"p_{cls}"]), float(r[f"r_{cls}"])


def _iso_f1(ax, cls, xlim, ylim):
    for f in ISO_F1[cls]:
        p = np.linspace(max(xlim[0], f / 2 + 1e-3), xlim[1], 300)
        r = f * p / (2 * p - f)
        ok = (r >= ylim[0]) & (r <= ylim[1])
        if not ok.any():
            continue
        ax.plot(p[ok], r[ok], color="#d9d9d9", lw=0.6, zorder=0)
        i = np.where(ok)[0][-1]
        x, y = p[i], r[i]
        if y <= ylim[0] + 1e-6:      # leaves through the bottom edge
            off, ha, va = (2, 2), "left", "bottom"
        else:                        # reaches the right edge
            off, ha, va = (-2, 2), "right", "bottom"
        ax.annotate(f"$F_1{{=}}{f:.1f}$", (x, y), xytext=off, textcoords="offset points",
                    fontsize=5.6, color="#9a9a9a", ha=ha, va=va)


def _panel(ax, best, cls, title, xlim, ylim):
    _iso_f1(ax, cls, xlim, ylim)
    lo, hi = max(xlim[0], ylim[0]), min(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], color="#b5b5b5", lw=0.7, ls="-", zorder=0)
    ax.annotate("$P{=}R$", (hi, hi), xytext=(-3, -7), textcoords="offset points",
                fontsize=6, color="#8a8a8a", ha="right", va="top")
    # arrows zero-shot -> RAGTAG -> BRAGTAG
    for m in MODELS:
        path = [_pt(best, "zero_shot", "PA", m, cls), _pt(best, "ragtag", "PS", m, cls),
                _pt(best, "bragtag", "PS", m, cls)]
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=6,
                                        color=COLORS[m], lw=0.7, alpha=0.6,
                                        shrinkA=3.5, shrinkB=3.5))
    for method, setting, marker, filled, _ in METHODS:
        for m in MODELS:
            x, y = _pt(best, method, setting, m, cls)
            if marker == "x":
                ax.plot(x, y, marker="x", ms=5, mec=COLORS[m], mew=1.1, ls="none", zorder=3)
            else:
                ax.plot(x, y, marker=marker, ms=4.6 if marker != "^" else 5.2,
                        mfc=COLORS[m] if filled else "white",
                        mec="white" if filled else COLORS[m], mew=0.6 if filled else 1.0,
                        ls="none", zorder=4 if filled else 3)
    for setting in ("PS", "PA"):
        x, y = _pt(best, "votag", setting, "-", cls)
        ax.plot(x, y, marker="*", ms=6.5, mfc=GREY, mec="white", mew=0.5, ls="none", zorder=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(f"{cls.capitalize()} precision")
    ax.set_ylabel(f"{cls.capitalize()} recall")
    ax.set_title(title, loc="left")
    ax.grid(True, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, default=CELLS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    apply_style()
    best = best_configs(pd.read_csv(args.cells))
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.9), gridspec_kw={"hspace": 0.38})
    for ax, (cls, title, xlim, ylim) in zip(axes, PANELS):
        _panel(ax, best, cls, title, xlim, ylim)

    method_handles = [
        Line2D([], [], marker=mk, ls="none", ms=5 if mk == "x" else 4.6,
               mfc="#444444" if filled else "white", mec="#444444",
               mew=1.1 if mk == "x" else (0.6 if filled else 1.0), label=lbl)
        for _, _, mk, filled, lbl in METHODS
    ] + [Line2D([], [], marker="*", ls="none", ms=6.5, mfc=GREY, mec="white", mew=0.5,
                label="VOTAG (PS, PA)")]
    fig.legend(handles=method_handles, loc="lower center", ncol=3, frameon=False,
               handletextpad=0.4, columnspacing=1.2, bbox_to_anchor=(0.54, 0.035))
    model_handles = [Line2D([], [], marker="o", ls="none", ms=4.6, mfc=COLORS[m], mec="white",
                            mew=0.6, label=m) for m in MODELS]
    fig.legend(handles=model_handles, loc="lower center", ncol=4, frameon=False,
               handletextpad=0.4, columnspacing=1.2, bbox_to_anchor=(0.54, -0.005))
    fig.subplots_adjust(left=0.14, right=0.985, top=0.96, bottom=0.165)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"error_profile.{ext}", dpi=200)
    print(f"wrote {args.out_dir / 'error_profile.pdf'}")


if __name__ == "__main__":
    main()

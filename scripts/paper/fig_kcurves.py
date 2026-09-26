"""SANER2027/figures/kcurves.pdf -- macro F1 across the k grid; one figure for RQ1-RQ3.

  (a) VOTAG, PS (solid) and PA (dashed), k = 1..20, 25, 30; the two peaks are ringed.
  (b) RAGTAG, four Qwen sizes, PS (solid, filled markers) and PA (dashed, hollow),
      k = 0..15; k = 0 is zero-shot, which is identical in both scopes. The dotted line
      is VOTAG's best (PA, k = 16). Each model's best PS k is ringed.
  (c) BRAGTAG (PS, markers) against RAGTAG-PS (thin lines, the same curves as in b).
      Each model's best BRAGTAG k is ringed.

Panels (b) and (c) share one y-range; (a) has its own because VOTAG lives in a
narrower band. Reads paper/tables/triangulation_all_cells.csv (written by
tab_triangulation.py on the lab machine), so it runs on any machine.
Convention: pooled over the 3,300 test issues, raw predictions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from _figstyle import COLORS, MARKERS, MODELS, apply_style, ring  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "paper" / "tables" / "triangulation_all_cells.csv"
OUT_DIR = REPO / "SANER2027" / "figures"

KS_RAG = [0, 1, 3, 6, 9, 12, 15]
VOTAG_FLOOR = ("PA", 16)  # VOTAG's best configuration


def _series(cells: pd.DataFrame, method: str, setting: str, model: str) -> pd.DataFrame:
    s = cells[(cells.method == method) & (cells.setting == setting) & (cells.model == model)]
    s = s.assign(k=s.k.astype(int)).sort_values("k")
    return s[["k", "f1_macro"]]


def _with_zero_shot(cells: pd.DataFrame, s: pd.DataFrame, model: str) -> pd.DataFrame:
    z = cells[(cells.method == "zero_shot") & (cells.model == model)].iloc[0]
    return pd.concat([pd.DataFrame({"k": [0], "f1_macro": [z.f1_macro]}), s], ignore_index=True)


def _panel_votag(ax, cells):
    for setting, ls, label in (("PS", "-", "PS"), ("PA", "--", "PA")):
        s = _series(cells, "votag", setting, "-")
        ax.plot(s.k, s.f1_macro, ls=ls, color="#333333", marker="o", ms=2.4,
                mfc="white" if setting == "PA" else "#333333", mew=0.7, label=label)
        peak = s.loc[s.f1_macro.idxmax()]
        ring(ax, peak.k, peak.f1_macro, "#333333")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("$k$ (retrieved neighbors)")
    ax.set_ylabel("Macro $F_1$ (%)")
    ax.set_title("(a) VOTAG", loc="left")


def _panel_ragtag(ax, cells):
    for m in MODELS:
        ps = _with_zero_shot(cells, _series(cells, "ragtag", "PS", m), m)
        pa = _with_zero_shot(cells, _series(cells, "ragtag", "PA", m), m)
        ax.plot(pa.k, pa.f1_macro, ls="--", lw=0.8, color=COLORS[m], marker=MARKERS[m],
                mfc="white", mew=0.7, ms=3.0, alpha=0.85, zorder=2)
        ax.plot(ps.k, ps.f1_macro, ls="-", color=COLORS[m], marker=MARKERS[m], zorder=3)
        best = ps[ps.k > 0].loc[lambda d: d.f1_macro.idxmax()]
        ring(ax, best.k, best.f1_macro, COLORS[m])
    floor = cells[(cells.method == "votag") & (cells.setting == VOTAG_FLOOR[0])
                  & (cells.k.astype(str) == str(VOTAG_FLOOR[1]))].iloc[0].f1_macro
    ax.axhline(floor, ls=":", lw=0.8, color="#555555", zorder=1)
    ax.text(15, floor + 0.4, f"VOTAG best ({floor:.1f})", ha="right", va="bottom",
            fontsize=6.3, color="#555555")
    ax.set_ylabel("Macro $F_1$ (%)")
    ax.set_xticks(KS_RAG)
    ax.set_xlabel("$k$ (few-shot neighbors; $k{=}0$ is zero-shot)")
    ax.set_title("(b) RAGTAG", loc="left")


def _panel_bragtag(ax, cells):
    for m in MODELS:
        rag = _series(cells, "ragtag", "PS", m)
        brag = _series(cells, "bragtag", "PS", m)
        ax.plot(rag.k, rag.f1_macro, ls="-", lw=0.7, color=COLORS[m], alpha=0.45, zorder=2)
        ax.plot(brag.k, brag.f1_macro, ls="-", color=COLORS[m], marker=MARKERS[m], zorder=3)
        best = brag.loc[brag.f1_macro.idxmax()]
        ring(ax, best.k, best.f1_macro, COLORS[m])
    ax.set_xticks([1, 3, 6, 9, 12, 15])
    ax.set_xlabel("$k$ (few-shot neighbors)")
    ax.set_title("(c) BRAGTAG vs. RAGTAG (PS)", loc="left")
    handles = [Line2D([], [], color="#333333", marker="o", label="BRAGTAG"),
               Line2D([], [], color="#333333", lw=0.7, alpha=0.45, label="RAGTAG (as in b)")]
    ax.legend(handles=handles, loc="lower right", frameon=False, handlelength=1.8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, default=CELLS)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    apply_style()
    cells = pd.read_csv(args.cells)
    cells["f1_macro"] = 100 * cells["f1_macro"]  # percent, as in the paper
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), gridspec_kw={"wspace": 0.24})
    _panel_votag(axes[0], cells)
    _panel_ragtag(axes[1], cells)
    _panel_bragtag(axes[2], cells)
    for ax in axes:
        ax.grid(True, axis="y", zorder=0)
        ax.set_axisbelow(True)
    lo, hi = 60.0, 79.5
    for ax in axes[1:]:
        ax.set_ylim(lo, hi)
    axes[2].set_yticklabels([])
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=m) for m in MODELS]
    handles += [Line2D([], [], color="#333333", ls="-", marker="o", ms=2.4, label="PS (solid, filled)"),
                Line2D([], [], color="#333333", ls="--", lw=0.8, marker="o", ms=2.4, mfc="white",
                       label="PA (dashed, hollow)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, handlelength=2.0,
               columnspacing=1.4, handletextpad=0.5, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.29)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"kcurves.{ext}", dpi=200)
    print(f"wrote {args.out_dir / 'kcurves.pdf'}")


if __name__ == "__main__":
    main()

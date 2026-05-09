"""Generate paper/figures/cost_analysis.{pdf,png} for §5.5 cost analysis.

Two-panel figure:
  (a) Peak GPU RAM by model size: RAGTAG/BRAGTAG (same) vs Fine-Tune.
  (b) Total GPU time by model size: RAGTAG inference, BRAGTAG inference,
      Fine-Tune as a stacked bar (training + inference). Shows the
      RAG-vs-FT crossover at Qwen-32B.

Numbers come from the same helpers as tab_method_comparison.py so the
figure stays in sync with the table.
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

RAG_COLOR  = "#4C72B0"  # blue
BRAG_COLOR = "#55A868"  # green
FT_TRAIN   = "#D17B0F"  # darker orange (training)
FT_INFER   = "#FBB04E"  # lighter orange (inference)
SHARED_COLOR = "#4C72B0"  # blue for the combined RAG/BRAG bar in panel (a)


def _build():
    rows = {}
    for tag, lbl in MODELS:
        rows[lbl] = {
            "ragtag":           _row_few_shot(tag, lbl, "ragtag"),
            "ragtag_debias_m3": _row_few_shot(tag, lbl, "ragtag_debias_m3"),
            "finetune":         _row_finetune(tag, lbl),
        }
    return rows


def _plot(rows, out_pdf, out_png):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.0, 4.2))

    n = len(MODELS)
    x = np.arange(n)
    labels = [lbl for _, lbl in MODELS]

    # ---- Panel (a): peak GPU RAM ----
    width = 0.36
    ram_rag = [rows[lbl]["ragtag"]["gpu_ram_mb"] / 1024 for lbl in labels]
    ram_ft  = [rows[lbl]["finetune"]["gpu_ram_mb"] / 1024 for lbl in labels]

    bars_rag = ax_l.bar(x - width/2, ram_rag, width, color=SHARED_COLOR,
                        edgecolor="0.3", linewidth=0.5,
                        label="RAGTAG/BRAGTAG (inference)")
    bars_ft = ax_l.bar(x + width/2, ram_ft, width, color=FT_TRAIN,
                       edgecolor="0.3", linewidth=0.5,
                       label="Fine-Tune (training+inference)")
    ax_l.bar_label(bars_rag, fmt="%.1f", padding=2, fontsize=8)
    ax_l.bar_label(bars_ft,  fmt="%.1f", padding=2, fontsize=8)

    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, fontsize=9)
    ax_l.set_ylabel("Peak GPU RAM (GB)", fontsize=9)
    ax_l.set_title("(a) Peak GPU RAM by model size", fontsize=10, fontweight="bold")
    ax_l.set_ylim(0, max(ram_ft) * 1.15)
    ax_l.grid(True, axis="y", alpha=0.3)
    ax_l.legend(loc="upper left", fontsize=8.5, frameon=False)

    # ---- Panel (b): total time, FT stacked (train + inference) ----
    width2 = 0.26
    rag_t   = [rows[lbl]["ragtag"]["total_time_s"] / 3600 for lbl in labels]
    brag_t  = [rows[lbl]["ragtag_debias_m3"]["total_time_s"] / 3600 for lbl in labels]
    ft_train = [rows[lbl]["finetune"]["train_time_s"] / 3600 for lbl in labels]
    ft_infer = [rows[lbl]["finetune"]["infer_time_s"] / 3600 for lbl in labels]
    ft_total = [t + i for t, i in zip(ft_train, ft_infer)]

    b_rag  = ax_r.bar(x - width2,        rag_t,    width2, color=RAG_COLOR,
                      edgecolor="0.3", linewidth=0.5, label="RAGTAG (inference)")
    b_brag = ax_r.bar(x,                 brag_t,   width2, color=BRAG_COLOR,
                      edgecolor="0.3", linewidth=0.5, label="BRAGTAG (inference)")
    b_ft_tr = ax_r.bar(x + width2,       ft_train, width2, color=FT_TRAIN,
                       edgecolor="0.3", linewidth=0.5, label="Fine-Tune training")
    b_ft_in = ax_r.bar(x + width2,       ft_infer, width2, bottom=ft_train,
                       color=FT_INFER, edgecolor="0.3", linewidth=0.5,
                       label="Fine-Tune inference")

    # Annotate totals on top of each bar
    for xi, v in zip(x - width2, rag_t):
        ax_r.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="0.25")
    for xi, v in zip(x, brag_t):
        ax_r.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="0.25")
    for xi, v in zip(x + width2, ft_total):
        ax_r.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="0.25")

    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, fontsize=9)
    ax_r.set_ylabel("Total GPU time (h)", fontsize=9)
    ax_r.set_title("(b) Total GPU time by model size", fontsize=10, fontweight="bold")
    ax_r.set_ylim(0, max(max(rag_t), max(brag_t), max(ft_total)) * 1.12)
    ax_r.grid(True, axis="y", alpha=0.3)
    ax_r.legend(loc="upper left", fontsize=8.5, frameon=False)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = _build()

    print("Panel (a) RAM (GB):")
    print(f"  {'Model':<10} {'RAG/BRAG':>10}  {'Fine-Tune':>10}  {'RAG % of FT':>12}")
    for _, lbl in MODELS:
        rag_gb = rows[lbl]["ragtag"]["gpu_ram_mb"] / 1024
        ft_gb  = rows[lbl]["finetune"]["gpu_ram_mb"] / 1024
        print(f"  {lbl:<10} {rag_gb:>10.1f}  {ft_gb:>10.1f}  {100*rag_gb/ft_gb:>10.0f}%")

    print()
    print("Panel (b) total time (h):")
    print(f"  {'Model':<10} {'RAG':>6} {'BRAG':>6} {'FT(tr)':>7} {'FT(in)':>7} {'FT(tot)':>8}")
    for _, lbl in MODELS:
        rag = rows[lbl]["ragtag"]["total_time_s"] / 3600
        brag = rows[lbl]["ragtag_debias_m3"]["total_time_s"] / 3600
        ft_tr = rows[lbl]["finetune"]["train_time_s"] / 3600
        ft_in = rows[lbl]["finetune"]["infer_time_s"] / 3600
        print(f"  {lbl:<10} {rag:>6.2f} {brag:>6.2f} {ft_tr:>7.2f} {ft_in:>7.2f} {ft_tr+ft_in:>8.2f}")

    out_pdf = FIG_DIR / "cost_analysis.pdf"
    out_png = FIG_DIR / "cost_analysis.png"
    _plot(rows, out_pdf, out_png)
    print(f"\nwrote {out_pdf.relative_to(REPO_ROOT)}")
    print(f"wrote {out_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

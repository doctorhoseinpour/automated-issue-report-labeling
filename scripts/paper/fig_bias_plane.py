"""paper/figures/bias_plane.{pdf,png} -- layout A of the metric triangulation.

One column-width panel: x = bug precision, y = question recall. Every
(method, model) best configuration is a point (colour = model size, marker =
method); arrows trace zero-shot -> RAGTAG -> BRAGTAG per model. VOTAG (PS best
k) and the two fine-tuning scopes are shown as separate markers. The bug bias
of the paper (bug over-predicted, question under-recalled) is the bottom-left
corner; every intervention moves points up and to the right.

Convention: pooled, raw predictions, best k on raw macro F1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _triangulation import MODEL_LABELS, REPO_ROOT, all_cells, best_configs  # noqa: E402

FIG_DIR = REPO_ROOT / "paper" / "figures"
COLORS = {"Qwen-3B": "#0072B2", "Qwen-7B": "#E69F00", "Qwen-14B": "#8c564b", "Qwen-32B": "#D55E00"}
MARKERS = {  # (method, setting) -> (marker, filled, label)
    ("zero_shot", "PA"): ("x", True, "Zero-shot"),
    ("ragtag", "PS"): ("o", True, r"RAGTAG (PS, best $k$)"),
    ("bragtag", "PS"): ("^", True, r"BRAGTAG (PS, best $k$)"),
    ("finetune", "PS"): ("s", False, "Fine-Tune (PS)"),
    ("finetune", "PA"): ("s", True, "Fine-Tune (PA)"),
}


def _pt(best, method, setting, model):
    r = best[(best.method == method) & (best.setting == setting) & (best.model == model)].iloc[0]
    return float(r.p_bug), float(r.r_question)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    best = best_configs(all_cells())

    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    # arrows zero-shot -> RAGTAG -> BRAGTAG
    for m in MODEL_LABELS:
        path = [_pt(best, "zero_shot", "PA", m), _pt(best, "ragtag", "PS", m), _pt(best, "bragtag", "PS", m)]
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=COLORS[m], lw=1.1, alpha=0.75,
                                        shrinkA=4, shrinkB=4))
    # points
    for (method, setting), (marker, filled, _) in MARKERS.items():
        for m in MODEL_LABELS:
            x, y = _pt(best, method, setting, m)
            ax.scatter([x], [y], marker=marker, s=60 if marker != "x" else 70,
                       facecolors=COLORS[m] if filled else "white",
                       edgecolors=COLORS[m], linewidths=1.6, zorder=5)
    # VOTAG-PS
    v = best[(best.method == "votag") & (best.setting == "PS")].iloc[0]
    ax.scatter([v.p_bug], [v.r_question], marker="*", s=130, facecolors="0.45",
               edgecolors="black", linewidths=0.6, zorder=6)
    ax.annotate("VOTAG (PS)", (v.p_bug, v.r_question), xytext=(6, -12),
                textcoords="offset points", fontsize=8, color="0.3")

    ax.set_xlabel("Bug precision  (higher = fewer issues wrongly labeled bug)")
    ax.set_ylabel("Question recall  (higher = more questions recovered)")
    ax.grid(True, alpha=0.3)

    model_handles = [Line2D([0], [0], color=COLORS[m], lw=3, label=m) for m in MODEL_LABELS]
    method_handles = [
        Line2D([0], [0], marker=mk, linestyle="", markersize=7, color="0.25",
               markerfacecolor=("0.25" if filled else "white"), label=lbl)
        for (mk, filled, lbl) in MARKERS.values()
    ]
    leg1 = ax.legend(handles=model_handles, loc="lower right", fontsize=8, frameon=False, title=None)
    ax.add_artist(leg1)
    ax.legend(handles=method_handles, loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "bias_plane.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "bias_plane.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote paper/figures/bias_plane.{pdf,png}")
    for (method, setting) in MARKERS:
        for m in MODEL_LABELS:
            x, y = _pt(best, method, setting, m)
            print(f"{method:<9} {setting} {m:<8} bugP={x:.3f} qR={y:.3f}")


if __name__ == "__main__":
    main()

"""Generate paper/figures/ragtag_kcurve.{pdf,png} with POOLED PS aggregation.

Single-panel figure: RAGTAG macro F1 vs k for all four Qwen sizes, with PS
(solid line, filled markers) and PA (dashed thin line, hollow markers) overlaid
in the same color per model. The PS/PA pairs sitting nearly on top of each
other visually communicates "PS is marginally better than PA at every (model,
k)." A horizontal dashed line at VTAG's best (0.604, PA k=16) shows that
every LLM-based result clears the retrieval-only floor.

This figure reports RAW \\ragtag\\ macro F1 (no \\votag-rescue applied). The
rescue is reserved for the Fine-Tune comparison subsection where vanilla and
debiased \\ragtag\\ are shown side-by-side with their rescued counterparts.

k=0 (zero-shot) is the no-retrieval anchor. The PA zero_shot prediction
file is used for both panels (zero-shot is retrieval-independent).

Convention: pooled macro F1, see paper/sections/04_setup.tex.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "issues11k"
FIG_DIR = REPO_ROOT / "paper" / "figures"

LABELS = ["bug", "feature", "question"]

MODELS = [
    # (model_tag, label, color, marker)
    # Colors picked from a colorblind-safe palette; markers are also distinct
    # so the lines remain identifiable even if two colors look similar.
    ("unsloth_Qwen2_5_3B_Instruct_bnb_4bit",  "Qwen-3B",  "#0072B2", "o"),  # blue
    ("unsloth_Qwen2_5_7B_Instruct_bnb_4bit",  "Qwen-7B",  "#E69F00", "s"),  # orange
    ("unsloth_Qwen2_5_14B_Instruct_bnb_4bit", "Qwen-14B", "#8c564b", "^"),  # brown
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "Qwen-32B", "#D55E00", "D"),  # vermilion
]

K_VALUES = [0, 1, 3, 6, 9, 12, 15]

# Best VOTAG macro F1 (pooled) — the retrieval-only floor every LLM-based
# method must clear. We use the PA-best as "the" VOTAG floor in this figure
# since it is the higher of the two (0.604 > 0.595 in PS); any line above it
# beats VOTAG in either setting.
VOTAG_BEST_F1 = 0.604
VOTAG_BEST_K = 16
VOTAG_BEST_SETTING = "PA"


def _macro_f1(df: pd.DataFrame) -> float:
    return f1_score(
        df["ground_truth"], df["predicted_label"],
        average="macro", labels=LABELS, zero_division=0,
    )


def _filename_for_k(k: int) -> str:
    return "preds_zero_shot.csv" if k == 0 else f"preds_k{k}.csv"


def _pa_macro(model: str, k: int) -> float | None:
    """Raw PA macro F1 for one (model, k); no rescue applied."""
    p = RESULTS / "agnostic" / model / "ragtag" / "predictions" / _filename_for_k(k)
    if not p.exists():
        return None
    df = pd.read_csv(p, usecols=["ground_truth", "predicted_label"])
    return _macro_f1(df)


def _ps_macro(model: str, k: int, projects: list[str]) -> float | None:
    """Raw pooled PS macro F1 for one (model, k); no rescue applied.

    For k=0 (zero-shot), fall back to the PA zero_shot file since zero-shot
    is retrieval-independent and PS-zero_shot is not consistently materialised
    on disk for every model.
    """
    if k == 0:
        return _pa_macro(model, 0)
    parts = []
    for proj in projects:
        p = (RESULTS / "project_specific" / proj / model / "ragtag"
             / "predictions" / f"preds_k{k}.csv")
        if not p.exists():
            return None
        parts.append(pd.read_csv(p, usecols=["ground_truth", "predicted_label"]))
    if not parts:
        return None
    return _macro_f1(pd.concat(parts, ignore_index=True))


def _build_curves() -> dict:
    projects = sorted(p.name for p in (RESULTS / "project_specific").iterdir() if p.is_dir())
    out = {"PA": {}, "PS": {}}
    for tag, _, _, _ in MODELS:
        out["PA"][tag] = {k: _pa_macro(tag, k) for k in K_VALUES}
        out["PS"][tag] = {k: _ps_macro(tag, k, projects) for k in K_VALUES}
    return out


def _plot(curves: dict, out_pdf: Path, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # Per-model: PS (solid, filled marker) overlaid with PA (dashed thin,
    # hollow marker) in the same color. The pair sitting nearly on top of
    # each other visually communicates the marginal PS-vs-PA gap.
    for tag, label, color, marker in MODELS:
        for setting, ls, lw, alpha, mfc in [
            ("PS", "-",  1.8, 1.00, color),    # solid, filled marker
            ("PA", "--", 1.0, 0.55, "white"),  # dashed, hollow marker
        ]:
            ks, ys = [], []
            for k in K_VALUES:
                v = curves[setting][tag][k]
                if v is not None:
                    ks.append(k)
                    ys.append(v)
            if not ks:
                continue
            ax.plot(ks, ys, color=color, linestyle=ls, linewidth=lw,
                    marker=marker, markersize=6.0,
                    markerfacecolor=mfc, markeredgecolor=color,
                    alpha=alpha,
                    # Only the PS line carries the model legend entry.
                    label=(label if setting == "PS" else None))

    # VTAG retrieval-only floor.
    ax.axhline(VOTAG_BEST_F1, color="0.35", linestyle=":", linewidth=1.4,
               label=f"VTAG best ({VOTAG_BEST_SETTING}, $k{{=}}{VOTAG_BEST_K}$): {VOTAG_BEST_F1:.3f}")

    ax.axvline(0, color="black", linestyle=":", alpha=0.35, linewidth=0.9)
    ax.set_xlabel(r"$k$ (number of retrieved few-shot neighbors; $k{=}0$ is zero-shot)")
    ax.set_ylabel("Macro $F_1$ (pooled)")
    ax.set_xticks(K_VALUES)
    ax.grid(True, alpha=0.3)

    # Build a legend that adds two style-key handles (PS solid / PA dashed)
    # in addition to the model-color entries.
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=1.8,
               marker="o", markersize=6, markerfacecolor="black",
               label="PS (solid, filled)"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0,
               marker="o", markersize=6, markerfacecolor="white",
               markeredgecolor="black", alpha=0.7,
               label="PA (dashed, hollow)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    handles = handles + style_handles
    labels = labels + [h.get_label() for h in style_handles]
    ax.legend(handles, labels, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), ncol=4,
              fontsize=9, frameon=False, borderaxespad=0)

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    curves = _build_curves()

    # Print the data table.
    print(f"{'model':<10} {'set':<3} " + "  ".join(f"k={k:<3}" for k in K_VALUES))
    print("-" * 70)
    for tag, label, *_ in MODELS:
        for setting in ("PS", "PA"):
            row = curves[setting][tag]
            cells = []
            for k in K_VALUES:
                v = row[k]
                cells.append(f"{v:.4f}" if v is not None else " --- ")
            print(f"{label:<10} {setting:<3} " + "  ".join(f"{c:<5}" for c in cells))

    out_pdf = FIG_DIR / "ragtag_kcurve.pdf"
    out_png = FIG_DIR / "ragtag_kcurve.png"
    _plot(curves, out_pdf, out_png)
    print(f"\nwrote {out_pdf.relative_to(REPO_ROOT)}")
    print(f"wrote {out_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

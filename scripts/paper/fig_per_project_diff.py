"""SANER2027/figures/per_project_diff.pdf -- per-(project, model) macro-F1 difference
between BRAGTAG (PS, best k) and PA LoRA fine-tuning, 11 projects x 4 Qwen sizes,
drawn at column width (projects as rows so the names read horizontally).
Blue = BRAGTAG ahead, red = fine-tuning ahead; every cell prints its value.

Data: paper/tables/per_project_diff.csv (default; runs on any machine), or
--from-results to recompute from results/ on the lab machine, which also rewrites
that CSV with full-precision values. Per-project scores are computed on each
project's 300 test issues (raw predictions); best k is the pooled raw macro-F1 optimum,
as in the results table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _figstyle import MODELS, apply_style  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "paper" / "tables" / "per_project_diff.csv"
OUT_DIR = REPO / "SANER2027" / "figures"
PROJECT_SHORT = {"dart-lang_sdk": "dart-lang"}  # every other tag is "<owner>_<name>" with owner == name or name used in the paper
CSV_HEADER = """# Per-(project, model) macro-F1 difference: BRAGTAG (PS, best k) minus PA LoRA fine-tuning,
# evaluated on each project's 300 test issues (raw predictions).
# Written by scripts/paper/fig_per_project_diff.py --from-results on the lab machine.
"""


def _short(tag: str) -> str:
    return PROJECT_SHORT.get(tag, tag.split("_", 1)[1] if "_" in tag else tag)


def from_results() -> pd.DataFrame:
    """Recompute the 4 x 11 matrix from results/ (lab machine only)."""
    from sklearn.metrics import f1_score

    from _rescue import RESULTS, _project_list
    from significance_method_comparison import MODELS as MODEL_TAGS, _best_k_raw, _ft_pa, _ps_preds

    def macro(df):
        return f1_score(df["ground_truth"], df["predicted_label"], labels=["bug", "feature", "question"],
                        average="macro", zero_division=0)

    proj_tags = pd.read_csv(RESULTS / "agnostic" / "neighbors" / "test_split.csv", usecols=["repo"])
    proj_tags = proj_tags["repo"].str.replace("/", "_", n=1).tolist()
    projects = _project_list()
    rows = []
    for tag, label in MODEL_TAGS:
        k = _best_k_raw(tag, "ragtag_debias_m3")
        brag = _ps_preds(tag, "ragtag_debias_m3", k, rescue=False)
        ft = _ft_pa(tag, rescue=False).assign(proj=proj_tags)
        row = {"model": label, "k": k}
        for proj in projects:
            row[_short(proj)] = macro(brag[proj]) - macro(ft[ft.proj == proj])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", type=Path, default=CSV)
    ap.add_argument("--from-results", action="store_true", help="recompute from results/ and rewrite --csv")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.from_results:
        df = from_results()
        with open(args.csv, "w") as fh:
            fh.write(CSV_HEADER)
            df.to_csv(fh, index=False, float_format="%.5f")
        print(f"rewrote {args.csv}")
    else:
        df = pd.read_csv(args.csv, comment="#")
    df = df.set_index("model").loc[MODELS]
    ks = df["k"].astype(int).tolist()
    mat = df.drop(columns="k").T  # projects x models
    projects = list(mat.index)
    values = mat.values.astype(float)

    apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    vmax = 0.10
    im = ax.imshow(values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([f"{m}\n$k{{=}}{k}$" for m, k in zip(MODELS, ks)])
    ax.set_yticks(range(len(projects)))
    ax.set_yticklabels(projects, family="monospace", fontsize=6.8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # white cell gaps
    ax.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(projects), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(projects)):
        for j in range(len(MODELS)):
            v = values[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=6.4,
                    color="white" if abs(v) > 0.065 else "#222222")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=6.3, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label("BRAGTAG $-$ Fine-Tune (macro $F_1$)", fontsize=6.8)
    fig.subplots_adjust(left=0.20, right=0.88, top=0.99, bottom=0.10)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"per_project_diff.{ext}", dpi=200)
    print(f"wrote {args.out_dir / 'per_project_diff.pdf'}")


if __name__ == "__main__":
    main()

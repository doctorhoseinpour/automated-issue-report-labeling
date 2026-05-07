"""Generate paper/tables/vtag_peak.tex with POOLED PS aggregation.

Emits a booktabs table of \\votag's peak macro F1 + per-class F1 for both
PA and PS settings at each setting's best k under pooled aggregation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "issues11k"
TABLES_DIR = REPO_ROOT / "paper" / "tables"

LABELS = ["bug", "feature", "question"]
PA_BEST_K = 16
PS_BEST_K = 15


def _pa_preds(k: int) -> pd.DataFrame:
    return pd.read_csv(
        RESULTS / "agnostic" / "vtag" / "predictions" / f"preds_k{k}.csv",
        usecols=["ground_truth", "predicted_label"],
    )


def _ps_preds_pooled(projects: list[str], k: int) -> pd.DataFrame:
    parts = []
    for proj in projects:
        parts.append(pd.read_csv(
            RESULTS / "project_specific" / proj / "vtag" / "predictions"
            / f"preds_k{k}.csv",
            usecols=["ground_truth", "predicted_label"],
        ))
    return pd.concat(parts, ignore_index=True)


def _metrics(df: pd.DataFrame) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        df["ground_truth"], df["predicted_label"],
        labels=LABELS, average="macro", zero_division=0,
    )
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(
        df["ground_truth"], df["predicted_label"],
        labels=LABELS, zero_division=0,
    )
    acc = accuracy_score(df["ground_truth"], df["predicted_label"])
    return {
        "macro_f1": f1,
        "accuracy": acc,
        **{f"f1_{lab}": f1_per[i] for i, lab in enumerate(LABELS)},
    }


def _emit_tabular(rows: list[dict]) -> str:
    """Just the tabular block (no caption, no float wrapper).

    The composing .tex file provides the figure/table float and the caption,
    so the same data can be placed inline or side-by-side as needed.
    """
    def f(x): return f"{x:.3f}"
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"  \toprule",
        r"  Setting & Best $k$ & Macro $F_1$ & Accuracy & $F_1^{\text{bug}}$ & $F_1^{\text{feature}}$ & $F_1^{\text{question}}$ \\",
        r"  \midrule",
    ]
    for r in rows:
        lines.append(
            f"  {r['setting']} & {r['k']} & {f(r['macro_f1'])} & "
            f"{f(r['accuracy'])} & {f(r['f1_bug'])} & {f(r['f1_feature'])} & "
            f"{f(r['f1_question'])} \\\\"
        )
    lines += [
        r"  \bottomrule",
        r"\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    projects = sorted(p.name for p in (RESULTS / "project_specific").iterdir() if p.is_dir())

    rows = []
    for setting, k, df in [
        ("PS", PS_BEST_K, _ps_preds_pooled(projects, PS_BEST_K)),
        ("PA", PA_BEST_K, _pa_preds(PA_BEST_K)),
    ]:
        m = _metrics(df)
        rows.append({"setting": setting, "k": k, **m})
        print(f"{setting} (k={k}): "
              f"macro_F1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}  "
              f"bug={m['f1_bug']:.4f}  feature={m['f1_feature']:.4f}  "
              f"question={m['f1_question']:.4f}")

    out = TABLES_DIR / "vtag_peak.tex"
    out.write_text(_emit_tabular(rows))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

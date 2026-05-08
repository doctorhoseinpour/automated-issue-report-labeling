"""Generate paper/tables/bragtag_results.tex with POOLED PS aggregation.

Comprehensive reference table for \\bragtag analysis. For each Qwen size,
reports both \\ragtag and \\bragtag at their respective best k:
  - best k
  - macro F1 (raw)
  - per-class F1 (bug / feature / question)
  - invalid prediction rate

Convention: pooled PS, raw (no \\votag-rescue applied). See
paper/sections/04_setup.tex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rescue import load_raw_preds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = REPO_ROOT / "paper" / "tables"

LABELS = ["bug", "feature", "question"]
KS = [1, 3, 6, 9, 12, 15]

MODELS = [
    ("unsloth_Qwen2_5_3B_Instruct_bnb_4bit",  "Qwen-3B"),
    ("unsloth_Qwen2_5_7B_Instruct_bnb_4bit",  "Qwen-7B"),
    ("unsloth_Qwen2_5_14B_Instruct_bnb_4bit", "Qwen-14B"),
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "Qwen-32B"),
]


def _macro(df) -> float:
    return f1_score(df["ground_truth"], df["predicted_label"],
                    labels=LABELS, average="macro", zero_division=0)


def _per_class(df) -> dict[str, float]:
    _, _, f1, _ = precision_recall_fscore_support(
        df["ground_truth"], df["predicted_label"],
        labels=LABELS, zero_division=0)
    return dict(zip(LABELS, f1))


def _row(model: str, approach: str, label: str) -> dict:
    """Build one row of the table for (model, approach) at its best k."""
    best_k = max(KS, key=lambda k: _macro(load_raw_preds(model, "PS", k, approach)))
    raw = load_raw_preds(model, "PS", best_k, approach)
    pc = _per_class(raw)
    n_inv = (raw["predicted_label"] == "invalid").sum()
    return {
        "model": label,
        "method": approach,
        "best_k": best_k,
        "macro_raw": _macro(raw),
        "f1_bug": pc["bug"],
        "f1_feature": pc["feature"],
        "f1_question": pc["question"],
        "invalid_pct": 100.0 * n_inv / len(raw),
    }


def _emit_tex(rows: list[dict]) -> str:
    def f3(x): return f"{x:.3f}"
    def fp(x): return f"{x:.1f}\\%"

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{\ragtag\ vs \bragtag\ at each model's best $k$ (PS, pooled).}",
        r"  \label{tab:bragtag-results}",
        r"  \small",
        r"  \begin{tabular}{llccccccc}",
        r"    \toprule",
        r"    Model & Method & $k^*$ & Macro $F_1$ & $F_1^{\text{bug}}$ & $F_1^{\text{feat}}$ & $F_1^{\text{q}}$ & Invalid rate \\",
        r"    \midrule",
    ]
    prev_model = None
    for r in rows:
        model_cell = r["model"] if r["model"] != prev_model else ""
        if r["model"] != prev_model and prev_model is not None:
            lines.append(r"    \addlinespace[2pt]")
        method_name = "\\ragtag" if r["method"] == "ragtag" else "\\bragtag"
        lines.append(
            f"    {model_cell} & {method_name} & {r['best_k']} & "
            f"{f3(r['macro_raw'])} & "
            f"{f3(r['f1_bug'])} & {f3(r['f1_feature'])} & {f3(r['f1_question'])} & "
            f"{fp(r['invalid_pct'])} \\\\"
        )
        prev_model = r["model"]
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, lbl in MODELS:
        rows.append(_row(tag, "ragtag", lbl))
        rows.append(_row(tag, "ragtag_debias_m3", lbl))

    print(f"{'Model':<10} {'Method':<10} {'k*':<3}  {'macro raw':>9}  "
          f"{'F1 bug':>7}  {'F1 feat':>8}  {'F1 q':>6}  {'Inv %':>6}")
    for r in rows:
        method = "RAGTAG" if r["method"] == "ragtag" else "BRAGTAG"
        print(f"{r['model']:<10} {method:<10} {r['best_k']:<3}  "
              f"{r['macro_raw']:>9.4f}  "
              f"{r['f1_bug']:>7.4f}  {r['f1_feature']:>8.4f}  {r['f1_question']:>6.4f}  "
              f"{r['invalid_pct']:>5.1f}%")

    out = TABLES_DIR / "bragtag_results.tex"
    out.write_text(_emit_tex(rows))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

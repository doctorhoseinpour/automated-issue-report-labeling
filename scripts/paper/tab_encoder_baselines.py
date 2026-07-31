"""Generate paper/tables/encoder_baselines.tex — encoder baselines vs. the
paper's LLM-based methods (SANER revision, ESEM Review #126C).

Rows: 3 encoder baselines x {PA, PS} plus, for context, the paper's existing
best configurations (RAGTAG-PS, BRAGTAG-PS at best k; Fine-Tune-PA) per Qwen
size, all on raw predictions with pooled aggregation.

Metrics per row: macro F1, accuracy, macro precision/recall, per-class F1,
peak GPU RAM, train/infer/total wall time. Encoders emit no invalid outputs
by construction (argmax over a fixed 3-class head), so no +VOTAG column.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rescue import RESULTS, load_raw_preds  # noqa: E402
from _encoders import ENCODERS, load_encoder_cost, load_encoder_preds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = REPO_ROOT / "paper" / "tables"

LABELS = ["bug", "feature", "question"]
KS_RAG = [1, 3, 6, 9, 12, 15]

QWEN_MODELS = [
    ("unsloth_Qwen2_5_3B_Instruct_bnb_4bit",  "Qwen-3B"),
    ("unsloth_Qwen2_5_7B_Instruct_bnb_4bit",  "Qwen-7B"),
    ("unsloth_Qwen2_5_14B_Instruct_bnb_4bit", "Qwen-14B"),
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "Qwen-32B"),
]


def _metrics(df: pd.DataFrame) -> dict:
    y, p = df["ground_truth"], df["predicted_label"]
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    return {
        "macro": f1_score(y, p, labels=LABELS, average="macro", zero_division=0),
        "accuracy": accuracy_score(y, p),
        "precision_macro": pr.mean(),
        "recall_macro": rc.mean(),
        "f1_bug": f1[0], "f1_feature": f1[1], "f1_question": f1[2],
    }


def encoder_rows() -> list[dict]:
    rows = []
    for tag, approach, fname, lbl in ENCODERS:
        for setting in ("PA", "PS"):
            df = load_encoder_preds(tag, approach, fname, setting)
            if len(df) != 3300:
                raise RuntimeError(f"{lbl} {setting}: pooled {len(df)} rows, expected 3300")
            cost = load_encoder_cost(tag, approach, setting)
            rows.append({"method": lbl, "setting": setting, **_metrics(df), **cost})
    return rows


def llm_context_rows() -> list[dict]:
    """Paper methods on raw predictions (metrics only; costs live in
    tab_method_comparison.py and the paper table)."""
    rows = []
    for tag, lbl in QWEN_MODELS:
        for approach, name in (("ragtag", "RAGTAG-PS"), ("ragtag_debias_m3", "BRAGTAG-PS")):
            best_k = max(KS_RAG, key=lambda k: _metrics(load_raw_preds(tag, "PS", k, approach))["macro"])
            df = load_raw_preds(tag, "PS", best_k, approach)
            rows.append({"method": f"{name} k={best_k} ({lbl})", "setting": "PS",
                         **_metrics(df),
                         "train_time_s": None, "infer_time_s": None,
                         "total_time_s": None, "gpu_ram_mb": None})
        ft = pd.read_csv(RESULTS / "agnostic" / tag / "finetune_fixed" / "preds_finetune_fixed.csv",
                         usecols=["test_idx", "ground_truth", "predicted_label"])
        rows.append({"method": f"Fine-Tune-PA ({lbl})", "setting": "PA",
                     **_metrics(ft),
                     "train_time_s": None, "infer_time_s": None,
                     "total_time_s": None, "gpu_ram_mb": None})
    return rows


def _emit_tex(rows: list[dict]) -> str:
    def f3(x): return f"{x:.3f}"

    def fmin(x): return "--" if x is None else f"{x/60:.1f}"

    def fgb(x): return "--" if x is None else f"{x/1024:.1f}"

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Encoder baselines on the 11-project benchmark under both data "
        r"scopes (PS pooled concat-then-evaluate). SetFit follows the literature "
        r"configuration (batch 16, 1 epoch, 20 pair-generation iterations, "
        r"logistic-regression head); RoBERTa-base follows Colavito et al.'s recipe "
        r"(lr $2{\times}10^{-5}$, batch 16, weight decay 0.01, 15 epochs). "
        r"Encoders produce no invalid outputs by construction. "
        r"Times are wall-clock on a single GPU and exclude model load; "
        r"PS times are summed across the 11 per-project runs.}",
        r"  \label{tab:encoder-baselines}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{llcccccccccc}",
        r"    \toprule",
        r"    Method & Scope & Macro $F_1$ & Acc. & $P_{\text{macro}}$ & $R_{\text{macro}}$ & "
        r"$F_1^{\text{bug}}$ & $F_1^{\text{feat}}$ & $F_1^{\text{q}}$ & "
        r"RAM (GB) & Train (min) & Infer (min) \\",
        r"    \midrule",
    ]
    prev = None
    for r in rows:
        if prev is not None and r["method"] != prev:
            lines.append(r"    \addlinespace[2pt]")
        method_cell = r["method"] if r["method"] != prev else ""
        lines.append(
            f"    {method_cell} & {r['setting']} & {f3(r['macro'])} & {f3(r['accuracy'])} & "
            f"{f3(r['precision_macro'])} & {f3(r['recall_macro'])} & "
            f"{f3(r['f1_bug'])} & {f3(r['f1_feature'])} & {f3(r['f1_question'])} & "
            f"{fgb(r['gpu_ram_mb'])} & {fmin(r['train_time_s'])} & {fmin(r['infer_time_s'])} \\\\"
        )
        prev = r["method"]
    lines += [r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}", ""]
    return "\n".join(lines)


def _print_rows(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"{'Method':<28} {'Set':<3} {'macro':>7} {'acc':>7} {'P_mac':>7} {'R_mac':>7} "
          f"{'F1 bug':>7} {'F1 feat':>8} {'F1 q':>7} {'RAM GB':>7} {'Train':>9} {'Infer':>9}")
    for r in rows:
        ram = "--" if r["gpu_ram_mb"] is None else f"{r['gpu_ram_mb']/1024:.1f}"
        tr = "--" if r["train_time_s"] is None else f"{r['train_time_s']/60:.1f}m"
        inf = "--" if r["infer_time_s"] is None else f"{r['infer_time_s']/60:.1f}m"
        print(f"{r['method']:<28} {r['setting']:<3} {r['macro']:>7.4f} {r['accuracy']:>7.4f} "
              f"{r['precision_macro']:>7.4f} {r['recall_macro']:>7.4f} "
              f"{r['f1_bug']:>7.4f} {r['f1_feature']:>8.4f} {r['f1_question']:>7.4f} "
              f"{ram:>7} {tr:>9} {inf:>9}")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    enc = encoder_rows()
    _print_rows("Encoder baselines (raw predictions, pooled):", enc)
    ctx = llm_context_rows()
    _print_rows("Paper methods for context (raw predictions, pooled):", ctx)

    out = TABLES_DIR / "encoder_baselines.tex"
    out.write_text(_emit_tex(enc))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

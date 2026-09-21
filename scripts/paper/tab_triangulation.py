"""Metric-triangulation tables for the SANER 2027 revision.

Writes
  paper/tables/triangulation_all_cells.csv   every (method, setting, model, k) cell
  paper/tables/triangulation.tex             layout A: one paper-wide table* (22 rows)
  paper/tables/bragtag_results_ext.tex       layout B: bragtag_results + P/R + bug share
  paper/tables/method_comparison_ext.tex     layout B: method_comparison + macro P/R + question P/R
and prints the numbers quoted in the prose (findings F1-F7 of the plan).

Convention: pooled, raw predictions, best k on raw macro F1. No accuracy column
is emitted (author decision).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _triangulation import (  # noqa: E402
    KS_RAG, LABELS, MODEL_LABELS, REPO_ROOT, all_cells, best_configs,
)
import tab_method_comparison as tmc  # noqa: E402  (cost columns, raw mode)

TABLES_DIR = REPO_ROOT / "paper" / "tables"

METHOD_TEX = {
    ("votag", "PS"): r"\votag\ (PS)", ("votag", "PA"): r"\votag\ (PA)",
    ("zero_shot", "PA"): "Zero-shot",
    ("ragtag", "PS"): r"\ragtag\ (PS)", ("bragtag", "PS"): r"\bragtag\ (PS)",
    ("finetune", "PS"): "Fine-Tune (PS)", ("finetune", "PA"): "Fine-Tune (PA)",
}


def f3(x): return f"{x:.3f}"
def fpct(x): return f"{100*x:.1f}\\%"
def fk(k): return "--" if k == "-" else str(int(k))


def _pr_cells(r) -> str:
    return " & ".join(f3(r[f"p_{c}"]) + " & " + f3(r[f"r_{c}"]) for c in LABELS)


def _prf_cells(r) -> str:
    return " & ".join(f3(r[f"p_{c}"]) + " & " + f3(r[f"r_{c}"]) + " & " + f3(r[f"f1_{c}"]) for c in LABELS)


# ------------------------------------------------------------- layout A --
def emit_triangulation(best: pd.DataFrame) -> str:
    L = [
        r"\begin{table*}[t]",
        r"  \centering\color{violet}",
        r"  \caption{Metric triangulation of every method at its best configuration "
        r"(pooled over the 3{,}300 test issues; raw predictions, invalid outputs count as "
        r"incorrect). $P$/$R$ are per-class precision/recall; $P_{\text{mac}}$/$R_{\text{mac}}$ "
        r"their macro averages. Bug share is the fraction of test issues predicted as "
        r"bug (the true share is 33.3\%). \ragtag/\bragtag\ use the PS data scope at "
        r"their best $k$; zero-shot is $k{=}0$.}",
        r"  \label{tab:triangulation}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llccccrrrrrrcc}",
        r"    \toprule",
        r"    Method & Model & $k^*$ & Macro $F_1$ & $P_{\text{mac}}$ & $R_{\text{mac}}$ & "
        r"\multicolumn{2}{c}{Bug} & \multicolumn{2}{c}{Feature} & \multicolumn{2}{c}{Question} & "
        r"Bug share & Invalid \\",
        r"    \cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
        r"     & & & & & & $P$ & $R$ & $P$ & $R$ & $P$ & $R$ & & \\",
        r"    \midrule",
    ]
    prev = None
    for _, r in best.iterrows():
        key = (r.method, r.setting)
        if prev is not None and key != prev and not (prev[0] == "votag" and r.method == "votag"):
            L.append(r"    \addlinespace[2pt]")
        model = "--" if r.model == "-" else r.model
        L.append(f"    {METHOD_TEX[key]} & {model} & {fk(r.k)} & {f3(r.f1_macro)} & "
                 f"{f3(r.p_macro)} & {f3(r.r_macro)} & {_pr_cells(r)} & "
                 f"{fpct(r.share_bug)} & {fpct(r.invalid_rate)} \\\\")
        prev = key
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


# ------------------------------------------------------------- layout B --
def emit_bragtag_ext(best: pd.DataFrame) -> str:
    L = [
        r"\begin{table*}[t]",
        r"  \centering\color{blue}",
        r"  \caption{\ragtag\ vs.\ \bragtag\ at each model's best $k$ (PS, pooled, raw "
        r"predictions). $P$/$R$/$F_1$ are per-class precision, recall, and $F_1$; $P_{\text{mac}}$/$R_{\text{mac}}$ "
        r"are macro precision and recall.}",
        r"  \label{tab:bragtag-results}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3.5pt}",
        r"  \begin{tabular}{llccccrrrrrrrrrc}",
        r"    \toprule",
        r"    Model & Method & $k^*$ & Macro $F_1$ & $P_{\text{mac}}$ & $R_{\text{mac}}$ & "
        r"\multicolumn{3}{c}{Bug} & \multicolumn{3}{c}{Feature} & \multicolumn{3}{c}{Question} & "
        r"Invalid \\",
        r"    \cmidrule(lr){7-9}\cmidrule(lr){10-12}\cmidrule(lr){13-15}",
        r"     & & & & & & $P$ & $R$ & $F_1$ & $P$ & $R$ & $F_1$ & $P$ & $R$ & $F_1$ & \\",
        r"    \midrule",
    ]
    for i, m in enumerate(MODEL_LABELS):
        if i:
            L.append(r"    \addlinespace[2pt]")
        for method, name in (("ragtag", r"\ragtag"), ("bragtag", r"\bragtag")):
            r = best[(best.method == method) & (best.model == m)].iloc[0]
            cell = m if method == "ragtag" else ""
            L.append(f"    {cell} & {name} & {fk(r.k)} & {f3(r.f1_macro)} & {f3(r.p_macro)} & "
                     f"{f3(r.r_macro)} & {_prf_cells(r)} & {fpct(r.invalid_rate)} \\\\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def _cost_rows():
    """(model, method, name, cost dict) from tab_method_comparison, raw mode."""
    out = []
    for tag, m in tmc.MODELS:
        for c, (method, setting, name) in zip(
            (tmc._row_few_shot(tag, m, "ragtag"), tmc._row_few_shot(tag, m, "ragtag_debias_m3"),
             tmc._row_finetune(tag, m)),
            (("ragtag", "PS", r"\ragtag"), ("bragtag", "PS", r"\bragtag"), ("finetune", "PA", "Fine-Tune"))):
            out.append((m, method, setting, name, c))
    return out


def emit_method_comparison_ext(best: pd.DataFrame, cost_rows) -> str:
    """RQ4 comparison table: P/R columns as in bragtag_results_ext, macro F1
    second to last, +VOTAG (fallback macro F1) last. Costs live in
    emit_method_cost()."""
    L = [
        r"\begin{table*}[t]",
        r"  \centering\color{blue}",
        r"  \caption{\ragtag, \bragtag, and LoRA fine-tuning at each method's best configuration "
        r"(\ragtag/\bragtag\ at their best $k$ under the PS data scope, fine-tuning under PA; "
        r"pooled, raw predictions). $P$/$R$ are per-class precision/recall, $P_{\text{mac}}$/$R_{\text{mac}}$ "
        r"their macro averages. The +\votag\ column reports macro $F_1$ after using \votag\ as a "
        r"fallback for invalid LLM outputs.}",
        r"  \label{tab:method-comparison-ext}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3.2pt}",
        r"  \begin{tabular}{llcccrrrrrrccc}",
        r"    \toprule",
        r"    Model & Method & $k^*$ & $P_{\text{mac}}$ & $R_{\text{mac}}$ & "
        r"\multicolumn{2}{c}{Bug} & \multicolumn{2}{c}{Feature} & \multicolumn{2}{c}{Question} & "
        r"Macro $F_1$ & +\votag & Invalid \\",
        r"    \cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
        r"     & & & & & $P$ & $R$ & $P$ & $R$ & $P$ & $R$ & & & \\",
        r"    \midrule",
    ]
    prev = None
    for m, method, setting, name, c in cost_rows:
        r = best[(best.method == method) & (best.setting == setting) & (best.model == m)].iloc[0]
        assert abs(r.f1_macro - c["macro"]) < 5e-4, (m, method, r.f1_macro, c["macro"])
        if prev is not None and m != prev:
            L.append(r"    \addlinespace[2pt]")
        cell = m if m != prev else ""
        L.append(f"    {cell} & {name} & {fk(r.k)} & {f3(r.p_macro)} & {f3(r.r_macro)} & {_pr_cells(r)} & "
                 f"{f3(r.f1_macro)} & {f3(c['macro_rescued'])} & {fpct(r.invalid_rate)} \\\\")
        prev = m
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def emit_method_cost(cost_rows) -> str:
    """Compact cost table (column width): peak GPU memory and runtimes."""
    def fhrs(x): return f"{x/3600:.2f}"
    def fgb(x): return f"{x/1024:.1f}"
    L = [
        r"\begin{table}[t]",
        r"  \centering\color{blue}",
        r"  \caption{Computational cost of \ragtag, \bragtag, and LoRA fine-tuning at the "
        r"configurations of \Cref{tab:method-comparison-ext}. RAM is the observed peak GPU memory; "
        r"Train and Infer are wall-clock runtimes on a single GPU (\ragtag/\bragtag\ have no "
        r"training phase), and Total is their sum, excluding model load.}",
        r"  \label{tab:method-cost}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3.5pt}",
        r"  \begin{tabular}{llcccc}",
        r"    \toprule",
        r"    Model & Method & RAM (GB) & Train (h) & Infer (h) & Total (h) \\",
        r"    \midrule",
    ]
    prev = None
    for m, method, setting, name, c in cost_rows:
        if prev is not None and m != prev:
            L.append(r"    \addlinespace[2pt]")
        cell = m if m != prev else ""
        train = "--" if c["train_time_s"] == 0.0 else fhrs(c["train_time_s"])
        L.append(f"    {cell} & {name} & {fgb(c['gpu_ram_mb'])} & {train} & "
                 f"{fhrs(c['infer_time_s'])} & {fhrs(c['total_time_s'])} \\\\")
        prev = m
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


# ---------------------------------------------------------- prose numbers --
def print_prose_numbers(cells: pd.DataFrame, best: pd.DataFrame) -> None:
    def b(method, setting, model):
        return best[(best.method == method) & (best.setting == setting) & (best.model == model)].iloc[0]

    print("\n=== best configurations (F1 | Pmac Rmac | bug P/R | feat P/R | q P/R | bug share | inv | q->b) ===")
    for _, r in best.iterrows():
        print(f"{r.method:<9} {r.setting} {str(r.model):<8} k={str(r.k):<3} {r.f1_macro:.3f} | "
              f"{r.p_macro:.3f} {r.r_macro:.3f} | {r.p_bug:.3f}/{r.r_bug:.3f} | "
              f"{r.p_feature:.3f}/{r.r_feature:.3f} | {r.p_question:.3f}/{r.r_question:.3f} | "
              f"{100*r.share_bug:.1f}% | {100*r.invalid_rate:.1f}% | {100*r.conf_question_to_bug:.1f}%")

    print("\n=== 4-model means at best config ===")
    for method, setting in (("zero_shot", "PA"), ("ragtag", "PS"), ("bragtag", "PS"),
                            ("finetune", "PS"), ("finetune", "PA")):
        sub = best[(best.method == method) & (best.setting == setting)]
        print(f"{method:<9} {setting} " + " ".join(
            f"{c}={sub[c].mean():.3f}" for c in
            ("f1_macro", "p_macro", "r_macro", "p_bug", "r_bug", "p_question", "r_question",
             "p_feature", "r_feature", "share_bug", "share_question", "invalid_rate")))

    print("\n=== BRAGTAG - RAGTAG at best k ===")
    for m in MODEL_LABELS:
        a, c = b("ragtag", "PS", m), b("bragtag", "PS", m)
        print(f"{m:<8} dF1={c.f1_macro-a.f1_macro:+.3f} dPmac={c.p_macro-a.p_macro:+.3f} dRmac={c.r_macro-a.r_macro:+.3f} "
              f"q dP={c.p_question-a.p_question:+.3f} dR={c.r_question-a.r_question:+.3f} "
              f"bug dP={c.p_bug-a.p_bug:+.3f} dR={c.r_bug-a.r_bug:+.3f} "
              f"feat dP={c.p_feature-a.p_feature:+.3f} dR={c.r_feature-a.r_feature:+.3f} "
              f"dshare_bug={c.share_bug-a.share_bug:+.3f} b->q {100*a.conf_bug_to_question:.1f}->{100*c.conf_bug_to_question:.1f}%")

    print("\n=== FT-PA - RAG / FT-PA - BRAG / RAG - FT-PS ===")
    for m in MODEL_LABELS:
        rg, bg = b("ragtag", "PS", m), b("bragtag", "PS", m)
        fps, fpa = b("finetune", "PS", m), b("finetune", "PA", m)
        print(f"{m:<8} FTPA-RAG dF1={fpa.f1_macro-rg.f1_macro:+.3f} dPmac={fpa.p_macro-rg.p_macro:+.3f} | "
              f"FTPA-BRAG dF1={fpa.f1_macro-bg.f1_macro:+.3f} dPmac={fpa.p_macro-bg.p_macro:+.3f} "
              f"dRmac={fpa.r_macro-bg.r_macro:+.3f} | RAG-FTPS dF1={rg.f1_macro-fps.f1_macro:+.3f} | "
              f"FT PA-PS: bug dR={fpa.r_bug-fps.r_bug:+.3f} dP={fpa.p_bug-fps.p_bug:+.3f} "
              f"q dR={fpa.r_question-fps.r_question:+.3f} dP={fpa.p_question-fps.p_question:+.3f} "
              f"share {fps.share_bug:.3f}->{fpa.share_bug:.3f}")

    print("\n=== recall spread across classes (std) at best config ===")
    for method, setting in (("zero_shot", "PA"), ("ragtag", "PS"), ("bragtag", "PS"), ("finetune", "PA")):
        vals = [best[(best.method == method) & (best.setting == setting) & (best.model == m)].iloc[0]
                for m in MODEL_LABELS]
        print(f"{method:<9} " + " ".join(f"{r.model}={pd.Series([r.r_bug, r.r_feature, r.r_question]).std(ddof=0):.3f}" for r in vals))

    print("\n=== along k (RAGTAG-PS): bug share / question recall / invalid ===")
    for m in MODEL_LABELS:
        zs = b("zero_shot", "PA", m)
        sub = cells[(cells.method == "ragtag") & (cells.setting == "PS") & (cells.model == m)].sort_values("k")
        print(f"{m:<8} k0 {zs.share_bug:.3f}/{zs.r_question:.3f}/{100*zs.invalid_rate:.1f}% " +
              " ".join(f"k{int(r.k)} {r.share_bug:.3f}/{r.r_question:.3f}/{100*r.invalid_rate:.1f}%" for _, r in sub.iterrows()))
    print("\n=== VOTAG-PS along k: bug share / question recall / bug P / bug R ===")
    sub = cells[(cells.method == "votag") & (cells.setting == "PS")].sort_values("k")
    print(" ".join(f"k{int(r.k)} {r.share_bug:.3f}/{r.r_question:.3f}/{r.p_bug:.3f}/{r.r_bug:.3f}" for _, r in sub.iterrows()))


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cells = all_cells()
    cells.to_csv(TABLES_DIR / "triangulation_all_cells.csv", index=False, float_format="%.5f")
    best = best_configs(cells)
    (TABLES_DIR / "triangulation.tex").write_text(emit_triangulation(best))
    (TABLES_DIR / "bragtag_results_ext.tex").write_text(emit_bragtag_ext(best))
    cost_rows = _cost_rows()
    (TABLES_DIR / "method_comparison_ext.tex").write_text(emit_method_comparison_ext(best, cost_rows))
    (TABLES_DIR / "method_cost.tex").write_text(emit_method_cost(cost_rows))
    print_prose_numbers(cells, best)
    for f in ("triangulation_all_cells.csv", "triangulation.tex", "bragtag_results_ext.tex",
              "method_comparison_ext.tex", "method_cost.tex"):
        print(f"wrote paper/tables/{f}")


if __name__ == "__main__":
    main()

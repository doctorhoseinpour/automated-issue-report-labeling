"""Generate paper/tables/method_comparison.tex for §5.5 (Method Comparison).

For each Qwen size, reports three rows (one per method) at each method's
best configuration with the scope-matched \\votag\\ fallback applied:
  - \\ragtag-PS at best k, fallback = \\votag-PS at k=15
  - \\bragtag-PS at best k, fallback = \\votag-PS at k=15
  - Fine-Tune-PA,            fallback = \\votag-PA at k=16

Per row: best k, macro F1 (rescued), per-class F1 (rescued), invalid rate
(pre-fallback, the value the fallback rescued from), total GPU time
(inference only for \\ragtag/\\bragtag; training+inference for FT;
excludes model load per project convention), GPU RAM (peak).

Cost data sourcing for \\ragtag/\\bragtag:
  - 14B@k12 RAGTAG, 7B@k12 BRAGTAG, 14B@k15 BRAGTAG: cost_metrics.csv (results/)
  - 32B@k12 RAGTAG/BRAGTAG: cost_metrics.csv (fetched/w6, w7)
  - 3B@k3 RAGTAG, 7B@k6 RAGTAG, 3B@k6 BRAGTAG: recovered from per-project
    prediction-file mtime deltas (the original local run was overwritten
    by the k=12/15 extension; predictions survived but cost_metrics did
    not). mtime(preds_k{best_k}) - mtime(preds_k{prev_k}) is the wall time
    of inference at best_k for that project, summed across the 11 PS
    projects.

GPU RAM is essentially k-invariant for \\ragtag/\\bragtag (≤0.2% variation
across k for a given model) so we read it once per model from any
available row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rescue import (  # noqa: E402
    RESULTS,
    VTAG_BEST_K_PA,
    _project_list,
    load_raw_preds,
    load_rescued_preds,
)

# Default mode: "raw" (invalid LLM outputs count as incorrect; matches the
# headline reporting in §5.4). Pass --with-fallback on the CLI to regenerate
# the deployment-realistic-fallback variant for §7 / appendix use.
MODE = "raw"

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = REPO_ROOT / "paper" / "tables"

LABELS = ["bug", "feature", "question"]
KS_RAG = [1, 3, 6, 9, 12, 15]

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


PROJECTS = _project_list()
KS_ORDER = [0, 1, 3, 6, 9, 12, 15]  # for prev-k lookup in mtime recovery


def _ps_cost_path(model: str, approach: str, proj: str) -> Path:
    return (RESULTS / "project_specific" / proj / model / approach
            / "predictions" / "cost_metrics.csv")


def _ps_pred_mtime(model: str, approach: str, proj: str, k: int) -> float:
    label = "zero_shot" if k == 0 else f"k{k}"
    return (RESULTS / "project_specific" / proj / model / approach
            / "predictions" / f"preds_{label}.csv").stat().st_mtime


def _wall_time_from_cost_metrics(model: str, approach: str, k: int) -> float | None:
    """Try to recover wall_time_s for (model, approach, k) on PS from per-project
    cost_metrics.csv files. Returns None if any project lacks a row at this k.
    """
    label = f"k{k}"
    total = 0.0
    for proj in PROJECTS:
        f = _ps_cost_path(model, approach, proj)
        if not f.exists():
            return None
        df = pd.read_csv(f)
        match = df[df["k_label"] == label]
        if match.empty:
            return None
        total += float(match["wall_time_s"].iloc[0])
    return total


def _wall_time_from_fetched_32b(approach: str, k: int) -> float | None:
    """Fallback for 32B PS rows that exist only in fetched/ (waves w6/w7)."""
    wave = {"ragtag": "w6-qwen32b-ragtag", "ragtag_debias_m3": "w7-qwen32b-debias"}[approach]
    label = f"k{k}"
    total = 0.0
    matched = 0
    for proj in PROJECTS:
        proj_dash = proj.replace("_", "-").lower()  # wave dirnames are lowercase
        cands = list(Path("fetched").glob(
            f"{wave}-{proj_dash}/issues11k/project_specific/{proj}/"
            f"unsloth_Qwen2_5_32B_Instruct_bnb_4bit/{approach}/predictions/cost_metrics.csv"
        ))
        if not cands:
            return None
        df = pd.read_csv(cands[0])
        m = df[df["k_label"] == label]
        if m.empty:
            return None
        total += float(m["wall_time_s"].iloc[0])
        matched += 1
    return total if matched == len(PROJECTS) else None


def _wall_time_from_mtime(model: str, approach: str, k: int) -> float:
    """Recover PS wall_time at best k from prediction-file mtime deltas.

    delta_proj = mtime(preds_k{k}) - mtime(preds_k{prev_k}) where prev_k is the
    most recent k in KS_ORDER < k. Sum across all 11 PS projects. The original
    local run wrote predictions sequentially per k for the same project, so the
    delta is the inference wall-time of the current k for that project.
    """
    prev_k = max(p for p in KS_ORDER if p < k)
    total = 0.0
    for proj in PROJECTS:
        t_target = _ps_pred_mtime(model, approach, proj, k)
        t_prev   = _ps_pred_mtime(model, approach, proj, prev_k)
        total += (t_target - t_prev)
    return total


def _ps_gpu_ram_mb(model: str, approach: str) -> float:
    """Pick any project's gpu_peak_memory_mb (k-invariant up to ~0.2%)."""
    for proj in PROJECTS:
        f = _ps_cost_path(model, approach, proj)
        if f.exists():
            df = pd.read_csv(f)
            return float(df["gpu_peak_memory_mb"].max())
    raise RuntimeError(f"no cost_metrics for {model}/{approach}")


def _ft_pa_raw(model: str) -> pd.DataFrame:
    """Fine-Tune-PA raw predictions (single agnostic file)."""
    return pd.read_csv(
        RESULTS / "agnostic" / model / "finetune_fixed" / "preds_finetune_fixed.csv",
        usecols=["test_idx", "ground_truth", "predicted_label"],
    )


def _ft_pa_rescued(model: str) -> pd.DataFrame:
    """Fine-Tune-PA with \\votag-PA fallback for invalid outputs.

    \\votag-PA is the agnostic file at its best k. Substitution is by test_idx
    so per-issue alignment is exact.
    """
    ft = _ft_pa_raw(model)
    valid = set(LABELS)
    inv_mask = ~ft["predicted_label"].isin(valid)
    if inv_mask.any():
        vtg = pd.read_csv(
            RESULTS / "agnostic" / "vtag" / "predictions" / f"preds_k{VTAG_BEST_K_PA}.csv",
            usecols=["test_idx", "predicted_label"],
        ).set_index("test_idx")
        ft = ft.copy()
        ft.loc[inv_mask, "predicted_label"] = (
            ft.loc[inv_mask, "test_idx"].map(vtg["predicted_label"])
        )
    return ft


def _load_preds(model: str, setting: str, k: int, approach: str) -> pd.DataFrame:
    """Mode-aware few-shot loader: raw (default) or fallback-rescued."""
    if MODE == "raw":
        return load_raw_preds(model, setting, k, approach)
    return load_rescued_preds(model, setting, k, approach)


def _load_ft(model: str) -> pd.DataFrame:
    """Mode-aware FT-PA loader: raw (default) or fallback-rescued."""
    return _ft_pa_raw(model) if MODE == "raw" else _ft_pa_rescued(model)


def _row_few_shot(model_tag: str, model_lbl: str, approach: str) -> dict:
    """RAGTAG or BRAGTAG row at best k on PS.

    Reports both raw and \\votag-rescued macro $F_1$ in the same row; per-class
    $F_1$ is raw. Best k is selected on raw macro $F_1$ (independent of mode).
    """
    best_k = max(KS_RAG, key=lambda k: _macro(load_raw_preds(model_tag, "PS", k, approach)))
    raw = load_raw_preds(model_tag, "PS", best_k, approach)
    rescued = load_rescued_preds(model_tag, "PS", best_k, approach)
    pc = _per_class(raw)
    n_inv = (raw["predicted_label"] == "invalid").sum()

    # GPU time: try cost_metrics first; for 32B fall back to fetched/; otherwise
    # recover from prediction-file mtime deltas (lost-cost-data cells).
    gpu_time_s = _wall_time_from_cost_metrics(model_tag, approach, best_k)
    time_source = "cost_metrics"
    if gpu_time_s is None and "32B" in model_lbl:
        gpu_time_s = _wall_time_from_fetched_32b(approach, best_k)
        time_source = "fetched"
    if gpu_time_s is None:
        gpu_time_s = _wall_time_from_mtime(model_tag, approach, best_k)
        time_source = "mtime"

    gpu_ram_mb = _ps_gpu_ram_mb(model_tag, approach)
    # \ragtag / \bragtag have no training phase; gpu_time_s is inference only.
    return {
        "model": model_lbl,
        "method": approach,
        "setting": "PS",
        "best_k": str(best_k),
        "macro": _macro(raw),
        "macro_rescued": _macro(rescued),
        "f1_bug": pc["bug"],
        "f1_feature": pc["feature"],
        "f1_question": pc["question"],
        "invalid_pct": 100.0 * n_inv / len(raw),
        "train_time_s": 0.0,
        "infer_time_s": gpu_time_s,
        "total_time_s": gpu_time_s,
        "gpu_ram_mb": gpu_ram_mb,
        "time_source": time_source,
    }


def _row_finetune(model_tag: str, model_lbl: str) -> dict:
    """Fine-Tune-PA row.

    Reports both raw and \\votag-PA-rescued macro $F_1$; per-class $F_1$ is raw.
    """
    raw = _ft_pa_raw(model_tag)
    rescued = _ft_pa_rescued(model_tag)
    pc = _per_class(raw)
    n_inv = (~raw["predicted_label"].isin(LABELS)).sum()

    # FT cost: training_time_s and wall_time_s (inference); excludes model_load_time_s.
    cost = pd.read_csv(
        RESULTS / "agnostic" / model_tag / "finetune_fixed" / "cost_metrics.csv"
    ).iloc[0]
    train_s = float(cost["training_time_s"])
    infer_s = float(cost["wall_time_s"])
    gpu_ram_mb = float(cost["gpu_peak_memory_mb"])

    return {
        "model": model_lbl,
        "method": "finetune",
        "setting": "PA",
        "best_k": "--",
        "macro": _macro(raw),
        "macro_rescued": _macro(rescued),
        "f1_bug": pc["bug"],
        "f1_feature": pc["feature"],
        "f1_question": pc["question"],
        "invalid_pct": 100.0 * n_inv / len(raw),
        "train_time_s": train_s,
        "infer_time_s": infer_s,
        "total_time_s": train_s + infer_s,
        "gpu_ram_mb": gpu_ram_mb,
        "time_source": "cost_metrics",
    }


def _emit_tex(rows: list[dict]) -> str:
    def f3(x): return f"{x:.3f}"
    def fp(x): return f"{x:.2f}\\%"
    def fhrs(x): return f"{x/3600:.2f}"      # seconds -> hours
    def fgb(x): return f"{x/1024:.1f}"       # MB -> GB

    method_name = {
        "ragtag":           r"\ragtag",
        "ragtag_debias_m3": r"\bragtag",
        "finetune":         r"Fine-Tune",
    }

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{\ragtag, \bragtag, and Fine-Tune at each method's best configuration "
        r"(\ragtag/\bragtag\ at their best $k$ on PS, Fine-Tune on PA). "
        r"Macro $F_1$ and per-class $F_1$ are computed on raw predictions "
        r"(invalid LLM outputs count as incorrect); "
        r"the +\votag\ column reports macro $F_1$ when invalid LLM outputs are filled by the deployment-realistic \votag\ fallback "
        r"(\votag-PS for \ragtag/\bragtag, \votag-PA for Fine-Tune). "
        r"Train and Inference are wall-times on a single GPU "
        r"(\ragtag/\bragtag have no training phase). "
        r"Total is their sum and excludes model load. "
        r"RAM is the peak GPU memory observed. Pooled.}",
        r"  \label{tab:method-comparison}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{llcccccccccc}",
        r"    \toprule",
        r"    Model & Method & $k^*$ & Macro $F_1$ & +\votag & "
        r"$F_1^{\text{bug}}$ & $F_1^{\text{feat}}$ & $F_1^{\text{q}}$ & "
        r"RAM (GB) & Train (h) & Infer (h) & Total (h) \\",
        r"    \midrule",
    ]

    prev_model = None
    for r in rows:
        model_cell = r["model"] if r["model"] != prev_model else ""
        if r["model"] != prev_model and prev_model is not None:
            lines.append(r"    \addlinespace[2pt]")
        # \ragtag / \bragtag don't train: show em-dash for Train.
        train_cell = "--" if r["train_time_s"] == 0.0 else fhrs(r["train_time_s"])
        lines.append(
            f"    {model_cell} & {method_name[r['method']]} & "
            f"{r['best_k']} & {f3(r['macro'])} & {f3(r['macro_rescued'])} & "
            f"{f3(r['f1_bug'])} & {f3(r['f1_feature'])} & {f3(r['f1_question'])} & "
            f"{fgb(r['gpu_ram_mb'])} & {train_cell} & {fhrs(r['infer_time_s'])} & {fhrs(r['total_time_s'])} \\\\"
        )
        prev_model = r["model"]

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}%",
        r"  }",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    global MODE
    if "--with-fallback" in sys.argv:
        MODE = "rescued"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for tag, lbl in MODELS:
        rows.append(_row_few_shot(tag, lbl, "ragtag"))
        rows.append(_row_few_shot(tag, lbl, "ragtag_debias_m3"))
        rows.append(_row_finetune(tag, lbl))

    print(f"{'Model':<10} {'Method':<10} {'Set':<3} {'k*':<3}  "
          f"{'macro':>7}  {'F1 bug':>7}  {'F1 feat':>8}  {'F1 q':>6}  {'Inv':>6}  "
          f"{'Train h':>8}  {'Infer h':>8}  {'Total h':>8}  {'GPU GB':>7}  src")
    for r in rows:
        method = {"ragtag": "RAGTAG", "ragtag_debias_m3": "BRAGTAG",
                  "finetune": "FT"}[r["method"]]
        train_str = "--" if r["train_time_s"] == 0.0 else f"{r['train_time_s']/3600:>6.2f}h"
        print(f"{r['model']:<10} {method:<10} {r['setting']:<3} {r['best_k']:<3}  "
              f"{r['macro']:>7.4f}  "
              f"{r['f1_bug']:>7.4f}  {r['f1_feature']:>8.4f}  {r['f1_question']:>6.4f}  "
              f"{r['invalid_pct']:>5.2f}%  "
              f"{train_str:>8}  {r['infer_time_s']/3600:>6.2f}h  "
              f"{r['total_time_s']/3600:>6.2f}h  {r['gpu_ram_mb']/1024:>6.1f}G  "
              f"{r['time_source']}")

    out = TABLES_DIR / "method_comparison.tex"
    out.write_text(_emit_tex(rows))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

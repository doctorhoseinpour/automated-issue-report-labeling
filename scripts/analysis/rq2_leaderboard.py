"""RQ2.1: Master leaderboard rebuild for the 4 Qwen models.

Per Qwen size: best of each approach in each setting. VTAG floor row.
Plus a debias-best preview row that previews the RQ3 finding.

Project-specific overall = mean of per-project macro F1 across 11 projects
(macro-macro). Agnostic overall = pooled macro on 3,300 issues directly
from the eval CSV.

Outputs:
  docs/analysis/rq2_leaderboard.csv          # tidy long form
  docs/analysis/rq2_leaderboard_wide.md      # human-readable markdown
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _utils import (  # noqa: E402
    DOCS_ANALYSIS,
    MODEL_ORDER,
    ensure_dirs,
    rel,
)

METRIC_COLS = [
    "f1_macro", "precision_macro", "recall_macro", "accuracy",
    "invalid_rate",
    "f1_bug", "f1_feature", "f1_question",
    "precision_bug", "recall_bug",
    "precision_feature", "recall_feature",
    "precision_question", "recall_question",
]


def _ps_aggregate(df_ps: pd.DataFrame) -> pd.Series:
    """Per-project mean of metrics; n_projects = number of distinct projects."""
    out = df_ps[METRIC_COLS].mean(numeric_only=True)
    out["n_projects"] = df_ps["project"].nunique()
    return out


def _best_row(rows: pd.DataFrame, by: str = "f1_macro") -> pd.Series | None:
    if rows.empty:
        return None
    return rows.loc[rows[by].idxmax()]


def _best_ps_aggregate(df_all_ps: pd.DataFrame, by: str = "f1_macro") -> tuple[pd.Series, str | None] | None:
    """Find the k that maximizes per-project mean F1, return (aggregate row, k_label)."""
    if df_all_ps.empty:
        return None
    candidates = []
    for k_label, sub in df_all_ps.groupby("k_label"):
        agg = _ps_aggregate(sub)
        agg["k_label"] = k_label
        candidates.append(agg)
    cand_df = pd.DataFrame(candidates)
    return cand_df.loc[cand_df[by].idxmax()], cand_df.loc[cand_df[by].idxmax()]["k_label"]


def _agg_for(cells: pd.DataFrame, model: str, approach: str, setting: str,
             k_filter: list[str] | None = None) -> tuple[pd.Series, str | None] | None:
    """Best (across k) row for (model, approach, setting). For ps, aggregate across projects."""
    if approach == "vtag" or approach == "vtag_debias":
        sub = cells[(cells["approach"] == approach) & (cells["setting"] == setting)]
    else:
        sub = cells[
            (cells["model"] == model)
            & (cells["approach"] == approach)
            & (cells["setting"] == setting)
        ]
    if k_filter is not None:
        sub = sub[sub["k_label"].isin(k_filter)]
    if sub.empty:
        return None
    if setting == "agnostic":
        # Agnostic _overall row already gives pooled metrics.
        sub = sub[sub["project"] == "_overall"]
        if sub.empty:
            return None
        best = _best_row(sub)
        return best, str(best["k_label"])
    # project_specific: aggregate across 11 projects per k, pick best k.
    return _best_ps_aggregate(sub)


def _row(model: str, approach_label: str, setting: str, k_label: str | None,
         metrics: pd.Series) -> dict:
    out = {
        "model": model,
        "approach": approach_label,
        "setting": setting,
        "k_winner": k_label,
    }
    for c in METRIC_COLS:
        out[c] = metrics.get(c)
    out["n_projects"] = metrics.get("n_projects")
    return out


def _emit_per_model(cells: pd.DataFrame, model: str) -> list[dict]:
    rows: list[dict] = []
    # Zero-shot (agnostic only — ps zero-shot is missing for 3B/7B/14B; see audit).
    zs_ag = _agg_for(cells, model, "ragtag", "agnostic", k_filter=["zero_shot"])
    if zs_ag is not None:
        rows.append(_row(model, "zero_shot", "agnostic", zs_ag[1], zs_ag[0]))

    # RAGTAG plain — best of k1/k3/k6/k9
    for setting in ("agnostic", "project_specific"):
        best = _agg_for(cells, model, "ragtag", setting,
                        k_filter=["k1", "k3", "k6", "k9"])
        if best is not None:
            rows.append(_row(model, "ragtag", setting, best[1], best[0]))

    # Debias-RAGTAG — project_specific only.
    for setting in ("project_specific",):
        best = _agg_for(cells, model, "ragtag_debias", setting,
                        k_filter=["k1", "k3", "k6", "k9"])
        if best is not None:
            rows.append(_row(model, "ragtag_debias", setting, best[1], best[0]))

    # FT — finetune_fixed has only one k label; it's both settings.
    for setting in ("agnostic", "project_specific"):
        best = _agg_for(cells, model, "ft", setting)
        if best is not None:
            rows.append(_row(model, "ft", setting, best[1], best[0]))

    return rows


def _emit_vtag(cells: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for approach_label in ("vtag", "vtag_debias"):
        for setting in ("agnostic", "project_specific"):
            best = _agg_for(cells, "_no_model", approach_label, setting)
            if best is None:
                continue
            metrics, k_label = best
            rows.append(_row("(no LLM)", approach_label, setting, k_label, metrics))
    return rows


def _to_md(table: pd.DataFrame) -> str:
    """Compact markdown table grouped by model with one line per approach×setting."""
    lines: list[str] = []
    lines.append("# RQ2.1 — Master leaderboard")
    lines.append("")
    lines.append("Best macro F1 per (model, approach, setting). For project-specific")
    lines.append("settings, F1 is the per-project mean across 11 projects (macro-macro).")
    lines.append("Agnostic F1 is computed on the pooled 3,300-issue test set directly.")
    lines.append("")
    header = (
        "| Model | Approach | Setting | k | F1 macro | Acc | "
        "F1 bug | F1 feat | F1 q | Inv |"
    )
    sep = "|---|---|---|---|---:|---:|---:|---:|---:|---:|"
    for model in MODEL_ORDER + ["(no LLM)"]:
        sub = table[table["model"] == model]
        if sub.empty:
            continue
        lines.append("")
        lines.append(f"### {model}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for _, r in sub.iterrows():
            f1m = r["f1_macro"]
            acc = r["accuracy"]
            f1b = r["f1_bug"]
            f1f = r["f1_feature"]
            f1q = r["f1_question"]
            inv = r["invalid_rate"]
            kshow = r["k_winner"] if pd.notna(r["k_winner"]) else "-"
            lines.append(
                f"| {model} | {r['approach']} | {r['setting']} | "
                f"{kshow} | {f1m:.4f} | "
                f"{acc:.4f} | {f1b:.4f} | {f1f:.4f} | {f1q:.4f} | "
                f"{inv:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ensure_dirs()
    cells = pd.read_csv(DOCS_ANALYSIS / "all_cells.csv")
    rows: list[dict] = []
    for model in MODEL_ORDER:
        rows.extend(_emit_per_model(cells, model))
    rows.extend(_emit_vtag(cells))
    table = pd.DataFrame(rows)

    table.to_csv(DOCS_ANALYSIS / "rq2_leaderboard.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq2_leaderboard.csv')} ({len(table)} rows)")

    md = _to_md(table)
    md_out = DOCS_ANALYSIS / "rq2_leaderboard_wide.md"
    md_out.write_text(md)
    print(f"wrote {rel(md_out)}")
    print()
    print(md)


if __name__ == "__main__":
    main()

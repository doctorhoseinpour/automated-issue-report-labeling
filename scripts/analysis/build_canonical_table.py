"""Phase 0: build the canonical results table that every downstream
analysis reads.

Walks results/issues11k/, finds every eval CSV, classifies it as
(model, setting, project, approach, k), applies v1/v2 resolution
(prefer *_v2 where it exists), and writes:

  docs/analysis/all_cells.csv   - one row per cell with all metrics
  docs/analysis/preds_index.csv - path map for predictions CSVs
  docs/analysis/cost_index.csv  - path map for cost_metrics.csv files
  docs/analysis/coverage_audit.md - human-readable gap audit
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _utils import (  # noqa: E402
    ACTIVE_MODEL_TAGS,
    DOCS_ANALYSIS,
    MODEL_ORDER,
    PROJECTS,
    RESULTS_DIR,
    V2_PREFERENCE,
    ensure_dirs,
    model_name,
    parse_k_from_eval_filename,
    rel,
)


def _approach_from_dir(dir_basename: str) -> tuple[str, bool] | None:
    """Map a directory basename like 'ragtag_v2' to ('ragtag', used_v2=True).

    Returns None if the directory is not an approach we track.
    """
    if dir_basename == "ragtag":
        return "ragtag", False
    if dir_basename == "ragtag_v2":
        return "ragtag", True
    if dir_basename == "ragtag_debias_m3":
        return "ragtag_debias", False
    if dir_basename == "ragtag_debias_m3_v2":
        return "ragtag_debias", True
    if dir_basename == "finetune_fixed":
        return "ft", False
    if dir_basename == "vtag":
        return "vtag", False
    if dir_basename == "vtag_debias_m3":
        return "vtag_debias", False
    return None


def _eval_csv_paths(approach_dir: Path) -> list[Path]:
    """Find eval CSVs under an approach directory.

    Looks in <dir>/evaluations/eval_*.csv and also <dir>/eval_*.csv
    (some agnostic FT cells store the file at the top level).
    Skips per_project/* (those are downstream breakdowns of agnostic cells).
    Skips all_results.csv (aggregate, not per-k).
    """
    paths: list[Path] = []
    eval_subdir = approach_dir / "evaluations"
    if eval_subdir.is_dir():
        for p in sorted(eval_subdir.glob("eval_*.csv")):
            if p.name == "all_results.csv":
                continue
            paths.append(p)
    for p in sorted(approach_dir.glob("eval_*.csv")):
        # Only matters for FT cells that put the file at top level.
        if p.name == "all_results.csv":
            continue
        paths.append(p)
    return paths


def _read_eval_row(eval_csv: Path) -> dict | None:
    df = pd.read_csv(eval_csv)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _resolve_v2_pref(rows: list[dict]) -> list[dict]:
    """Drop v1 rows where a v2 row exists for the same logical cell.

    Cell key: (model_tag, setting, project, approach, k_label).
    """
    by_key: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (
            r["_model_tag"],
            r["setting"],
            r["project"],
            r["approach"],
            r["k_label"],
        )
        by_key.setdefault(key, []).append(r)

    resolved: list[dict] = []
    for key, group in by_key.items():
        if len(group) == 1:
            resolved.append(group[0])
            continue
        # Prefer used_v2=True if any present.
        v2_rows = [r for r in group if r["used_v2"]]
        if v2_rows:
            resolved.extend(v2_rows[:1])
        else:
            resolved.append(group[0])
    return resolved


def _walk_setting(setting: str) -> list[dict]:
    """Walk one setting (agnostic or project_specific) and yield row dicts.

    Each row carries _model_tag for downstream resolution; that is dropped
    in the final CSV.
    """
    rows: list[dict] = []
    setting_dir = RESULTS_DIR / setting
    if not setting_dir.is_dir():
        return rows

    if setting == "agnostic":
        # agnostic/<model>/<approach> and agnostic/{vtag,vtag_debias_m3}
        for entry in sorted(setting_dir.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if name in ("neighbors",):
                continue
            if name in ACTIVE_MODEL_TAGS:
                # per-model approaches under it
                for ap_dir in sorted(entry.iterdir()):
                    if not ap_dir.is_dir():
                        continue
                    info = _approach_from_dir(ap_dir.name)
                    if info is None:
                        continue
                    approach, used_v2 = info
                    if used_v2 and (name, setting, approach) not in V2_PREFERENCE:
                        # v2 dir exists but we did not flag it for use; still
                        # capture so resolution can see both. We mark as v2.
                        pass
                    for ev in _eval_csv_paths(ap_dir):
                        rows.append(_make_row(
                            ev, name, setting, "_overall", approach, used_v2,
                        ))
            elif name == "vtag" or name == "vtag_debias_m3":
                approach = "vtag" if name == "vtag" else "vtag_debias"
                for ev in _eval_csv_paths(entry):
                    rows.append(_make_row(
                        ev, "_no_model", setting, "_overall", approach, False,
                    ))
            elif name in ("unsloth_Llama_3_2_3B_Instruct",
                          "unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit"):
                # Llama legacy: skip per plan decision.
                continue
            else:
                # Unrecognized — skip but flag in coverage audit.
                continue
        return rows

    # project_specific
    for proj_dir in sorted(setting_dir.iterdir()):
        if not proj_dir.is_dir():
            continue
        proj = proj_dir.name
        if proj not in PROJECTS:
            continue
        for entry in sorted(proj_dir.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if name in ("neighbors",):
                continue
            if name in ACTIVE_MODEL_TAGS:
                for ap_dir in sorted(entry.iterdir()):
                    if not ap_dir.is_dir():
                        continue
                    info = _approach_from_dir(ap_dir.name)
                    if info is None:
                        continue
                    approach, used_v2 = info
                    for ev in _eval_csv_paths(ap_dir):
                        rows.append(_make_row(
                            ev, name, setting, proj, approach, used_v2,
                        ))
            elif name == "vtag" or name == "vtag_debias_m3":
                approach = "vtag" if name == "vtag" else "vtag_debias"
                for ev in _eval_csv_paths(entry):
                    rows.append(_make_row(
                        ev, "_no_model", setting, proj, approach, False,
                    ))
            elif name in ("unsloth_Llama_3_2_3B_Instruct",
                          "unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit"):
                continue
    return rows


def _make_row(eval_csv: Path, model_tag: str, setting: str, project: str,
              approach: str, used_v2: bool) -> dict:
    k_label, k_int = parse_k_from_eval_filename(eval_csv.name)
    row_data = _read_eval_row(eval_csv) or {}
    row = {
        "model_tag": model_tag if model_tag != "_no_model" else "",
        "model": model_name(model_tag) if model_tag != "_no_model" else "",
        "setting": setting,
        "project": project,
        "approach": approach,
        "k_label": k_label,
        "k": k_int,
        "used_v2": used_v2,
        "source_path": rel(eval_csv),
        "_model_tag": model_tag,  # internal, dropped before write
    }
    # Fold in eval metrics, prefixed where they collide.
    for col, val in row_data.items():
        if col in row or col.startswith("_"):
            continue
        row[col] = val
    return row


def _index_predictions_and_costs(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each surviving cell, locate its predictions CSV and cost CSV."""
    pred_records = []
    cost_records = []
    for r in rows:
        ev_path = Path(r["source_path"])
        # The predictions live at <approach_dir>/predictions/preds_<klabel>.csv
        # or for FT, <approach_dir>/preds_finetune_fixed.csv
        # Walk up: source path is e.g. results/.../<approach>/evaluations/eval_X.csv
        # OR results/.../<approach>/eval_X.csv (top-level FT case)
        full = (Path("/home/ahosein/llm-labler") / ev_path).resolve()
        if full.parent.name == "evaluations":
            approach_dir = full.parent.parent
        else:
            approach_dir = full.parent

        klab = r["k_label"]
        pred_candidates = [
            approach_dir / "predictions" / f"preds_{klab}.csv",
            approach_dir / f"preds_{klab}.csv",
        ]
        for cand in pred_candidates:
            if cand.is_file():
                pred_records.append({
                    "model": r["model"],
                    "model_tag": r["model_tag"],
                    "setting": r["setting"],
                    "project": r["project"],
                    "approach": r["approach"],
                    "k_label": klab,
                    "k": r["k"],
                    "predictions_path": rel(cand),
                })
                break

        cost_candidates = [
            approach_dir / "predictions" / "cost_metrics.csv",
            approach_dir / "cost_metrics.csv",
        ]
        for cand in cost_candidates:
            if cand.is_file():
                cost_records.append({
                    "model": r["model"],
                    "model_tag": r["model_tag"],
                    "setting": r["setting"],
                    "project": r["project"],
                    "approach": r["approach"],
                    "k_label": klab,
                    "k": r["k"],
                    "cost_path": rel(cand),
                })
                break

    pred_df = pd.DataFrame(pred_records).drop_duplicates()
    cost_df = pd.DataFrame(cost_records).drop_duplicates()
    return pred_df, cost_df


def _coverage_audit(df: pd.DataFrame) -> str:
    """Human-readable audit of expected vs actual cells."""
    lines = ["# Coverage audit", ""]

    # Expected LLM cells: 4 models × {ag, ps×11} × 6 (zero_shot, k1,3,6,9, ft)
    # zero_shot is a special k label; ft is "finetune_fixed"
    # ragtag_debias is project_specific only
    def n_in(approach: str, setting: str, k_label: str | None = None) -> int:
        sub = df[(df["approach"] == approach) & (df["setting"] == setting)]
        if k_label is not None:
            sub = sub[sub["k_label"] == k_label]
        return len(sub)

    # Build a compact expected-vs-actual table.
    expected_cells: list[tuple[str, str, str | None, int]] = [
        ("ragtag", "agnostic", "zero_shot", 4),         # 4 models
        ("ragtag", "agnostic", "k1", 4),
        ("ragtag", "agnostic", "k3", 4),
        ("ragtag", "agnostic", "k6", 4),
        ("ragtag", "agnostic", "k9", 4),
        ("ft",     "agnostic", "finetune_fixed", 4),
        ("ragtag", "project_specific", "zero_shot", 44),  # 4 × 11
        ("ragtag", "project_specific", "k1", 44),
        ("ragtag", "project_specific", "k3", 44),
        ("ragtag", "project_specific", "k6", 44),
        ("ragtag", "project_specific", "k9", 44),
        ("ragtag_debias", "project_specific", "k1", 44),
        ("ragtag_debias", "project_specific", "k3", 44),
        ("ragtag_debias", "project_specific", "k6", 44),
        ("ragtag_debias", "project_specific", "k9", 44),
        ("ft",     "project_specific", "finetune_fixed", 44),
        ("vtag",        "agnostic", None, 22),  # 22 k values
        ("vtag",        "project_specific", None, 22 * 11),
        ("vtag_debias", "agnostic", None, 4),
        ("vtag_debias", "project_specific", None, 4 * 11),
    ]
    lines.append("| approach | setting | k | expected | actual |")
    lines.append("|---|---|---|---:|---:|")
    for ap, setting, klab, expected in expected_cells:
        actual = n_in(ap, setting, klab)
        flag = "" if actual == expected else " ⚠"
        lines.append(f"| {ap} | {setting} | {klab or '*'} | {expected} | {actual}{flag} |")
    lines.append("")

    # Invalid-rate sanity check: flag any cell with invalid_rate > 0.10.
    bad = df[df.get("invalid_rate", 0) > 0.10][[
        "model", "setting", "project", "approach", "k_label", "invalid_rate",
    ]]
    if not bad.empty:
        lines.append("## Cells with invalid_rate > 0.10")
        lines.append("")
        lines.append("| model | setting | project | approach | k | invalid_rate |")
        lines.append("|---|---|---|---|---|---:|")
        for _, r in bad.iterrows():
            lines.append(
                f"| {r['model']} | {r['setting']} | {r['project']} | "
                f"{r['approach']} | {r['k_label']} | {r['invalid_rate']:.4f} |",
            )
        lines.append("")
    else:
        lines.append("All cells have invalid_rate ≤ 0.10.")
        lines.append("")

    # Support sanity check: agnostic _overall expects 1100 per class; ps expects 100.
    def support_check(setting: str, project: str, expected: int) -> list[str]:
        sub = df[(df["setting"] == setting) & (df["project"] == project)]
        out = []
        for _, r in sub.iterrows():
            for cls in ("bug", "feature", "question"):
                col = f"support_{cls}"
                if col in r and pd.notna(r[col]) and int(r[col]) != expected:
                    out.append(
                        f"- {r['approach']} {r['k_label']} {r['model']} "
                        f"{cls}={int(r[col])} (expected {expected}) at {r['source_path']}",
                    )
        return out

    issues = support_check("agnostic", "_overall", 1100)
    for proj in PROJECTS:
        issues += support_check("project_specific", proj, 100)
    if issues:
        lines.append("## Support imbalances")
        lines.append("")
        lines.extend(issues)
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    rows: list[dict] = []
    for setting in ("agnostic", "project_specific"):
        rows.extend(_walk_setting(setting))

    rows = _resolve_v2_pref(rows)
    for r in rows:
        r.pop("_model_tag", None)

    df = pd.DataFrame(rows)
    # Stable column ordering.
    front = [
        "model", "model_tag", "setting", "project", "approach", "k_label", "k",
        "used_v2", "total_issues", "invalid_count", "invalid_rate", "accuracy",
        "f1_macro", "precision_macro", "recall_macro",
        "f1_weighted", "precision_weighted", "recall_weighted",
        "f1_bug", "precision_bug", "recall_bug", "support_bug",
        "f1_feature", "precision_feature", "recall_feature", "support_feature",
        "f1_question", "precision_question", "recall_question", "support_question",
        "source_path",
    ]
    other = [c for c in df.columns if c not in front]
    df = df[[c for c in front if c in df.columns] + other]

    out = DOCS_ANALYSIS / "all_cells.csv"
    df.to_csv(out, index=False)
    print(f"wrote {rel(out)} ({len(df)} rows)")

    pred_df, cost_df = _index_predictions_and_costs(rows)
    pred_out = DOCS_ANALYSIS / "preds_index.csv"
    pred_df.to_csv(pred_out, index=False)
    print(f"wrote {rel(pred_out)} ({len(pred_df)} rows)")

    cost_out = DOCS_ANALYSIS / "cost_index.csv"
    cost_df.to_csv(cost_out, index=False)
    print(f"wrote {rel(cost_out)} ({len(cost_df)} rows)")

    audit = _coverage_audit(df)
    audit_out = DOCS_ANALYSIS / "coverage_audit.md"
    audit_out.write_text(audit)
    print(f"wrote {rel(audit_out)}")


if __name__ == "__main__":
    main()

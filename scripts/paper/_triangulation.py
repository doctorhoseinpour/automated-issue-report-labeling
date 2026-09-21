"""Shared loaders + metric computation for the metric-triangulation analysis
(SANER 2027 revision, ESEM review B "methodological triangulation").

Every cell is evaluated on RAW predictions (invalid outputs count as wrong),
pooled concat-then-evaluate for PS (see paper/sections/04_setup.tex).

Metrics per cell: invalid rate, macro precision / recall / F1, per-class
precision / recall / F1, per-class predicted share, row-normalised confusion.
NOTE: on this exactly balanced test set macro recall == accuracy; accuracy is
computed only as an internal sanity check and is never emitted to LaTeX
(author decision, 2026-09-17).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rescue import RESULTS, _project_list, load_raw_preds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = ["bug", "feature", "question"]
USECOLS = ["ground_truth", "predicted_label"]

MODELS = [
    ("unsloth_Qwen2_5_3B_Instruct_bnb_4bit",  "Qwen-3B"),
    ("unsloth_Qwen2_5_7B_Instruct_bnb_4bit",  "Qwen-7B"),
    ("unsloth_Qwen2_5_14B_Instruct_bnb_4bit", "Qwen-14B"),
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "Qwen-32B"),
]
MODEL_LABELS = [lbl for _, lbl in MODELS]
KS_RAG = [1, 3, 6, 9, 12, 15]
KS_VTAG = list(range(1, 21)) + [25, 30]


# ---------------------------------------------------------------- loaders --
def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ground_truth"] = df["ground_truth"].astype(str).str.lower().str.strip()
    p = df["predicted_label"].astype(str).str.lower().str.strip()
    df["predicted_label"] = np.where(p.isin(LABELS), p, "invalid")
    return df[USECOLS]


def _read(path: Path) -> pd.DataFrame | None:
    return _norm(pd.read_csv(path, usecols=USECOLS)) if path.exists() else None


def _pool(paths: list[Path]) -> pd.DataFrame | None:
    parts = [_read(p) for p in paths]
    if any(p is None for p in parts):
        return None
    return pd.concat(parts, ignore_index=True)


def load_votag(setting: str, k: int) -> pd.DataFrame | None:
    if setting == "PA":
        return _read(RESULTS / "agnostic" / "vtag" / "predictions" / f"preds_k{k}.csv")
    return _pool([RESULTS / "project_specific" / pr / "vtag" / "predictions" / f"preds_k{k}.csv"
                  for pr in _project_list()])


def load_zero_shot(model: str) -> pd.DataFrame | None:
    return _read(RESULTS / "agnostic" / model / "ragtag" / "predictions" / "preds_zero_shot.csv")


def load_few_shot(model: str, setting: str, k: int, approach: str) -> pd.DataFrame | None:
    try:
        return _norm(load_raw_preds(model, setting, k, approach))
    except FileNotFoundError:
        return None


def load_finetune(model: str, setting: str) -> pd.DataFrame | None:
    if setting == "PA":
        return _read(RESULTS / "agnostic" / model / "finetune_fixed" / "preds_finetune_fixed.csv")
    return _pool([RESULTS / "project_specific" / pr / model / "finetune_fixed" / "preds_finetune_fixed.csv"
                  for pr in _project_list()])


# ---------------------------------------------------------------- metrics --
def metrics(df: pd.DataFrame) -> dict:
    y, p = df["ground_truth"], df["predicted_label"]
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, labels=LABELS, average="macro", zero_division=0)
    acc = accuracy_score(y, p)
    if abs(acc - mr) > 1e-9:
        raise RuntimeError("test set not balanced: accuracy != macro recall")
    out = {"n": len(df), "invalid_rate": float((p == "invalid").mean()),
           "p_macro": mp, "r_macro": mr, "f1_macro": mf}
    for i, lab in enumerate(LABELS):
        out[f"p_{lab}"], out[f"r_{lab}"], out[f"f1_{lab}"] = pr[i], rc[i], f1[i]
        out[f"support_{lab}"] = int(sup[i])
        out[f"share_{lab}"] = float((p == lab).mean())
    for t in LABELS:
        mask = y == t
        for pl in LABELS + ["invalid"]:
            out[f"conf_{t}_to_{pl}"] = float(((p == pl) & mask).sum() / mask.sum())
    return out


# ------------------------------------------------------------- all cells --
def all_cells() -> pd.DataFrame:
    rows = []

    def add(method, setting, model, k, df):
        if df is None:
            print(f"  [skip] {method} {setting} {model} k={k}", file=sys.stderr)
            return
        rows.append({"method": method, "setting": setting, "model": model, "k": k, **metrics(df)})

    for k in KS_VTAG:
        for s in ("PS", "PA"):
            add("votag", s, "-", k, load_votag(s, k))
    for tag, lbl in MODELS:
        add("zero_shot", "PA", lbl, 0, load_zero_shot(tag))
        for k in KS_RAG:
            add("ragtag", "PS", lbl, k, load_few_shot(tag, "PS", k, "ragtag"))
            add("ragtag", "PA", lbl, k, load_few_shot(tag, "PA", k, "ragtag"))
            add("bragtag", "PS", lbl, k, load_few_shot(tag, "PS", k, "ragtag_debias_m3"))
        for s in ("PS", "PA"):
            add("finetune", s, lbl, "-", load_finetune(tag, s))
    try:  # encoder baselines are optional (branch encoder-baselines)
        from _encoders import ENCODERS, load_encoder_preds
        for tag, approach, fname, lbl in ENCODERS:
            for s in ("PA", "PS"):
                try:
                    add("encoder", s, lbl, "-", _norm(load_encoder_preds(tag, approach, fname, s)))
                except FileNotFoundError:
                    pass
    except ImportError:
        pass
    return pd.DataFrame(rows)


def _best(cells: pd.DataFrame, method: str, setting: str, model: str) -> pd.Series:
    sub = cells[(cells.method == method) & (cells.setting == setting) & (cells.model == model)]
    if sub.empty:
        raise RuntimeError(f"no cells for {method}/{setting}/{model}")
    return sub.loc[sub["f1_macro"].idxmax()]


def best_configs(cells: pd.DataFrame) -> pd.DataFrame:
    """The 22 best-configuration rows in paper order (best k on raw macro F1)."""
    out = [_best(cells, "votag", "PS", "-"), _best(cells, "votag", "PA", "-")]
    for method, setting in (("zero_shot", "PA"), ("ragtag", "PS"), ("bragtag", "PS"),
                            ("finetune", "PS"), ("finetune", "PA")):
        out += [_best(cells, method, setting, m) for m in MODEL_LABELS]
    return pd.DataFrame(out).reset_index(drop=True)

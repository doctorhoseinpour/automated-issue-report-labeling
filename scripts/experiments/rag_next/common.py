"""Shared helpers for the rag_next exploration (post-deadline study).

Data layout (lab machine, never inside the archival results tree except the
exploration/ subfolder):
  results/issues11k/exploration/rag_next/splits/pool.csv
      all 6,600 issues: uid, split (train/test), role (inner/dev/test), repo,
      proj, label, created_at, title, body, gidx (test order), tidx (train order)

Dev protocol: inside the paper's train split, per (repo, label) group sorted by
created_at, the newest DEV_PER_GROUP issues are `dev`, the older ones `inner`.
All design/hyper-parameter choices are made by fitting on `inner` and scoring on
`dev`. The paper's test split (role == test) is touched only for final runs.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RES = Path(os.environ.get("RESULTS_DIR", REPO / "results" / "issues11k"))
EXP = RES / "exploration" / "rag_next"
SPLITS = EXP / "splits"
FEATS = EXP / "features"
LABELS = ["bug", "feature", "question"]
LAB2ID = {l: i for i, l in enumerate(LABELS)}
DEV_PER_GROUP = 30


def load_pool() -> pd.DataFrame:
    df = pd.read_csv(SPLITS / "pool.csv", keep_default_na=False)
    for c in ["title", "body"]:
        df[c] = df[c].astype(str)
    return df


def issue_text(title: str, body: str) -> str:
    """Same text as RAGTAG's format_issue / SetFit's issue_text."""
    return f"Title: {title}\nBody: {body}"


def macro_f1(y, p) -> float:
    y = np.asarray(y); p = np.asarray(p)
    f = []
    for c in range(3):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def per_class_f1(y, p):
    y = np.asarray(y); p = np.asarray(p)
    out = []
    for c in range(3):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return out


def boot_diff(y, pa, pb, B=2000, seed=0, strata=None):
    """Paired issue-level bootstrap: macro_f1(pb) - macro_f1(pa), percentile 95% CI.
    Invalid predictions should be encoded as -1 (always wrong)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pa = np.asarray(pa); pb = np.asarray(pb)
    n = len(y)
    d = np.empty(B)
    for b in range(B):
        ii = rng.integers(0, n, n)
        d[b] = macro_f1(y[ii], pb[ii]) - macro_f1(y[ii], pa[ii])
    return macro_f1(y, pb) - macro_f1(y, pa), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

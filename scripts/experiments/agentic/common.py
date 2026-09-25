"""Shared helpers for the agentic-IRC exploration (post-deadline study, 2026-09-24).

Dev protocol: the rag_next study's split (splits/pool.csv): inside the paper's train
split, the newest 30 issues of each (repo, label) group are `dev` (990), the older ones
`inner` (2,310). Dev-phase components are fit on inner and scored on dev. The paper's
test split is not read by anything in this directory except headroom_agentic.py, which
only reads the archival test predictions for descriptive headroom numbers.

Inputs are snapshotted into results/issues11k/exploration/agentic/inputs/ (see
snapshot_inputs) so that later rag_next reruns cannot change these pilots.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RES = Path(os.environ.get("RESULTS_DIR", REPO / "results" / "issues11k"))
RAGNEXT = RES / "exploration" / "rag_next"
EXP = RES / "exploration" / "agentic"
INP = EXP / "inputs"
LABELS = ["bug", "feature", "question"]
LAB2ID = {l: i for i, l in enumerate(LABELS)}
SETFIT_BODY = "Collab-uniba_github-issues-mpnet-st-e10"
# a stand-in score file can be supplied for smoke tests before the SetFit-dev run exists
SETFIT_DEV = Path(os.environ.get("AGENTIC_SETFIT_DEV", INP / "setfit_dev.csv"))


def snapshot_inputs():
    """Copy the rag_next artifacts these pilots depend on (idempotent)."""
    INP.mkdir(parents=True, exist_ok=True)
    for src, dst in [(RAGNEXT / "splits" / "pool.csv", INP / "pool.csv"),
                     (RAGNEXT / "features" / "minilm_raw.npy", INP / "minilm_raw.npy"),
                     (RAGNEXT / "features" / "nb_PS_raw_dev.npz", INP / "nb_PS_raw_dev.npz")]:
        if not dst.exists():
            shutil.copy2(src, dst)
    sf = INP / "setfit_dev.csv"
    if not sf.exists() and (RAGNEXT / "setfit_dev" / SETFIT_BODY).exists():
        pool = load_pool()
        rows = []
        for p in sorted(pool.proj.unique()):
            dev = pd.read_csv(RAGNEXT / "splits" / "ps" / p / "dev.csv", usecols=["uid"])
            pr = pd.read_csv(RAGNEXT / "setfit_dev" / SETFIT_BODY / p / "predictions" / "preds_setfit.csv")
            assert len(dev) == len(pr), p
            for u, r in zip(dev["uid"], pr["raw_output"]):
                d = json.loads(r)
                rows.append({"uid": int(u), **{f"p_{l}": d[l] for l in LABELS}})
        pd.DataFrame(rows).to_csv(sf, index=False)


def load_pool() -> pd.DataFrame:
    f = INP / "pool.csv" if (INP / "pool.csv").exists() else RAGNEXT / "splits" / "pool.csv"
    df = pd.read_csv(f, keep_default_na=False)
    for c in ["title", "body"]:
        df[c] = df[c].astype(str)
    return df


def load_setfit_dev() -> pd.DataFrame:
    """uid, p_bug, p_feature, p_question, pred, margin (p1 - p2)."""
    d = pd.read_csv(SETFIT_DEV)
    P = d[[f"p_{l}" for l in LABELS]].to_numpy()
    s = np.sort(P, 1)
    d["pred"] = [LABELS[i] for i in P.argmax(1)]
    d["margin"] = s[:, -1] - s[:, -2]
    return d


def routed_uids(n_route: int = 300) -> np.ndarray:
    """Dev issues with the smallest SetFit margin (ties broken by uid)."""
    d = load_setfit_dev().sort_values(["margin", "uid"])
    return d["uid"].to_numpy()[:n_route]


# ------------------------------------------------------------------ retrieval
class Retriever:
    """Project-specific retrieval over the phase's index issues (dev phase: role == inner),
    reproducing the paper's space (all-MiniLM-L6-v2, cosine)."""

    def __init__(self, pool: pd.DataFrame, index_role: str = "inner"):
        self.pool = pool.set_index("uid", drop=False)
        E = np.load(INP / "minilm_raw.npy")
        assert len(E) == len(pool)
        self.E = {int(u): E[i] for i, u in enumerate(pool.uid)}
        idx = pool[pool.role == index_role]
        self.index = {p: g.uid.to_numpy() for p, g in idx.groupby("proj")}
        self._st = None

    def _embed(self, text: str) -> np.ndarray:
        if self._st is None:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        v = self._st.encode([re.sub(r"\s+", " ", text).strip()], convert_to_numpy=True)[0]
        return v / max(np.linalg.norm(v), 1e-12)

    def similar(self, proj: str, qvec: np.ndarray, label: str | None, k: int, exclude=()):
        cand = self.index[proj]
        if label is not None:
            cand = cand[self.pool.loc[cand, "label"].to_numpy() == label]
        cand = np.array([c for c in cand if c not in exclude])
        if len(cand) == 0:
            return []
        M = np.stack([self.E[int(c)] for c in cand])
        s = M @ qvec
        o = np.argsort(-s, kind="stable")[:k]
        return [(int(cand[i]), float(s[i])) for i in o]

    def similar_to_issue(self, uid: int, label: str | None, k: int):
        proj = self.pool.loc[uid, "proj"]
        return self.similar(proj, self.E[int(uid)], label, k, exclude=(uid,))

    def search(self, proj: str, query: str, label: str | None, k: int):
        return self.similar(proj, self._embed(query), label, k)

    def label_stats(self, proj: str, pattern: str, max_chars: int = 6000):
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return None, f"invalid regex: {e}"
        cand = self.index[proj]
        sub = self.pool.loc[cand]
        hit = [bool(rx.search((t + "\n" + b)[:max_chars])) for t, b in zip(sub.title, sub.body)]
        sub = sub.assign(hit=hit)
        out = {}
        for l in LABELS:
            g = sub[sub.label == l]
            out[l] = (int(g.hit.sum()), len(g), g[g.hit].title.head(2).tolist())
        return out, None


# ------------------------------------------------------------------ metrics
def macro_f1(y, p) -> float:
    y = np.asarray(y); p = np.asarray(p)
    f = []
    for c in LABELS:
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def per_class_f1(y, p):
    y = np.asarray(y); p = np.asarray(p)
    out = []
    for c in LABELS:
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return out


def boot_diff(y, pa, pb, B=2000, seed=0):
    """Paired issue-level bootstrap of macro_f1(pb) - macro_f1(pa): (diff, lo, hi)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); pa = np.asarray(pa); pb = np.asarray(pb)
    n = len(y)
    d = np.empty(B)
    for b in range(B):
        ii = rng.integers(0, n, n)
        d[b] = macro_f1(y[ii], pb[ii]) - macro_f1(y[ii], pa[ii])
    return macro_f1(y, pb) - macro_f1(y, pa), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

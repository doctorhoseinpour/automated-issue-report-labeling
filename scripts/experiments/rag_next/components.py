"""Base components for the rag_next read-out / fusion experiments.

Every component returns honest out-of-sample class probabilities for the query
issues of a phase:
  phase "dev" : fit on role == inner, predict role == dev
  phase "test": fit on split == train, predict split == test
Rows are aligned to `query_uids(pool, phase)` (pool order).
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import EXP, FEATS, LAB2ID, LABELS, RES, SPLITS, load_pool

warnings.filterwarnings("ignore")


@lru_cache(maxsize=1)
def pool_cached():
    return load_pool()


def masks(phase):
    pool = pool_cached()
    if phase == "dev":
        return (pool.role == "inner").to_numpy(), (pool.role == "dev").to_numpy()
    return (pool.split == "train").to_numpy(), (pool.split == "test").to_numpy()


def query_uids(phase):
    pool = pool_cached()
    return pool.uid.to_numpy()[masks(phase)[1]]


def labels_of(uids):
    pool = pool_cached().set_index("uid")
    return pool.loc[uids, "label"].map(LAB2ID).to_numpy()


def proj_of(uids):
    pool = pool_cached().set_index("uid")
    return pool.loc[uids, "proj"].to_numpy()


# --------------------------------------------------------------------- LLM features
@lru_cache(maxsize=8)
def load_feats(name):
    z = np.load(FEATS / f"{name}.npz")
    return {k: z[k] for k in z.files}


def _rows(z, uids):
    pos = {int(u): i for i, u in enumerate(z["uids"])}
    return np.array([pos[int(u)] for u in uids])


def _lr(Xtr, ytr, Xte, C, class_weight=None):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=5000, class_weight=class_weight)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))


def budget_uids(phase, budget, seed):
    """A random subset of the fitting rows: `budget` issues per (project, label)."""
    pool = pool_cached()
    trm, _ = masks(phase)
    sub = pool[trm].groupby(["proj", "label"]).sample(n=budget, random_state=seed)
    return np.sort(sub.uid.to_numpy())


def probe(phase, feat, layer, C=0.01, scope="PA", train_feat=None, kind="h", budget=None, seed=0):
    """LR read-out of an LLM state: kind "h" = answer-position (last-token) state,
    "m" = mean over the issue's tokens. train_feat: optional different feature file
    for the fitting rows; defaults to `feat`. budget: fit on only `budget` issues per
    (project, label) drawn with `seed` (low-label analysis)."""
    pool = pool_cached()
    trm, qm = masks(phase)
    tr_uids, q_uids = pool.uid.to_numpy()[trm], pool.uid.to_numpy()[qm]
    if budget is not None:
        tr_uids = budget_uids(phase, budget, seed)
    zq = load_feats(feat)
    zt = load_feats(train_feat or feat)
    key = f"{kind}_L{layer}"
    Xtr = zt[key][_rows(zt, tr_uids)].astype(np.float32)
    Xq = zq[key][_rows(zq, q_uids)].astype(np.float32)
    ytr = labels_of(tr_uids)
    if scope == "PA":
        return _lr(Xtr, ytr, Xq, C)
    ptr, pq = proj_of(tr_uids), proj_of(q_uids)
    if scope == "AUG":
        return _lr_aug(Xtr, ytr, ptr, Xq, pq, C)
    P = np.zeros((len(q_uids), 3))
    for p in np.unique(pq):
        a, b = ptr == p, pq == p
        P[b] = _lr(Xtr[a], ytr[a], Xq[b], C)
    return P


def _lr_aug(Xtr, ytr, ptr, Xq, pq, C, proj_weight=1.0):
    """Feature augmentation (Daume III 2007): [x, x * 1{project = p}] with one pooled LR,
    i.e. shared weights plus L2-shrunk project-specific deviations."""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xq)
    projs = sorted(set(ptr) | set(pq))

    def aug(X, pr):
        blocks = [X] + [X * (proj_weight * (pr == p))[:, None] for p in projs]
        return np.concatenate(blocks, 1).astype(np.float32)

    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(aug(A, ptr), ytr)
    return clf.predict_proba(aug(B, pq))


def label_scores(phase, feat, calibrate="none", scope="PA", train_feat=None):
    """The LLM's own label distribution (constrained decoding).
    calibrate: none | lr (multinomial LR on the 3 log-probs, fit on training rows)."""
    pool = pool_cached()
    trm, qm = masks(phase)
    tr_uids, q_uids = pool.uid.to_numpy()[trm], pool.uid.to_numpy()[qm]
    zq = load_feats(feat)
    lq = zq["logp"][_rows(zq, q_uids)]
    if calibrate == "none":
        e = np.exp(lq - lq.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    zt = load_feats(train_feat or feat)
    lt = zt["logp"][_rows(zt, tr_uids)]
    ytr = labels_of(tr_uids)
    if scope == "PA":
        return _lr(lt, ytr, lq, 1.0)
    P = np.zeros((len(q_uids), 3))
    ptr, pq = proj_of(tr_uids), proj_of(q_uids)
    for p in np.unique(pq):
        a, b = ptr == p, pq == p
        P[b] = _lr(lt[a], ytr[a], lq[b], 1.0)
    return P


def probe_ensemble(phase, feat, layers, C=0.003, scope="AUG", kind="h", budget=None, seed=0):
    """Average of per-layer read-out probabilities (robust to the choice of layer)."""
    return np.mean([probe(phase, feat, l, C, scope, kind=kind, budget=budget, seed=seed) for l in layers], 0)


# --------------------------------------------------------------------- lexical
def tfidf(phase, C=16.0):
    """TF-IDF (word 1-2 + char_wb 3-5) multinomial LR, pooled over projects (PA)."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    pool = pool_cached()
    trm, qm = masks(phase)
    txt = lambda d: [f"Title: {t}\nBody: {b}"[:20000] for t, b in zip(d.title, d.body)]
    tr, q = pool[trm], pool[qm]
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200000, sublinear_tf=True)
    A = hstack([w.fit_transform(txt(tr)), c.fit_transform(txt(tr))]).tocsr()
    B = hstack([w.transform(txt(q)), c.transform(txt(q))]).tocsr()
    clf = LogisticRegression(C=C, max_iter=3000).fit(A, tr.label.map(LAB2ID).to_numpy())
    return clf.predict_proba(B)


# --------------------------------------------------------------------- kNN votes
def knn_vote(phase, setting="PS", variant="center", k=15, power=1.0):
    """Similarity-weighted vote distribution over the top-k neighbours (normalised)."""
    z = np.load(FEATS / f"nb_{setting}_{variant}_{phase}.npz")
    q_uids = query_uids(phase)
    rows = _rows({"uids": z["uids"]}, q_uids)
    nb, sim = z["nb"][rows, :k], z["sim"][rows, :k]
    lab = pool_cached().set_index("uid")["label"].map(LAB2ID)
    L = lab.loc[nb.ravel()].to_numpy().reshape(nb.shape)
    w = np.clip(sim, 1e-6, None) ** power
    S = np.stack([(w * (L == c)).sum(1) for c in range(3)], 1) + 1e-3
    return S / S.sum(1, keepdims=True)


# --------------------------------------------------------------------- SetFit
def setfit(phase, body="Collab-uniba_github-issues-mpnet-st-e10"):
    """SetFit-PS probabilities. dev: models trained on inner (run_setfit_dev.sh);
    test: the paper's archival SetFit-PS runs (trained on full train)."""
    import json
    q_uids = query_uids(phase)
    pool = pool_cached()
    out = {}
    if phase == "dev":
        for p in sorted(pool.proj.unique()):
            dev = pd.read_csv(SPLITS / "ps" / p / "dev.csv", usecols=["uid"])
            pr = pd.read_csv(EXP / "setfit_dev" / body / p / "predictions" / "preds_setfit.csv")
            for u, r in zip(dev["uid"], pr["raw_output"]):
                d = json.loads(r)
                out[int(u)] = [d[l] for l in LABELS]
    else:
        for p in sorted(pool.proj.unique()):
            te = pool[(pool.split == "test") & (pool.proj == p)]
            pr = pd.read_csv(RES / "project_specific" / p / body / "setfit" / "predictions" / "preds_setfit.csv")
            assert len(pr) == len(te)
            for u, r in zip(te["uid"], pr["raw_output"]):
                d = json.loads(r)
                out[int(u)] = [d[l] for l in LABELS]
    return np.array([out[int(u)] for u in q_uids])

#!/usr/bin/env python3
"""Dev-phase evaluation of read-outs over LLM decision states (CPU only).

Fits on role == inner, scores on role == dev (never touches test).

Read-outs
  gen-free argmax   argmax of the three label log-probs (constrained decoding)
  cal3              multinomial LR on the 3 label log-probs (bias/scale calibration)
  probe L{l} C      multinomial LR on the standardised hidden state of layer l
  knn L{l}          cosine kNN vote (k=15, similarity-weighted) in that space

Usage:
  venv/bin/python scripts/experiments/rag_next/probe_eval.py features/q7_k0
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import FEATS, LAB2ID, load_pool, macro_f1, per_class_f1

warnings.filterwarnings("ignore")


def load(name):
    z = np.load(FEATS / f"{name}.npz" if not str(name).endswith(".npz") else name)
    return z


def lr_fit_predict(Xtr, ytr, Xte, C, return_proba=False):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit(sc.transform(Xtr), ytr)
    P = clf.predict_proba(sc.transform(Xte))
    return P if return_proba else P.argmax(1)


def knn_vote(Xtr, ytr, Xte, k=15):
    A = Xtr - Xtr.mean(0)
    B = Xte - Xtr.mean(0)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
    S = B @ A.T
    idx = np.argsort(-S, 1)[:, :k]
    sc = np.zeros((len(B), 3))
    for c in range(3):
        sc[:, c] = (np.take_along_axis(S, idx, 1) * (ytr[idx] == c)).sum(1)
    return sc.argmax(1)


def report(tag, y, p):
    f = per_class_f1(y, p)
    print(f"  {tag:34s} macroF1={macro_f1(y, p):.4f}  bug={f[0]:.3f} feat={f[1]:.3f} q={f[2]:.3f}  "
          f"pred_bug={np.mean(p == 0):.3f}")


def main():
    name = sys.argv[1]
    Cs = [float(c) for c in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0.001", "0.003", "0.01", "0.03"])]
    z = load(name)
    pool = load_pool().set_index("uid")
    uids = z["uids"]
    role = pool.loc[uids, "role"].to_numpy()
    proj = pool.loc[uids, "proj"].to_numpy()
    y = pool.loc[uids, "label"].map(LAB2ID).to_numpy()
    tr, dv = role == "inner", role == "dev"
    print(f"{name}: n_inner={tr.sum()} n_dev={dv.sum()}")

    lp = z["logp"]
    report("label-scoring argmax (dev)", y[dv], lp[dv].argmax(1))
    report("cal3 PA (dev)", y[dv], lr_fit_predict(lp[tr], y[tr], lp[dv], 1.0))
    # per-project calibration
    p = np.zeros(dv.sum(), int)
    dv_idx = np.where(dv)[0]
    for pr in np.unique(proj):
        a, b = tr & (proj == pr), dv & (proj == pr)
        p[np.isin(dv_idx, np.where(b)[0])] = lr_fit_predict(lp[a], y[a], lp[b], 1.0)
    report("cal3 PS (dev)", y[dv], p)

    layers = sorted(int(k[3:]) for k in z.files if k.startswith("h_L"))
    for l in layers:
        X = z[f"h_L{l}"].astype(np.float32)
        for C in Cs:
            report(f"probe PA L{l} C={C}", y[dv], lr_fit_predict(X[tr], y[tr], X[dv], C))
        report(f"knn15 PA L{l}", y[dv], knn_vote(X[tr], y[tr], X[dv]))
    # PS probes at the best-looking layers only (per-project LR)
    for l in layers[-3:]:
        X = z[f"h_L{l}"].astype(np.float32)
        for C in Cs:
            p = np.zeros(dv.sum(), int)
            for pr in np.unique(proj):
                a, b = tr & (proj == pr), dv & (proj == pr)
                p[np.isin(dv_idx, np.where(b)[0])] = lr_fit_predict(X[a], y[a], X[b], C)
            report(f"probe PS L{l} C={C}", y[dv], p)


if __name__ == "__main__":
    main()

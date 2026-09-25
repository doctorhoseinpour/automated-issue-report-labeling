#!/usr/bin/env python3
"""Dev-phase sample-efficiency curve: labeled issues per (project, label) in
{5, 10, 20, 40, 70 (= all inner)}, 3 random draws each (the 70 budget is deterministic).

Methods (all fit/indexed on the sampled inner subset, scored on the full dev set):
  read-out   AUG probe, layer average, C = 0.003 (the dev-chosen default)
  read-out   PA probe on the final-layer state
  state kNN  similarity vote over decision-state neighbours (final layer, PS, k = 9)
  TF-IDF     PA LR (C = 16)

  python sample_efficiency.py q7_k0
"""
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import components as C
from common import LAB2ID, macro_f1
from dev_votag import vote

warnings.filterwarnings("ignore")


def main():
    feat = sys.argv[1]
    z = C.load_feats(feat)
    layers = sorted(int(k[3:]) for k in z if k.startswith("h_L"))
    avg_layers = [l for l in layers if l >= layers[-1] * 0.6]
    last = layers[-1]
    pool = C.pool_cached()
    inner = pool[pool.role == "inner"]
    dev = pool[pool.role == "dev"]
    ydv = dev.label.map(LAB2ID).to_numpy()
    pdv = dev.proj.to_numpy()
    rows_dv = C._rows(z, dev.uid.to_numpy())
    for budget in [5, 10, 20, 40, 70]:
        res = {k: [] for k in ["readout_AUG", "readout_PA_last", "state_kNN", "tfidf"]}
        for seed in ([0] if budget == 70 else [0, 1, 2]):
            sub = inner.groupby(["proj", "label"]).sample(n=budget, random_state=seed)
            ytr = sub.label.map(LAB2ID).to_numpy()
            ptr = sub.proj.to_numpy()
            rows_tr = C._rows(z, sub.uid.to_numpy())
            P = []
            for l in avg_layers:
                X = z[f"h_L{l}"].astype(np.float32)
                P.append(C._lr_aug(X[rows_tr], ytr, ptr, X[rows_dv], pdv, 0.003))
            res["readout_AUG"].append(macro_f1(ydv, np.mean(P, 0).argmax(1)))
            X = z[f"h_L{last}"].astype(np.float32)
            res["readout_PA_last"].append(macro_f1(ydv, C._lr(X[rows_tr], ytr, X[rows_dv], 0.003).argmax(1)))
            A, B = X[rows_tr], X[rows_dv]
            mu, sd = A.mean(0), A.std(0) + 1e-6
            A = (A - mu) / sd; B = (B - mu) / sd
            A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True)
            k = 9
            NB = np.zeros((len(B), k), int); S = np.zeros((len(B), k))
            for p in np.unique(pdv):
                qi, ii = np.where(pdv == p)[0], np.where(ptr == p)[0]
                s = B[qi] @ A[ii].T
                o = np.argsort(-s, 1)[:, :k]
                NB[qi] = ii[o]; S[qi] = np.take_along_axis(s, o, 1)
            res["state_kNN"].append(macro_f1(ydv, vote(ytr[NB], S, k)))
            # TF-IDF on the subset
            from scipy.sparse import hstack
            from sklearn.feature_extraction.text import TfidfVectorizer
            txt = lambda d: [f"Title: {t}\nBody: {b}"[:20000] for t, b in zip(d.title, d.body)]
            w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
            c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200000, sublinear_tf=True)
            Atf = hstack([w.fit_transform(txt(sub)), c.fit_transform(txt(sub))]).tocsr()
            Btf = hstack([w.transform(txt(dev)), c.transform(txt(dev))]).tocsr()
            res["tfidf"].append(macro_f1(ydv, LogisticRegression(C=16, max_iter=3000).fit(Atf, ytr).predict(Btf)))
        print(f"{feat} budget={budget:3d}/label/project (n={budget * 33 if budget < 70 else len(inner)}): " +
              "  ".join(f"{k}={np.mean(v):.3f}±{np.std(v):.3f}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()

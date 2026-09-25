#!/usr/bin/env python3
"""Dev scores for one model size: the LLM's own constrained label, PA probes per layer,
and the pre-set default read-out (AUG, layers at depth fractions 0.625-1.0, C = 0.003)
with two neighbouring C values; plus the decision-state kNN vote (final layer, PS, k=9).
  python dev_size.py q14_k0
"""
import sys

import numpy as np

import components as C
from common import macro_f1, per_class_f1
from dev_votag import vote

uids = C.query_uids("dev"); y = C.labels_of(uids)


def rep(t, p):
    f = per_class_f1(y, p)
    print(f"  {t:46s} F1={macro_f1(y, p):.4f} bug={f[0]:.3f} feat={f[1]:.3f} q={f[2]:.3f} "
          f"q->bug={np.mean(p[y == 2] == 0):.3f}", flush=True)


feat = sys.argv[1]
z = C.load_feats(feat)
L = sorted(int(k[3:]) for k in z if k.startswith("h_L"))
print(feat, "layers", L)
rep("label argmax (constrained decoding)", C.label_scores("dev", feat).argmax(1))
rep("label LR-calibrated PS", C.label_scores("dev", feat, "lr", "PS").argmax(1))
for l in L:
    rep(f"probe PA L{l} C=0.003", C.probe("dev", feat, l, 0.003, "PA").argmax(1))
for c in [0.001, 0.003, 0.01]:
    tag = "DEFAULT " if c == 0.003 else ""
    rep(f"{tag}AUG avg {L[1:]} C={c}", C.probe_ensemble("dev", feat, L[1:], c, "AUG").argmax(1))
# decision-state kNN (final layer, PS, k = 9), dev queries vs inner index
pool = C.pool_cached()
tr = (pool.role == "inner").to_numpy(); dv = (pool.role == "dev").to_numpy()
rows_tr = C._rows(z, pool.uid.to_numpy()[tr]); rows_dv = C._rows(z, pool.uid.to_numpy()[dv])
ytr = C.labels_of(pool.uid.to_numpy()[tr]); ptr = pool.proj.to_numpy()[tr]; pdv = pool.proj.to_numpy()[dv]
X = z[f"h_L{L[-1]}"].astype(np.float32)
A, B = X[rows_tr], X[rows_dv]
mu, sd = A.mean(0), A.std(0) + 1e-6
A = (A - mu) / sd; B = (B - mu) / sd
A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True)
NB = np.zeros((len(B), 15), int); S = np.zeros((len(B), 15))
for p in np.unique(pdv):
    qi, ii = np.where(pdv == p)[0], np.where(ptr == p)[0]
    s = B[qi] @ A[ii].T; o = np.argsort(-s, 1)[:, :15]
    NB[qi] = ii[o]; S[qi] = np.take_along_axis(s, o, 1)
for k in [9, 15]:
    rep(f"decision-state kNN L{L[-1]} PS @{k}", vote(ytr[NB], S, k))

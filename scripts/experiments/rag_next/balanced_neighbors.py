#!/usr/bin/env python3
"""Derive class-balanced demonstration lists from a ranked neighbour file: for each
query keep the top-m most similar issues of EACH label (from the top-50 pool), then
order the 3m by descending similarity. Removes hubness-driven bug over-representation
and shows same-topic contrasts between labels.
  python balanced_neighbors.py nb_PS_raw_dev_causal 3  -> nb_PS_raw_dev_causal_bal3.npz
"""
import sys

import numpy as np

from common import FEATS, LAB2ID, load_pool

name, m = sys.argv[1], int(sys.argv[2])
z = np.load(FEATS / f"{name}.npz")
lab = load_pool().set_index("uid")["label"].map(LAB2ID)
NB, SIM = z["nb"], z["sim"]
out_nb = -np.ones((len(NB), 3 * m), dtype=np.int64)
out_sim = np.zeros((len(NB), 3 * m), dtype=np.float32)
short = 0
for i in range(len(NB)):
    keep = []
    cnt = [0, 0, 0]
    for v, s in zip(NB[i], SIM[i]):
        if v < 0:
            continue
        c = lab.loc[v]
        if cnt[c] < m:
            keep.append((s, v)); cnt[c] += 1
        if min(cnt) >= m:
            break
    short += min(cnt) < m
    keep.sort(key=lambda t: -t[0])
    for j, (s, v) in enumerate(keep):
        out_nb[i, j], out_sim[i, j] = v, s
np.savez(FEATS / f"{name}_bal{m}.npz", uids=z["uids"], nb=out_nb, sim=out_sim)
print(f"wrote {name}_bal{m}.npz; queries with <m of some label in top-50: {short}")

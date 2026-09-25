#!/usr/bin/env python3
"""Task-conditioned retrieval: nearest neighbours in the LLM's zero-shot decision-state
space (hidden state at the answer position of the RAGTAG zero-shot prompt).

Training-free and label-free: states are standardised with the mean/std of the index
issues' states (no labels), then cosine similarity. PS restricts the index to the
query's project, as in the paper.
  phase dev : queries = dev,  index = inner
  phase test: queries = test, index = train
Writes features/nb_{setting}_{tag}_{phase}.npz (uids, nb, sim; K = 50), the same
format as neighbors.py, so llm_features.py / components.knn_vote can use it.

  python state_neighbors.py q7_k0 28 q7L28
"""
import sys

import numpy as np

from common import FEATS, load_pool

K = 50


def main():
    feat, layer, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    z = np.load(FEATS / f"{feat}.npz")
    pos = {int(u): i for i, u in enumerate(z["uids"])}
    X = z[f"h_L{layer}"].astype(np.float32)
    pool = load_pool()
    for phase, qrole, imask in [("dev", "dev", (pool.role == "inner").to_numpy()),
                                ("test", "test", (pool.split == "train").to_numpy())]:
        qmask = (pool.role == qrole).to_numpy()
        iu, qu = pool.uid.to_numpy()[imask], pool.uid.to_numpy()[qmask]
        ip, qp = pool.proj.to_numpy()[imask], pool.proj.to_numpy()[qmask]
        A = X[[pos[u] for u in iu]]
        B = X[[pos[u] for u in qu]]
        mu, sd = A.mean(0), A.std(0) + 1e-6
        A = (A - mu) / sd
        B = (B - mu) / sd
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        B /= np.linalg.norm(B, axis=1, keepdims=True)
        for setting in ["PS", "PA"]:
            out = FEATS / f"nb_{setting}_{tag}_{phase}.npz"
            if out.exists():
                continue
            NB = np.zeros((len(qu), K), np.int64)
            SIM = np.zeros((len(qu), K), np.float32)
            for p in np.unique(qp):
                qi = np.where(qp == p)[0]
                ii = np.where(ip == p)[0] if setting == "PS" else np.arange(len(iu))
                S = B[qi] @ A[ii].T
                o = np.argsort(-S, axis=1)[:, :K]
                NB[qi] = iu[ii][o]
                SIM[qi] = np.take_along_axis(S, o, 1)
            np.savez(out, uids=qu, nb=NB, sim=SIM)
            print("wrote", out.name, NB.shape)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""VOTAG (similarity-weighted kNN vote, vtag.py semantics) on dev and test queries
for the raw and centered neighbour files; a cheap reference for dev difficulty."""
import numpy as np

from common import FEATS, LAB2ID, load_pool, macro_f1


def vote(L, S, k):
    L, S = L[:, :k], S[:, :k].astype(np.float64)
    sc = np.stack([(S * (L == c)).sum(1) for c in range(3)], 1)
    mx = sc.max(1, keepdims=True)
    win = np.isclose(sc, mx)
    pred = sc.argmax(1)
    for i in np.where(win.sum(1) > 1)[0]:
        for c in L[i]:
            if win[i, c]:
                pred[i] = c
                break
    return pred


def main():
    pool = load_pool().set_index("uid")
    lab = pool["label"].map(LAB2ID)
    for phase, qrole in [("dev", "dev"), ("test", "test")]:
        for setting in ["PS", "PA"]:
            for variant in ["raw", "center"]:
                z = np.load(FEATS / f"nb_{setting}_{variant}_{phase}.npz")
                m = pool.loc[z["uids"], "role"].to_numpy() == qrole
                L = lab.loc[z["nb"][m].ravel()].to_numpy().reshape(m.sum(), -1)
                y = lab.loc[z["uids"][m]].to_numpy()
                s = " ".join(f"@{k}={macro_f1(y, vote(L, z['sim'][m], k)):.3f}" for k in [3, 6, 9, 12, 15, 20, 30])
                print(f"{phase:4s} {setting} {variant:6s} n={m.sum()} VOTAG {s}")


if __name__ == "__main__":
    main()

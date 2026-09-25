#!/usr/bin/env python3
"""Is leave-one-out retrieval for training queries easier than retrieval for dev/test
queries (temporal locality)? Compare homophily and VOTAG for inner-LOO vs dev queries
(phase dev), and train-LOO vs test queries (phase test). Also a 'causal' LOO that only
allows neighbours created before the query."""
import numpy as np
import pandas as pd

from common import FEATS, LAB2ID, load_pool, macro_f1
from dev_votag import vote

pool = load_pool().set_index("uid")
lab = pool["label"].map(LAB2ID)
ts = pd.to_datetime(pool["created_at"], utc=True)
for phase, qa, qb in [("dev", "inner", "dev"), ("test", "train", "test")]:
    for setting in ["PS"]:
        for variant in ["raw", "center"]:
            z = np.load(FEATS / f"nb_{setting}_{variant}_{phase}.npz")
            roles = pool.loc[z["uids"], "role"].to_numpy()
            splits = pool.loc[z["uids"], "split"].to_numpy()
            for name, m in [(qa, (roles == "inner") if qa == "inner" else (splits == "train")), (qb, roles == qb)]:
                L = lab.loc[z["nb"][m].ravel()].to_numpy().reshape(m.sum(), -1)
                y = lab.loc[z["uids"][m]].to_numpy()
                hom = (L[:, :9] == y[:, None]).mean()
                # time gap to neighbours (days)
                tq = ts.loc[z["uids"][m]].to_numpy()
                tn = ts.loc[z["nb"][m][:, :9].ravel()].to_numpy().reshape(m.sum(), 9)
                gap = np.abs((tn - tq[:, None]) / np.timedelta64(1, "D"))
                print(f"{phase:4s} {variant:6s} queries={name:5s} n={m.sum():4d} hom@9={hom:.3f} "
                      f"VOTAG@9={macro_f1(y, vote(L, z['sim'][m], 9)):.3f} @15={macro_f1(y, vote(L, z['sim'][m], 15)):.3f} "
                      f"median |gap| days={np.median(gap):.0f} sim@1={z['sim'][m][:,0].mean():.3f}")

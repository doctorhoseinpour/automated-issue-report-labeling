#!/usr/bin/env python3
"""Dev error analysis of the default read-out at one size: confusion matrix,
confidence of errors, and a random sample of question->bug errors (titles)."""
import sys

import numpy as np
import pandas as pd

import components as C

feat, layers = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
uids = C.query_uids("dev"); y = C.labels_of(uids)
P = C.probe_ensemble("dev", feat, layers, 0.003, "AUG"); p = P.argmax(1)
L = np.array(["bug", "feature", "question"])
print(pd.crosstab(pd.Series(L[y], name="true"), pd.Series(L[p], name="pred")))
conf = P.max(1)
for lo, hi in [(0, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
    m = (conf >= lo) & (conf < hi)
    print(f"  conf [{lo:.1f},{hi:.1f}): n={m.sum():4d} acc={np.mean(p[m] == y[m]):.3f}")
pool = C.pool_cached().set_index("uid")
err = np.where((y == 2) & (p == 0))[0]
rng = np.random.default_rng(0)
for i in rng.choice(err, size=min(12, len(err)), replace=False):
    r = pool.loc[uids[i]]
    body = r["body"].replace("\n", " ")[:110]
    print(f"  [{r['proj'][:12]:12s}] p(bug)={P[i,0]:.2f} | {r['title'][:80]} || {body}")

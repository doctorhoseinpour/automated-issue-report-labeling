#!/usr/bin/env python3
"""Cold start for a new project (dev): leave-one-project-out read-out. The head is fit on
the other 10 projects' inner issues (PA, layer average, C = 0.003) and scored on the
held-out project's dev issues; pooled over the 11 held-out projects. Compared with the
LLM's own zero-shot label (constrained decoding) and with the in-project read-out."""
import sys

import numpy as np

import components as C
from common import macro_f1, per_class_f1

DEPTH = {"q3_k0": [22, 27, 32, 36], "q7_k0": [18, 21, 24, 28], "q14_k0": [30, 36, 42, 48],
         "q32_k0": [40, 48, 56, 64]}
feat = sys.argv[1]
z = C.load_feats(feat)
pool = C.pool_cached()
tr, dv = pool[pool.role == "inner"], pool[pool.role == "dev"]
ytr, ydv = C.labels_of(tr.uid.to_numpy()), C.labels_of(dv.uid.to_numpy())
rt, rd = C._rows(z, tr.uid.to_numpy()), C._rows(z, dv.uid.to_numpy())
P = np.zeros((len(dv), 3))
for p in sorted(pool.proj.unique()):
    a, b = (tr.proj != p).to_numpy(), (dv.proj == p).to_numpy()
    P[b] = np.mean([C._lr(z[f"h_L{l}"][rt][a].astype(np.float32), ytr[a],
                          z[f"h_L{l}"][rd][b].astype(np.float32), 0.003) for l in DEPTH[feat]], 0)
zs = z["logp"][rd].argmax(1)
full = C.probe_ensemble("dev", feat, DEPTH[feat], 0.003, "AUG").argmax(1)
for name, p in [("zero-shot own label", zs), ("LOPO read-out (no target labels)", P.argmax(1)),
                ("in-project read-out (AUG, all projects)", full)]:
    f = per_class_f1(ydv, p)
    print(f"{feat} {name:42s} F1={macro_f1(ydv, p):.4f} bug={f[0]:.3f} feat={f[1]:.3f} q={f[2]:.3f}")
print("per held-out project (LOPO):", {p: round(macro_f1(ydv[(dv.proj == p).to_numpy()], P.argmax(1)[(dv.proj == p).to_numpy()]), 3)
                                       for p in sorted(pool.proj.unique())})

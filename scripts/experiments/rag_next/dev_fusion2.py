#!/usr/bin/env python3
"""Dev fusion across model sizes and components (stacker cross-fitted inside dev)."""
import itertools
import sys

import numpy as np

import components as C
from common import macro_f1
from fusion import cv_stack, product_of_experts

uids = C.query_uids("dev"); y = C.labels_of(uids)
strata = np.array([f"{p}|{l}" for p, l in zip(C.proj_of(uids), y)])
DEPTH = {"q3_k0": [22, 27, 32, 36], "q7_k0": [18, 21, 24, 28], "q14_k0": [30, 36, 42, 48],
         "q32_k0": [40, 48, 56, 64]}
sizes = sys.argv[1].split(",")
comps = {}
for f in sizes:
    comps[f"ro{f.split('_')[0][1:]}"] = C.probe_ensemble("dev", f, DEPTH[f], 0.003, "AUG")
comps["tfidf"] = C.tfidf("dev")
comps["knn"] = C.knn_vote("dev", "PS", "raw", 15)
if "--setfit" in sys.argv:
    comps["setfit"] = C.setfit("dev")
for n, P in comps.items():
    print(f"  single {n:10s} F1={macro_f1(y, P.argmax(1)):.4f}")
names = list(comps)
for r in range(2, len(names) + 1):
    for sub in itertools.combinations(names, r):
        mu, sd, _ = cv_stack([comps[n] for n in sub], y, strata)
        poe = macro_f1(y, product_of_experts([comps[n] for n in sub]))
        print(f"  stack {'+'.join(sub):44s} CV F1={mu:.4f}±{sd:.4f} | PoE {poe:.4f}", flush=True)

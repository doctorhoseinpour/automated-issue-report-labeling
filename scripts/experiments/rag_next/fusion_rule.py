#!/usr/bin/env python3
"""Apply the pre-registered fusion rule (notebook section 5) on dev and write configs/fusion.json.

  S*  = the Qwen size whose default read-out has the best dev macro F1
  F   = LR stacker over [read-out S*, TF-IDF LR, MiniLM kNN PS@15]
        + SetFit-PS probabilities only if the cross-fitted dev stacker gains >= +0.005
Also prints SetFit-PS on dev (the dev reference bar).
"""
import json
from pathlib import Path

import numpy as np

import components as C
from common import FEATS, macro_f1, per_class_f1
from fusion import cv_stack

DEPTH = {"3B": ("q3_k0", [22, 27, 32, 36]), "7B": ("q7_k0", [18, 21, 24, 28]),
         "14B": ("q14_k0", [30, 36, 42, 48]), "32B": ("q32_k0", [40, 48, 56, 64])}
uids = C.query_uids("dev"); y = C.labels_of(uids)
strata = np.array([f"{p}|{l}" for p, l in zip(C.proj_of(uids), y)])


def rep(t, P):
    p = P.argmax(1); f = per_class_f1(y, p)
    print(f"  {t:36s} F1={macro_f1(y, p):.4f} bug={f[0]:.3f} feat={f[1]:.3f} q={f[2]:.3f} "
          f"q->bug={np.mean(p[y == 2] == 0):.3f}", flush=True)
    return macro_f1(y, p)


sf = C.setfit("dev")
rep("SetFit-PS (issues) dev reference", sf)
ro, score = {}, {}
for size, (feat, layers) in DEPTH.items():
    if not (FEATS / f"{feat}.npz").exists():
        print(f"  {size}: features missing, skipped")
        continue
    ro[size] = C.probe_ensemble("dev", feat, layers, 0.003, "AUG")
    score[size] = rep(f"read-out {size}", ro[size])
s_star = max(score, key=score.get)
print(f"S* = {s_star}")
tf = C.tfidf("dev"); kn = C.knn_vote("dev", "PS", "raw", 15)
base, _, _ = cv_stack([ro[s_star], tf, kn], y, strata)
with_sf, _, _ = cv_stack([ro[s_star], tf, kn, sf], y, strata)
only_sf, _, _ = cv_stack([ro[s_star], sf], y, strata)
print(f"stack R{s_star}+tfidf+knn         CV F1={base:.4f}")
print(f"stack R{s_star}+tfidf+knn+setfit  CV F1={with_sf:.4f}  (gain {with_sf - base:+.4f})")
print(f"stack R{s_star}+setfit            CV F1={only_sf:.4f}")
use_sf = (with_sf - base) >= 0.005
feat, layers = DEPTH[s_star]
comps = [{"type": "probe", "feat": feat, "layers": layers, "C": 0.003, "scope": "AUG"},
         {"type": "tfidf", "C": 16.0},
         {"type": "knn", "setting": "PS", "variant": "raw", "k": 15}]
if use_sf:
    comps.append({"type": "setfit"})
cfg = {"name": "fusion", "rule": "pre-registered section 5", "s_star": s_star, "setfit_included": use_sf,
       "dev_cv": {"base": base, "with_setfit": with_sf}, "components": comps, "stacker_C": 1.0}
out = Path(__file__).resolve().parent / "configs" / "fusion.json"
json.dump(cfg, open(out, "w"), indent=1)
print("wrote", out, "setfit included:", use_sf)

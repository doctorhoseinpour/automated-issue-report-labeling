#!/usr/bin/env python3
"""Dev: constrained-decoding scores of RAG prompts (RAGTAG / BRAGTAG / decision-state
demos) and their fusion with the read-out."""
import numpy as np

import components as C
from common import macro_f1, per_class_f1
from fusion import cv_stack

uids = C.query_uids("dev"); y = C.labels_of(uids)
strata = np.array([f"{p}|{l}" for p, l in zip(C.proj_of(uids), y)])
for f in ["q7_k12_dev", "q7_k12b3_dev", "q7_k12_state_dev"]:
    P = C.label_scores("dev", f); p = P.argmax(1); z = C.load_feats(f)
    print(f"{f:20s} constrained F1={macro_f1(y, p):.4f} per-class={[round(v, 3) for v in per_class_f1(y, p)]} "
          f"q->bug={np.mean(p[y == 2] == 0):.3f} predbug={np.mean(p == 0):.3f} demos={z['n_demos'].mean():.1f}")
kst = C.knn_vote("dev", "PS", "q7L28", 12)
print("state kNN PS@12 vote F1=%.4f" % macro_f1(y, kst.argmax(1)))
sr = C.label_scores("dev", "q7_k12_state_dev")
print("agree(state-RAGTAG, state-vote@12)=%.3f" % np.mean(sr.argmax(1) == kst.argmax(1)))
pr = C.probe_ensemble("dev", "q7_k0", [18, 21, 24, 28], 0.003, "AUG")
tf = C.tfidf("dev")
for name, comps in [("stateRAG+stateknn", [sr, kst]), ("probe+stateRAG", [pr, sr]),
                    ("probe+tfidf+stateRAG", [pr, tf, sr])]:
    mu, sd, _ = cv_stack(comps, y, strata)
    print(f"stack {name:24s} CV F1={mu:.4f}±{sd:.4f}")

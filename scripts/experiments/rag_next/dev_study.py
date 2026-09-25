#!/usr/bin/env python3
"""Dev-phase study driver: score components and fusions on dev (fit on inner).

  venv/bin/python scripts/experiments/rag_next/dev_study.py probe q7_k0
  venv/bin/python scripts/experiments/rag_next/dev_study.py fusion q7_k0:L21:0.01:PA [--rag q7_k12b3] [--setfit]
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

import components as C
from common import macro_f1, per_class_f1
from fusion import cv_stack, product_of_experts


def rep(tag, y, p):
    f = per_class_f1(y, p)
    print(f"  {tag:46s} F1={macro_f1(y, p):.4f}  bug={f[0]:.3f} feat={f[1]:.3f} q={f[2]:.3f}  "
          f"q->bug={np.mean(p[y == 2] == 0):.3f} predbug={np.mean(p == 0):.3f}", flush=True)


def probe_grid(feat, Cs=(0.001, 0.003, 0.01, 0.03), kinds=("h", "m"), top=2):
    """Stage 1: PA probes for every (kind, layer, C). Stage 2: PS and AUG scopes for the
    `top` best (kind, layer) states of stage 1."""
    y = C.labels_of(C.query_uids("dev"))
    z = C.load_feats(feat)
    layers = sorted(int(k[3:]) for k in z if k.startswith("h_L"))
    print(f"== {feat}: label-scoring and probes (fit inner -> dev, n={len(y)})")
    rep("label argmax (constrained decoding)", y, C.label_scores("dev", feat).argmax(1))
    rep("label LR-calibrated PA", y, C.label_scores("dev", feat, "lr", "PA").argmax(1))
    rep("label LR-calibrated PS", y, C.label_scores("dev", feat, "lr", "PS").argmax(1))
    best = {}
    for kind in kinds:
        for l in layers:
            if f"{kind}_L{l}" not in z:
                continue
            for c in Cs:
                p = C.probe("dev", feat, l, c, "PA", kind=kind).argmax(1)
                rep(f"probe {kind} L{l} PA C={c}", y, p)
                best[(kind, l)] = max(best.get((kind, l), 0), macro_f1(y, p))
    for (kind, l), _ in sorted(best.items(), key=lambda t: -t[1])[:top]:
        for scope in ("PS", "AUG"):
            for c in Cs:
                p = C.probe("dev", feat, l, c, scope, kind=kind).argmax(1)
                rep(f"probe {kind} L{l} {scope} C={c}", y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "fusion"])
    ap.add_argument("spec")
    ap.add_argument("--rag", default=None, help="feature file with RAG-prompt label scores (dev queries)")
    ap.add_argument("--setfit", action="store_true")
    ap.add_argument("--tfidf", action="store_true")
    ap.add_argument("--max_combo", type=int, default=6)
    ap.add_argument("--knn", default="PS:center:15")
    args = ap.parse_args()
    if args.mode == "probe":
        probe_grid(args.spec)
        return
    feat, layer, c, scope, *rest = args.spec.split(":")
    kind = rest[0] if rest else "h"
    uids = C.query_uids("dev")
    y = C.labels_of(uids)
    strata = np.array([f"{p}|{l}" for p, l in zip(C.proj_of(uids), y)])
    if "+" in layer:  # layer ensemble, e.g. 18+21+24+28
        comps = {"probe": C.probe_ensemble("dev", feat, [int(x) for x in layer.split("+")], float(c), scope, kind)}
    else:
        comps = {"probe": C.probe("dev", feat, int(layer), float(c), scope, kind=kind)}
    if args.tfidf:
        comps["tfidf"] = C.tfidf("dev")
    s, v, k = args.knn.split(":")
    comps["knn"] = C.knn_vote("dev", s, v, int(k))
    comps["zs"] = C.label_scores("dev", feat)
    if args.rag:
        for r in args.rag.split(","):
            comps[f"rag:{r}"] = C.label_scores("dev", r)
    if args.setfit:
        comps["setfit"] = C.setfit("dev")
    print(f"== fusion on dev (n={len(y)}), stacker CV 5x5 inside dev")
    for name, P in comps.items():
        rep(f"single {name}", y, P.argmax(1))
    names = list(comps)
    for r in range(2, min(len(names), args.max_combo) + 1):
        for sub in itertools.combinations(names, r):
            mu, sd, oof = cv_stack([comps[n] for n in sub], y, strata)
            poe = product_of_experts([comps[n] for n in sub])
            print(f"  stack {'+'.join(sub):52s} CV F1={mu:.4f}±{sd:.4f} | PoE F1={macro_f1(y, poe):.4f}", flush=True)


if __name__ == "__main__":
    main()

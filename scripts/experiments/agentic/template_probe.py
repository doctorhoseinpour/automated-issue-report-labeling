#!/usr/bin/env python3
"""How much of the label is carried by a project's issue-template boilerplate? (dev only, CPU)

Per project, fit on role == inner and score role == dev:
  template-only   LR on the presence of "boilerplate lines" (normalised lines that occur
                  verbatim in >= 5% of the project's inner issues) plus an empty-body flag
  tfidf-LR        LR on word 1-2-gram TF-IDF of title + body (a cheap classical reference)

Usage (lab machine): venv/bin/python scripts/experiments/agentic/template_probe.py
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS, load_pool  # noqa: E402


def norm(line):
    return re.sub(r"\s+", " ", line.strip().lower())


def lines(body):
    return set(n for n in (norm(x) for x in body.split("\n")) if 3 <= len(n) <= 120)


def main():
    p = load_pool()
    ys, pt, pf = [], [], []
    for proj, g in p[p.role.isin(["inner", "dev"])].groupby("proj"):
        tr, dv = g[g.role == "inner"], g[g.role == "dev"]
        cnt = collections.Counter(l for b in tr.body for l in lines(b))
        boiler = [l for l, c in cnt.items() if c >= 0.05 * len(tr)]
        idx = {l: i for i, l in enumerate(boiler)}

        def feat(df):
            X = np.zeros((len(df), len(boiler) + 1))
            for r, b in enumerate(df.body):
                for l in lines(b):
                    if l in idx:
                        X[r, idx[l]] = 1
                X[r, -1] = float(b.strip() == "")
            return X

        a = LogisticRegression(C=1.0, max_iter=3000).fit(feat(tr), tr.label).predict(feat(dv))
        v = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=50000)
        b = LogisticRegression(C=10, max_iter=3000).fit(v.fit_transform(tr.title + " \n " + tr.body), tr.label) \
            .predict(v.transform(dv.title + " \n " + dv.body))
        ys += list(dv.label); pt += list(a); pf += list(b)
        print(f"{proj:24s} boilerplate lines {len(boiler):3d}  template-only F1 "
              f"{f1_score(dv.label, a, labels=LABELS, average='macro'):.3f}   tfidf-LR F1 "
              f"{f1_score(dv.label, b, labels=LABELS, average='macro'):.3f}")
    print("pooled template-only macro F1", round(f1_score(ys, pt, labels=LABELS, average="macro"), 4),
          "| tfidf-LR", round(f1_score(ys, pf, labels=LABELS, average="macro"), 4))


if __name__ == "__main__":
    main()

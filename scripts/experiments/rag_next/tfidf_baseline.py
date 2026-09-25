#!/usr/bin/env python3
"""Lexical reference: TF-IDF (word 1-2gram + char 3-5gram) + logistic regression,
PS and PA, fit on inner -> dev (dev phase only). Measures how much of the gap is
surface-form convention (templates, keywords)."""
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import LAB2ID, issue_text, load_pool, macro_f1, per_class_f1

pool = load_pool()
tr, dv = pool[pool.role == "inner"], pool[pool.role == "dev"]
txt = lambda d: [issue_text(t, b)[:20000] for t, b in zip(d.title, d.body)]
ytr, ydv = tr.label.map(LAB2ID).to_numpy(), dv.label.map(LAB2ID).to_numpy()


def feats(a, b):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200000, sublinear_tf=True)
    Xa = hstack([w.fit_transform(a), c.fit_transform(a)]).tocsr()
    Xb = hstack([w.transform(b), c.transform(b)]).tocsr()
    return Xa, Xb


for C in [1, 4, 16, 64, 256]:
    Xa, Xb = feats(txt(tr), txt(dv))
    p = LogisticRegression(C=C, max_iter=3000).fit(Xa, ytr).predict(Xb)
    f = per_class_f1(ydv, p)
    print(f"TF-IDF LR PA C={C}: dev macroF1={macro_f1(ydv, p):.4f} (bug {f[0]:.3f} feat {f[1]:.3f} q {f[2]:.3f})")
    if C > 16:
        continue  # PS is below PA at every C tried; skip the slow per-project fits
    pp = np.zeros(len(dv), int)
    for proj in sorted(pool.proj.unique()):
        a, b = (tr.proj == proj).to_numpy(), (dv.proj == proj).to_numpy()
        Xa, Xb = feats(txt(tr[a]), txt(dv[b]))
        pp[b] = LogisticRegression(C=C, max_iter=3000).fit(Xa, ytr[a]).predict(Xb)
    f = per_class_f1(ydv, pp)
    print(f"TF-IDF LR PS C={C}: dev macroF1={macro_f1(ydv, pp):.4f} (bug {f[0]:.3f} feat {f[1]:.3f} q {f[2]:.3f})")

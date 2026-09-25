#!/usr/bin/env python3
"""POST-HOC, descriptive robustness of the newllms test results (run after all 18 pre-registered
test evaluations; selects nothing, changes no method). Same analyses as rag_next/posthoc_test.py:
exact McNemar on correctness and a project-cluster bootstrap (resample the 11 projects, 2,000x)
of the pooled macro-F1 difference, for R and K against SetFit-PS and the same model's RAG@K*;
per-project macro F1 of R, RAG@K* and SetFit-PS; question->bug rates.

  python posthoc_test.py
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from nm_common import LABELS, MODELS, NM, RES, RN, macro_f1

L2I = {l: i for i, l in enumerate(LABELS)}
enc = lambda s: np.array([L2I.get(str(x).strip().lower(), -1) for x in s])  # noqa: E731
test = pd.read_csv(RES / "agnostic" / "neighbors" / "test_split.csv", keep_default_na=False)
y = enc(test["labels"])
proj = test["repo"].to_numpy()
projects = np.unique(proj)
idx_by = {p: np.where(proj == p)[0] for p in projects}
m = pd.read_parquet(RN / "headroom" / "master_preds.parquet")
sf = enc(m["setfit_issues_PS"])
P = NM / "test_preds"
pred = lambda f: enc(pd.read_csv(P / f, keep_default_na=False)["predicted_label"])  # noqa: E731


def cluster_ci(a, b, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        ii = np.concatenate([idx_by[p] for p in rng.choice(projects, len(projects), replace=True)])
        d.append(macro_f1(y[ii], a[ii]) - macro_f1(y[ii], b[ii]))
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


rows, pp = [], []
for tag, meta in MODELS.items():
    R, K, G = pred(f"readout_{tag}.csv"), pred(f"stateknn_{tag}.csv"), pred(f"rag_{tag}.csv")
    for cname, c in [("R", R), ("K", K)]:
        for bname, b in [("SetFit-PS", sf), ("RAG@K*", G)]:
            n01 = int(np.sum((c == y) & (b != y))); n10 = int(np.sum((c != y) & (b == y)))
            lo, hi = cluster_ci(c, b)
            rows.append({"model": meta["name"], "cand": cname, "vs": bname, "diff": macro_f1(y, c) - macro_f1(y, b),
                         "mcnemar": f"{n01}/{n10}", "p": binomtest(n01, n01 + n10, 0.5).pvalue,
                         "cluster_lo": lo, "cluster_hi": hi})
    for p in projects:
        k = proj == p
        pp.append({"project": p, "model": tag, "R": macro_f1(y[k], R[k]), "RAG": macro_f1(y[k], G[k]),
                   "SetFit": macro_f1(y[k], sf[k])})
r = pd.DataFrame(rows)
print("== McNemar and project-cluster bootstrap (post-hoc)")
print(r.round(4).to_string(index=False))
pp = pd.DataFrame(pp)
w = pp.pivot_table(index="project", columns="model", values=["R", "RAG"]).round(3)
w["SetFit"] = pp.groupby("project")["SetFit"].first().round(3)
print("\n== per-project macro F1 (test)")
print(w.to_string())
r.to_csv(NM / "test_eval" / "posthoc_robustness.csv", index=False)
pp.to_csv(NM / "test_eval" / "posthoc_per_project.csv", index=False)

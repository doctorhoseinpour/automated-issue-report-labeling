#!/usr/bin/env python3
"""POST-HOC, descriptive analysis of test predictions (run only after all pre-registered
test evaluations; changes no method and selects nothing).

  1. Complementarity of the read-out with SetFit-PS / FT-14B / BRAGTAG-32B
     (per-issue correctness correlation, oracle accuracy).
  2. q->bug and predicted-bug share for every method on test (temporal shift check).
  3. ansible: what the read-out gets wrong vs SetFit (confusion + examples).
"""
import numpy as np
import pandas as pd

from common import EXP, LABELS, RES

L2I = {l: i for i, l in enumerate(LABELS)}
enc = lambda s: np.array([L2I.get(str(x).strip().lower(), -1) for x in s])

test = pd.read_csv(RES / "agnostic" / "neighbors" / "test_split.csv", keep_default_na=False)
y = enc(test["labels"])
m = pd.read_parquet(EXP / "headroom" / "master_preds.parquet")
preds = {"SetFit-PS": enc(m["setfit_issues_PS"]), "FT-PA-14B": enc(m["ft_PA_14B"]),
         "BRAGTAG-PS-32B": enc(m["bragtag_PS_32B_k12"]), "BRAGTAG-PS-14B": enc(m["bragtag_PS_14B_k15"])}
for name in ["readout_14B", "readout_7B", "stateknn_14B", "readout_32B", "stateknn_32B", "fusion"]:
    f = EXP / "test_preds" / f"{name}.csv"
    if f.exists():
        preds[name] = enc(pd.read_csv(f, keep_default_na=False)["predicted_label"])

print("== 1. complementarity with readout_14B")
r = preds["readout_14B"] == y
for k, v in preds.items():
    if k == "readout_14B":
        continue
    c = v == y
    print(f"  {k:16s} acc={c.mean():.3f}  corr(correct)={np.corrcoef(r, c)[0, 1]:.3f}  "
          f"oracle={np.mean(r | c):.3f}  readout-only-right={np.mean(r & ~c):.3f}  other-only-right={np.mean(~r & c):.3f}")

print("== 2. q->bug and predicted-bug share on test")
for k, v in preds.items():
    print(f"  {k:16s} q->bug={np.mean(v[y == 2] == 0):.3f}  pred_bug={np.mean(v == 0):.3f}  "
          f"bug->question={np.mean(v[y == 0] == 2):.3f}")

print("== 3. ansible")
a = (test["repo"] == "ansible/ansible").to_numpy()
for k in ["SetFit-PS", "readout_14B"]:
    print(k)
    print(pd.crosstab(pd.Series(np.array(LABELS)[y[a]], name="true"),
                      pd.Series(np.array(LABELS)[preds[k][a]], name="pred")))
wrong = np.where(a & (preds["readout_14B"] != y) & (preds["SetFit-PS"] == y))[0]
print(f"ansible: read-out wrong & SetFit right: {len(wrong)}")
for i in wrong[:10]:
    print(f"  true={LABELS[y[i]]:8s} readout={LABELS[preds['readout_14B'][i]]:8s} | {test.loc[i, 'title'][:90]}")

print("== 4. robustness of the headline comparison (post-hoc)")
from scipy.stats import binomtest

from common import macro_f1

proj = test["repo"].to_numpy()
projects = np.unique(proj)
rng = np.random.default_rng(0)
for cand in ["readout_14B", "readout_32B", "fusion"]:
    if cand not in preds:
        continue
    for base in ["SetFit-PS", "FT-PA-14B"]:
        a, b = preds[cand], preds[base]
        # exact McNemar on correctness
        n01 = int(np.sum((a == y) & (b != y))); n10 = int(np.sum((a != y) & (b == y)))
        p = binomtest(n01, n01 + n10, 0.5).pvalue
        # project-cluster bootstrap of the pooled macro-F1 difference
        idx_by = {pr: np.where(proj == pr)[0] for pr in projects}
        d = []
        for _ in range(2000):
            ii = np.concatenate([idx_by[pr] for pr in rng.choice(projects, len(projects), replace=True)])
            d.append(macro_f1(y[ii], a[ii]) - macro_f1(y[ii], b[ii]))
        print(f"  {cand} vs {base}: diff={macro_f1(y, a) - macro_f1(y, b):+.4f}  McNemar {n01}/{n10} p={p:.4f}  "
              f"project-cluster bootstrap 95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")

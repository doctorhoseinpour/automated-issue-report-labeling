#!/usr/bin/env python3
"""Headroom analysis over the existing test predictions (read-only).

Answers, on the paper's 3,300-issue test split:
  1. Verify the bar numbers (pooled macro F1 per method).
  2. Is the true label present among the retrieved neighbors (coverage@k)?
  3. When does the LLM override its neighbors (right vs wrong)?
  4. How complementary are SetFit / FT / BRAGTAG errors (oracle ceilings)?
  5. How many issues does *every* method get wrong (label-noise proxy)?

Usage (lab machine):
  venv/bin/python scripts/experiments/rag_next/headroom.py > results/issues11k/exploration/rag_next/headroom/headroom.txt
"""
from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[3]
RES = Path(os.environ.get("RESULTS_DIR", REPO / "results" / "issues11k"))
H = RES / "exploration" / "rag_next" / "headroom"
LABELS = ["bug", "feature", "question"]


def mf1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def pcf1(y, p):
    return f1_score(y, p, labels=LABELS, average=None, zero_division=0)


def read(path):
    """Parquet with Arrow-backed strings turned into plain numpy object columns."""
    d = pd.read_parquet(path)
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]):
            d[c] = np.asarray(d[c].astype(str).tolist(), dtype=object)
    return d


def main():
    m = read(H / "master_preds.parquet")
    nps = read(H / "neighbors_ps.parquet")
    npa = read(H / "neighbors_pa.parquet")
    y = m["label"].to_numpy()
    meth = [c for c in m.columns if c not in
            ("gidx", "repo", "proj", "local_idx", "created_at", "label", "title_len", "body_len")]

    print("=" * 70, "\n1. Pooled macro F1 per method (raw; invalid = wrong)\n", "=" * 70)
    rows = []
    for c in meth:
        p = m[c].to_numpy()
        f = pcf1(y, p)
        rows.append((c, mf1(y, p), *f, (p == "invalid").mean(), (p == "bug").mean(),
                     ((y == "question") & (p == "bug")).sum() / (y == "question").sum()))
    t = pd.DataFrame(rows, columns=["method", "macroF1", "f1_bug", "f1_feat", "f1_q",
                                    "invalid", "pred_bug_share", "q2bug"])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 400)
    print(t.sort_values("macroF1", ascending=False).head(25).round(4).to_string(index=False))
    t.to_csv(H / "method_scores.csv", index=False)

    # best configs used as references
    ref = {
        "setfit_issues_PS": "setfit_issues_PS",
        "setfit_mpnet_PS": "setfit_mpnet_PS",
        "ft_PA_14B": "ft_PA_14B",
        "bragtag32": "bragtag_PS_32B_k12",
        "ragtag32": "ragtag_PS_32B_k12",
        "vtag_ps15": "vtag_PS_k15",
    }
    for size in ["3B", "7B", "14B", "32B"]:
        best = max([c for c in meth if c.startswith(f"bragtag_PS_{size}_")], key=lambda c: mf1(y, m[c].to_numpy()))
        ref[f"bragtag_best_{size}"] = best
        bestr = max([c for c in meth if c.startswith(f"ragtag_PS_{size}_k") and not c.endswith("_k0")],
                    key=lambda c: mf1(y, m[c].to_numpy()))
        ref[f"ragtag_best_{size}"] = bestr
    print("\nReference configs:", ref)

    print("\n" + "=" * 70, "\n2. Neighbor label coverage (true label among top-k)\n", "=" * 70)
    for name, nb in [("PS", nps), ("PA", npa)]:
        L = nb[[f"l{r}" for r in range(30)]].to_numpy()
        S = nb[[f"s{r}" for r in range(30)]].to_numpy()
        for k in [1, 3, 6, 9, 12, 15, 30]:
            cov = (L[:, :k] == y[:, None]).any(1)
            maj = (L[:, :k] == y[:, None]).mean(1)
            s = f"  {name} k={k:2d}: coverage={cov.mean():.3f}"
            for lab in LABELS:
                mk = y == lab
                s += f" | {lab}: cov={cov[mk].mean():.3f} share={maj[mk].mean():.3f}"
            print(s)
        # label shares of neighbor slots
        for k in [9, 15]:
            sl = pd.Series(L[:, :k].ravel()).value_counts(normalize=True).round(3).to_dict()
            print(f"  {name} neighbor-slot label shares @{k}: {sl}")

    print("\n" + "=" * 70, "\n3. LLM vs neighbor-majority (PS, best RAGTAG/BRAGTAG per size)\n", "=" * 70)
    L = nps[[f"l{r}" for r in range(30)]].to_numpy()
    S = nps[[f"s{r}" for r in range(30)]].to_numpy()
    for size in ["3B", "7B", "14B", "32B"]:
        for kind in ["ragtag", "bragtag"]:
            c = ref[f"{kind}_best_{size}"]
            k = int(c.split("_k")[-1])
            vt = m[f"vtag_PS_k{k}"].to_numpy() if f"vtag_PS_k{k}" in m else None
            p = m[c].to_numpy()
            agree = p == vt
            right_ll = (p == y)
            right_vt = (vt == y)
            override = ~agree & (p != "invalid")
            print(f"  {c}: agree-with-VOTAG@{k}={agree.mean():.3f}; "
                  f"acc|agree={right_ll[agree].mean():.3f}; overrides={override.mean():.3f} "
                  f"(LLM right {np.mean(right_ll[override]):.3f}, VOTAG right {np.mean(right_vt[override]):.3f}); "
                  f"LLM wrong & VOTAG right={np.mean(~right_ll & right_vt):.3f}; "
                  f"LLM right & VOTAG wrong={np.mean(right_ll & ~right_vt):.3f}")

    print("\n" + "=" * 70, "\n4. Complementarity / oracle ceilings\n", "=" * 70)
    keys = ["setfit_issues_PS", "setfit_mpnet_PS", "ft_PA_14B", "ft_PA_32B", "bragtag32", "ragtag32",
            "bragtag_best_14B", "bragtag_best_7B", "vtag_ps15", "roberta_PA"]
    cols = {k: m[ref.get(k, k)].to_numpy() for k in keys}
    C = {k: (v == y) for k, v in cols.items()}
    print("  single acc:", {k: round(v.mean(), 4) for k, v in C.items()})
    for a, b in combinations(keys, 2):
        both_wrong = (~C[a] & ~C[b]).mean()
        oracle = (C[a] | C[b]).mean()
        # error correlation (phi over correctness)
        r = np.corrcoef(C[a].astype(float), C[b].astype(float))[0, 1]
        if a.startswith("setfit_issues") or b.startswith("setfit_issues") or (a, b) in [("ft_PA_14B", "bragtag32")]:
            print(f"  {a:18s} + {b:18s}: oracle acc={oracle:.4f} both-wrong={both_wrong:.4f} corr={r:.3f}")
    # three-way
    allk = [k for k in keys]
    anyright = np.zeros(len(y), bool)
    for k in allk:
        anyright |= C[k]
    print(f"  oracle over {len(allk)} methods: acc={anyright.mean():.4f}")
    # all LLM configs + encoders
    allc = [c for c in meth if not c.startswith("vtag")]
    R = np.stack([(m[c].to_numpy() == y) for c in allc], 1)
    print(f"  oracle over ALL {len(allc)} non-VOTAG configs: acc={R.any(1).mean():.4f}")
    nright = R.mean(1)
    print("  fraction of configs correct per issue: quantiles",
          np.quantile(nright, [0.05, 0.1, 0.2, 0.3, 0.5]).round(3))
    for thr in [0.0, 0.1, 0.2]:
        mk = nright <= thr
        print(f"   issues with <= {thr:.0%} of configs right: {mk.sum()} ({mk.mean():.3f}); "
              f"label mix {pd.Series(y[mk]).value_counts().to_dict()}")
    # the 'consistently wrong' predicted label
    mk = nright <= 0.1
    maj_wrong = []
    for i in np.where(mk)[0]:
        vals = pd.Series([m[c].to_numpy()[i] for c in allc]).value_counts()
        maj_wrong.append(vals.index[0])
    print("   their majority predicted label x true label:")
    print(pd.crosstab(pd.Series(y[mk], name="true"), pd.Series(maj_wrong, name="consensus_pred")))
    # save per-issue difficulty
    pd.DataFrame({"gidx": m["gidx"], "frac_configs_right": nright}).to_parquet(H / "issue_difficulty.parquet")

    print("\n" + "=" * 70, "\n5. Per-project macro F1 for reference methods\n", "=" * 70)
    rows = []
    for proj, g in m.groupby("proj"):
        yy = g["label"].to_numpy()
        rows.append({"proj": proj, **{k: round(mf1(yy, g[ref.get(k, k)].to_numpy()), 3) for k in keys}})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 70, "\n6. Confusions for references\n", "=" * 70)
    for k in ["setfit_issues_PS", "ft_PA_14B", "bragtag32"]:
        print(k)
        print(pd.crosstab(pd.Series(y, name="true"), pd.Series(cols[k], name="pred")))

    print("\n" + "=" * 70, "\n7. SetFit confidence vs accuracy (PS issues)\n", "=" * 70)
    pr = read(H / "setfit_probs.parquet")
    P = pr[[f"setfit_issues_PS_p_{l}" for l in LABELS]].to_numpy()
    conf = P.max(1)
    sf = cols["setfit_issues_PS"]
    for lo, hi in [(0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
        mk = (conf >= lo) & (conf < hi)
        if mk.sum():
            other = {kk: (cols[kk][mk] == y[mk]).mean().round(3) for kk in ["ft_PA_14B", "bragtag32", "vtag_ps15"]}
            print(f"  conf [{lo:.1f},{hi:.1f}): n={mk.sum():4d} setfit acc={np.mean(sf[mk] == y[mk]):.3f}  others acc={other}")
    # top-2 structure
    srt = np.sort(P, 1)
    marg = srt[:, -1] - srt[:, -2]
    top2 = np.argsort(-P, 1)[:, :2]
    in2 = np.array([LABELS.index(y[i]) in top2[i] for i in range(len(y))])
    print(f"  true label in SetFit top-2: {in2.mean():.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score the pilot arms on dev (never test).

For every arm (p0..p5 of adjudicate.py, plus Qwen-7B dev read-outs from rag_next when
present) it reports, on the routed set R (SetFit's lowest-margin dev issues):
  accuracy, macro F1, per-class F1, question->bug rate, fixes/breaks vs SetFit;
and on the full dev split (990): the cascade "SetFit outside R, arm inside R", and a
cross-fitted fusion inside R (LR on SetFit + arm log-probs, 5-fold x 5 repeats), each
as a paired bootstrap difference against SetFit alone. Also: a non-agentic global
stacker over all dev (SetFit + Qwen-7B read-outs), cost per routed issue, agent tool use.

Usage (lab machine):
  venv/bin/python scripts/experiments/agentic/analyze_pilot.py --tag q14 [--n_route 300]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import EXP, LABELS, RAGNEXT, boot_diff, load_pool, load_setfit_dev, macro_f1, per_class_f1, routed_uids  # noqa: E402

L2I = {l: i for i, l in enumerate(LABELS)}


def norm_lp(lp):
    lp = lp - lp.max(1, keepdims=True)
    return lp - np.log(np.exp(lp).sum(1, keepdims=True))


def cv_fuse(X, y, strata, repeats=5, seed=0, C=1.0):
    """Out-of-fold predictions of an LR stacker; returns (mean F1 over repeats, oof of repeat 0)."""
    scores, oof0 = [], None
    if min(Counter(strata).values()) < 5:  # tiny strata (e.g. project x label on few issues): stratify by label
        strata = y
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + r)
        oof = np.zeros(len(y), int)
        for a, b in skf.split(X, strata):
            oof[b] = LogisticRegression(C=C, max_iter=5000).fit(X[a], y[a]).predict(X[b])
        scores.append(macro_f1(np.array(LABELS)[y], np.array(LABELS)[oof]))
        oof0 = oof if oof0 is None else oof0
    return float(np.mean(scores)), float(np.std(scores)), oof0


def tfidf_logp(pool, uids):
    """Per-project TF-IDF (1-2 grams) + LR, fit on role == inner; log-probs for the given dev uids."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    by = pool.set_index("uid")
    out = {}
    for proj in by.loc[uids, "proj"].unique():
        tr = pool[(pool.proj == proj) & (pool.role == "inner")]
        q = [u for u in uids if by.loc[u, "proj"] == proj]
        v = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=50000)
        clf = LogisticRegression(C=10, max_iter=3000).fit(v.fit_transform(tr.title + " \n " + tr.body), tr.label)
        P = clf.predict_proba(v.transform(by.loc[q, "title"] + " \n " + by.loc[q, "body"]))
        P = P[:, [list(clf.classes_).index(l) for l in LABELS]]
        for u, row in zip(q, P):
            out[int(u)] = np.log(np.clip(row, 1e-4, 1))
    return np.array([out[int(u)] for u in uids])


def ragnext_readout(name, uids):
    f = RAGNEXT / "features" / f"{name}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    pos = {int(u): i for i, u in enumerate(z["uids"])}
    if not all(int(u) in pos for u in uids):
        return None
    return norm_lp(z["logp"][[pos[int(u)] for u in uids]].astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_route", type=int, default=300)
    ap.add_argument("--root", default=str(EXP / "pilot"))
    ap.add_argument("--extra_tag", default=None, help="another run dir (e.g. a routed subsample) for paired contrasts")
    args = ap.parse_args()
    out = Path(args.root) / args.tag

    pool = load_pool().set_index("uid")
    sf = load_setfit_dev()
    sf["label"] = pool.loc[sf.uid, "label"].to_numpy()
    sf["proj"] = pool.loc[sf.uid, "proj"].to_numpy()
    sf = sf.sort_values("uid").reset_index(drop=True)
    y_all = sf["label"].to_numpy()
    S_all = sf["pred"].to_numpy()
    R = set(int(u) for u in routed_uids(args.n_route))
    inR = sf["uid"].isin(R).to_numpy()
    Psf = sf[[f"p_{l}" for l in LABELS]].to_numpy()
    lpsf = np.log(np.clip(Psf, 1e-4, 1))

    print("=" * 80)
    f = per_class_f1(y_all, S_all)
    print(f"SetFit-PS (inner->dev): macro F1 {macro_f1(y_all, S_all):.4f} acc {np.mean(S_all == y_all):.4f} "
          f"F1 bug/feat/q {f[0]:.3f}/{f[1]:.3f}/{f[2]:.3f}  q->bug {np.mean(S_all[y_all == 'question'] == 'bug'):.3f}")
    err = S_all != y_all
    print(f"routed R: {inR.sum()} of {len(sf)} ({inR.mean():.1%}); SetFit acc in R {np.mean(S_all[inR] == y_all[inR]):.3f}, "
          f"outside {np.mean(S_all[~inR] == y_all[~inR]):.3f}; share of SetFit errors inside R {err[inR].sum() / err.sum():.3f}; "
          f"max margin in R {sf.margin[inR].max():.3f}")
    print("label mix in R:", Counter(y_all[inR]))

    arms = {}
    for a in ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"]:
        fa = out / f"{a}.csv"
        if fa.exists():
            d = pd.read_csv(fa).set_index("uid")
            arms[a] = d
    uR = sf["uid"].to_numpy()[inR]
    yR = y_all[inR]
    SR = S_all[inR]
    lpsfR = lpsf[inR]
    strata = np.array([f"{p}|{l}" for p, l in zip(sf["proj"].to_numpy()[inR], yR)])
    # non-LLM second opinion (control): per-project TF-IDF LR, fit on inner
    extra = {"TFIDF-LR (non-LLM)": tfidf_logp(pool.reset_index(), uR)}
    # rag_next Qwen-7B read-outs over the same routed issues (constrained decoding)
    for name, tag in [("q7_k0", "ZS-7B"), ("q7_k12_dev", "RAGTAG-7B k12"), ("q7_k12b3_dev", "BRAGTAG-7B k12"),
                      ("q14_k0", "ZS-14B")]:
        lp = ragnext_readout(name, uR)
        if lp is not None:
            extra[tag] = lp

    rows = []
    cost = []
    for name, lp in [(a, d.loc[uR, [f"lp_{l}" for l in LABELS]].to_numpy()) for a, d in arms.items()] + list(extra.items()):
        lp = norm_lp(lp)
        pR = np.array(LABELS)[lp.argmax(1)]
        fc = per_class_f1(yR, pR)
        full = S_all.copy()
        full[inR] = pR
        d, lo, hi = boot_diff(y_all, S_all, full)
        X = np.concatenate([lpsfR, lp], 1)
        yid = np.array([L2I[v] for v in yR])
        mu, sd, oof = cv_fuse(X, yid, strata)
        fus = S_all.copy()
        fus[inR] = np.array(LABELS)[oof]
        fd, flo, fhi = boot_diff(y_all, S_all, fus)
        poe = S_all.copy()
        poe[inR] = np.array(LABELS)[(lpsfR + lp).argmax(1)]
        pdiff = macro_f1(y_all, poe) - macro_f1(y_all, S_all)
        rows.append(dict(arm=name, acc_R=np.mean(pR == yR), F1_R=macro_f1(yR, pR), f1_bug=fc[0], f1_feat=fc[1], f1_q=fc[2],
                         q2bug_R=np.mean(pR[yR == "question"] == "bug"),
                         fixes=int(np.sum((SR != yR) & (pR == yR))), breaks=int(np.sum((SR == yR) & (pR != yR))),
                         cascade_dF1=d, cascade_lo=lo, cascade_hi=hi,
                         fuse_dF1=fd, fuse_lo=flo, fuse_hi=fhi, fuse_F1_R_cvmean=mu, fuse_F1_R_cvsd=sd,
                         poe_dF1=pdiff))
        if name in arms:
            meta = json.load(open(out / f"{name}.json"))
            dd = arms[name]
            cost.append(dict(arm=name, n=len(dd), prompt_tok_per_issue=dd.prompt_tokens.mean(),
                             gen_tok_per_issue=dd.gen_tokens.mean(), calls_per_issue=dd.n_calls.mean(),
                             wall_s_per_issue=meta["wall_s"] / meta["n"], gpu_peak_gb=meta["gpu_peak_mb"] / 1024,
                             top1_is_label=meta["top1_is_label"]))
    pd.set_option("display.width", 250)
    print("\nSetFit on R: acc %.3f, macro F1 %.3f" % (np.mean(SR == yR), macro_f1(yR, SR)))
    t = pd.DataFrame(rows)
    print("\nArms on R, and full-dev deltas vs SetFit alone (paired bootstrap 95% CI, B=2000):")
    print(t.round(4).to_string(index=False))
    t.to_csv(out / "summary_arms.csv", index=False)
    if cost:
        c = pd.DataFrame(cost)
        print("\nCost per routed issue (measured; prompt tokens count every prefill, incl. re-scoring passes):")
        print(c.round(3).to_string(index=False))
        c.to_csv(out / "summary_cost.csv", index=False)

    paired_contrasts(out, Path(args.root) / args.extra_tag if args.extra_tag else None, pool)

    # agent behaviour
    for ag, base in [("p4", "p2"), ("p6", "p1")]:
        agent_stats(out, ag, base, arms, uR, yR)
    if args.extra_tag:
        for ag, base in [("p4", "p2"), ("p6", "p1")]:
            ex = {a: pd.read_csv(Path(args.root) / args.extra_tag / f"{a}.csv").set_index("uid")
                  for a in [ag, base] if (Path(args.root) / args.extra_tag / f"{a}.csv").exists()}
            if ag in ex:
                ex.setdefault(base, arms.get(base))
                u = ex[ag].index.to_numpy()
                agent_stats(Path(args.root) / args.extra_tag, ag, base, ex, u, ex[ag]["label"].to_numpy())

    _global_stacker(sf, y_all, S_all, lpsf)


def paired_contrasts(out, extra, pool, B=4000, seed=0):
    """Routed-accuracy differences between arms on their common issues, paired bootstrap 95% CI."""
    arms = {}
    for d in [out] + ([extra] if extra is not None else []):
        for f in sorted(d.glob("p[0-9].csv")):
            arms.setdefault(f.stem, pd.read_csv(f).set_index("uid"))
    pairs = [("p1", "p0", "contrastive prompt vs paper prompt"), ("p2", "p1", "SetFit hint"),
             ("p8", "p2", "6 vs 3 examples per label (compute-matched)"), ("p3", "p2", "reasoning (CoT)"),
             ("p5", "p2", "agent-chosen extra information"), ("p4", "p5", "AGENCY given same info (replay)"),
             ("p4", "p3", "tools given reasoning"), ("p4", "p2", "agent vs fixed bundle"),
             ("p6", "p7", "AGENCY, no hint (replay)"), ("p6", "p1", "pure-LLM agent vs its bundle"),
             ("p7", "p1", "agent-chosen info, no hint"), ("p1", "p9", "contrastive prompt vs matched-size BRAGTAG")]
    rng = np.random.default_rng(seed)
    print("\nPaired contrasts on routed issues (accuracy difference a - b, paired bootstrap 95% CI):")
    for a, b, what in pairs:
        if a not in arms or b not in arms:
            continue
        u = np.intersect1d(arms[a].index.to_numpy(), arms[b].index.to_numpy())
        y = arms[a].loc[u, "label"].to_numpy()
        ca = (arms[a].loc[u, "pred"].to_numpy() == y).astype(float)
        cb = (arms[b].loc[u, "pred"].to_numpy() == y).astype(float)
        dd = ca - cb
        bs = np.array([dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(B)])
        print(f"  {a}-{b:3s} n={len(u):3d}  {dd.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]"
              f"  (a right/b wrong {int(((ca == 1) & (cb == 0)).sum())}, reverse {int(((ca == 0) & (cb == 1)).sum())})  {what}")


def agent_stats(out, ag, base, arms, uR, yR):
    tf = out / f"{ag}_traces.jsonl"
    if tf.exists():
        tr = [json.loads(l) for l in open(tf)]
        n_calls = Counter(len(x["trace"]) for x in tr)
        names = Counter(c["name"] for x in tr for c in x["trace"])
        pats = Counter(str(c["args"].get("pattern")) for x in tr for c in x["trace"] if c["name"] == "label_stats")
        print(f"\nAgent {ag}: tool calls per issue", dict(sorted(n_calls.items())), "| by tool", dict(names))
        print("  most common label_stats patterns:", pats.most_common(12))
        errs = sum(1 for x in tr for c in x["trace"] if str(c["result"]).startswith("error"))
        print(f"  tool errors: {errs}")
        if ag in arms and base in arms:
            a4 = arms[ag].loc[uR, "pred"].to_numpy()
            a2 = arms[base].loc[uR, "pred"].to_numpy()
            pos = {int(u): i for i, u in enumerate(uR)}
            used = np.zeros(len(uR), bool)
            for x in tr:
                used[pos[int(x["uid"])]] = len(x["trace"]) > 0
            print(f"  {ag} vs {base} disagree on {np.mean(a4 != a2):.3f} of R; tools used on {used.mean():.2f} of R; "
                  f"on those: {base} acc {np.mean(a2[used] == yR[used]):.3f}, {ag} acc {np.mean(a4[used] == yR[used]):.3f}; "
                  f"on the rest: {base} acc {np.mean(a2[~used] == yR[~used]):.3f}, {ag} acc {np.mean(a4[~used] == yR[~used]):.3f}")


def _global_stacker(sf, y_all, S_all, lpsf):
    """Non-agentic control: LR stacker over all dev issues, when rag_next dev read-outs exist."""
    uall = sf["uid"].to_numpy()
    comps = {"setfit": lpsf}
    for name, tag in [("q7_k0", "zs7"), ("q7_k12b3_dev", "bragtag7"), ("q14_k0", "zs14")]:
        lp = ragnext_readout(name, uall)
        if lp is not None:
            comps[tag] = lp
    if len(comps) > 1:
        yid = np.array([L2I[v] for v in y_all])
        strata_all = np.array([f"{p}|{l}" for p, l in zip(sf["proj"], y_all)])
        print("\nNon-agentic control: LR stacker over all 990 dev issues (5-fold x 5 repeats, stratified by project x label)")
        keys = list(comps)
        for r in range(1, len(keys) + 1):
            from itertools import combinations
            for sub in combinations(keys, r):
                X = np.concatenate([comps[k] for k in sub], 1)
                mu, sd, oof = cv_fuse(X, yid, strata_all)
                pr = np.array(LABELS)[oof]
                d, lo, hi = boot_diff(y_all, S_all, pr)
                print(f"  {'+'.join(sub):28s} CV F1 {mu:.4f}±{sd:.4f}  vs SetFit {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()

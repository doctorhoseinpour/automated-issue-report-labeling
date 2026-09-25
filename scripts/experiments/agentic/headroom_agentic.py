#!/usr/bin/env python3
"""Agent-oriented headroom diagnostics over the EXISTING test predictions (read-only).

Everything here is descriptive: oracle ceilings, disagreement structure, and how
well cheap signals would *find* SetFit's errors. No method is fit or selected on
the test split; routing thresholds for the proposal are chosen on dev instead.

Input: a snapshot of rag_next's aggregation of results/issues11k predictions
(master_preds.parquet, setfit_probs.parquet), copied into
results/issues11k/exploration/agentic/headroom/ so later rag_next reruns cannot
shift these numbers.

Usage (lab machine, repo root):
  venv/bin/python scripts/experiments/agentic/headroom_agentic.py \
      > results/issues11k/exploration/agentic/headroom/headroom_agentic.txt
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

REPO = Path(__file__).resolve().parents[3]
RES = Path(os.environ.get("RESULTS_DIR", REPO / "results" / "issues11k"))
H = RES / "exploration" / "agentic" / "headroom"
LABELS = ["bug", "feature", "question"]


def mf1(y, p):
    return f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def read(path):
    d = pd.read_parquet(path)
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]):
            d[c] = np.asarray(d[c].astype(str).tolist(), dtype=object)
    return d


def boot_ci(y, p, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    v = [mf1(y[ii], p[ii]) for ii in (rng.integers(0, n, n) for _ in range(B))]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    m = read(H / "master_preds.parquet")
    pr = read(H / "setfit_probs.parquet")
    y = m["label"].to_numpy()
    n = len(y)
    S = m["setfit_issues_PS"].to_numpy()
    P = pr[[f"setfit_issues_PS_p_{l}" for l in LABELS]].to_numpy()

    best = {  # best-k per size, as in the paper (selected on test by the paper, reused here only as references)
        "bragtag": {"3B": "bragtag_PS_3B_k6", "7B": "bragtag_PS_7B_k12", "14B": "bragtag_PS_14B_k15", "32B": "bragtag_PS_32B_k12"},
        "ragtag": {"3B": "ragtag_PS_3B_k3", "7B": "ragtag_PS_7B_k6", "14B": "ragtag_PS_14B_k12", "32B": "ragtag_PS_32B_k12"},
        "ft": {s: f"ft_PA_{s}" for s in ["3B", "7B", "14B", "32B"]},
        "zs": {"3B": "ragtag_PA_3B_k0", "7B": "ragtag_PA_7B_k0", "14B": "ragtag_PA_14B_k0", "32B": "ragtag_PS_32B_k0"},
    }
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)

    print("=" * 78, "\nA. Bars (pooled macro F1 on 3,300 test issues, raw; 95% bootstrap CI)\n" + "=" * 78)
    for c in ["setfit_issues_PS", "setfit_mpnet_PS", "ft_PA_14B", "bragtag_PS_32B_k12", "bragtag_PS_14B_k15",
              "bragtag_PS_7B_k12", "bragtag_PS_3B_k6", "ragtag_PS_32B_k12", "ragtag_PS_14B_k12", "ragtag_PS_7B_k6",
              "ragtag_PS_3B_k3", "vtag_PS_k15", "roberta_PA"]:
        p = m[c].to_numpy()
        lo, hi = boot_ci(y, p)
        print(f"  {c:22s} {mf1(y, p):.4f}  [{lo:.3f}, {hi:.3f}]")

    print("\n" + "=" * 78, "\nB. SetFit-PS vs one other method: disagreement structure\n" + "=" * 78)
    rows = []
    others = ["setfit_mpnet_PS", "roberta_PA", "vtag_PS_k15"] + \
        [best[f][s] for f in ["zs", "ragtag", "bragtag", "ft"] for s in ["3B", "7B", "14B", "32B"]]
    for c in others:
        L = m[c].to_numpy()
        dis = S != L
        sr, lr = S == y, L == y
        orc = np.where(sr | lr, y, S)
        take_l = np.where(dis & (L != "invalid"), L, S)
        rows.append(dict(other=c, disagree=dis.mean(), acc_agree=sr[~dis].mean(),
                         S_right_dis=sr[dis].mean(), L_right_dis=lr[dis].mean(),
                         neither_dis=(~sr & ~lr)[dis].mean(),
                         oracle_F1=mf1(y, orc), takeL_on_dis_F1=mf1(y, take_l)))
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    print("  (SetFit alone = %.4f; oracle_F1 picks the right one whenever either is right;"
          " takeL_on_dis_F1 = always trust the other method on disagreement)" % mf1(y, S))

    print("\n" + "=" * 78, "\nC. Where SetFit is wrong: confusion x who fixes it\n" + "=" * 78)
    wrong = S != y
    print(f"  SetFit errors: {wrong.sum()} / {n} ({wrong.mean():.3f})")
    fixers = {"FT-14B": "ft_PA_14B", "BRAGTAG-32B": "bragtag_PS_32B_k12", "BRAGTAG-7B": "bragtag_PS_7B_k12",
              "RAGTAG-7B": "ragtag_PS_7B_k6", "ZS-32B": "ragtag_PS_32B_k0", "SetFit-mpnet": "setfit_mpnet_PS"}
    llm_cols = [c for c in m.columns if c.startswith(("ragtag_PS_", "bragtag_PS_", "ft_PA_", "ragtag_PA_"))
                and ("llama" not in c)]
    anyllm = np.stack([m[c].to_numpy() == y for c in llm_cols], 1)
    rows = []
    for (tl, pl), g in pd.DataFrame({"t": y[wrong], "p": S[wrong], "i": np.where(wrong)[0]}).groupby(["t", "p"]):
        ii = g["i"].to_numpy()
        r = dict(true=tl, setfit=pl, n=len(ii))
        for k, c in fixers.items():
            r[k] = (m[c].to_numpy()[ii] == y[ii]).mean()
        r["any_LLM_cfg"] = anyllm[ii].any(1).mean()
        r["frac_LLM_cfgs_right"] = anyllm[ii].mean()
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n" + "=" * 78, "\nD. Routing signals: do they find SetFit's errors? (AUROC; recall of errors at budget)\n" + "=" * 78)
    srt = np.sort(P, 1)
    sig = {
        "1-maxprob": 1 - P.max(1),
        "-margin(p1-p2)": -(srt[:, -1] - srt[:, -2]),
        "entropy": -(P * np.log(np.clip(P, 1e-12, 1))).sum(1),
        "dis_setfit_mpnet": (S != m["setfit_mpnet_PS"].to_numpy()).astype(float),
        "dis_vtag15": (S != m["vtag_PS_k15"].to_numpy()).astype(float),
        "dis_ZS7B": (S != m["ragtag_PA_7B_k0"].to_numpy()).astype(float),
        "dis_RAGTAG7B": (S != m["ragtag_PS_7B_k6"].to_numpy()).astype(float),
        "dis_BRAGTAG7B": (S != m["bragtag_PS_7B_k12"].to_numpy()).astype(float),
        "dis_BRAGTAG14B": (S != m["bragtag_PS_14B_k15"].to_numpy()).astype(float),
        "dis_BRAGTAG32B": (S != m["bragtag_PS_32B_k12"].to_numpy()).astype(float),
        "dis_FT14B": (S != m["ft_PA_14B"].to_numpy()).astype(float),
    }
    panel7 = ["ragtag_PA_7B_k0", "ragtag_PS_7B_k6", "bragtag_PS_7B_k12", "vtag_PS_k15", "setfit_mpnet_PS"]
    sig["n_dis_panel(7B+votag+mpnet)"] = np.stack([S != m[c].to_numpy() for c in panel7], 1).sum(1).astype(float)
    sig["n_dis_panel+margin"] = sig["n_dis_panel(7B+votag+mpnet)"] + 0.5 * (1 - (srt[:, -1] - srt[:, -2]))
    rng = np.random.default_rng(0)
    rows = []
    for k, s in sig.items():
        s2 = s + 1e-9 * rng.standard_normal(n)  # break ties randomly
        order = np.argsort(-s2)
        r = dict(signal=k, AUROC=roc_auc_score(wrong, s2), routed_if_binary=(s > 0).mean() if set(np.unique(s)) <= {0, 1} else np.nan)
        for b in [0.1, 0.2, 0.3, 0.4]:
            top = np.zeros(n, bool)
            top[order[: int(b * n)]] = True
            r[f"err_recall@{int(b*100)}%"] = (top & wrong).sum() / wrong.sum()
            r[f"setfit_acc_in_routed@{int(b*100)}%"] = (S[top] == y[top]).mean()
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n" + "=" * 78, "\nE. Budgeted oracle: route top-b% by a signal, then an ORACLE adjudicator"
          " chooses among {SetFit, panel member}.\n   Upper bound on what any adjudicator restricted to those candidates can add.\n" + "=" * 78)
    cand_cols = ["bragtag_PS_32B_k12", "ft_PA_14B", "bragtag_PS_7B_k12", "ragtag_PA_7B_k0", "vtag_PS_k15"]
    for sname in ["n_dis_panel+margin", "dis_BRAGTAG7B", "-margin(p1-p2)"]:
        s2 = sig[sname] + 1e-9 * rng.standard_normal(n)
        order = np.argsort(-s2)
        line = f"  {sname:22s}"
        for b in [0.1, 0.2, 0.3, 0.5, 1.0]:
            top = np.zeros(n, bool)
            top[order[: int(b * n)]] = True
            for cc in [["bragtag_PS_7B_k12"], ["bragtag_PS_32B_k12"], cand_cols]:
                C = np.stack([m[c].to_numpy() == y for c in cc], 1).any(1)
                pred = np.where(top & C, y, S)
                line += f" | b={b:.0%} {'+'.join(x.split('_')[0] + x.split('_')[2] if 'PS' in x or 'PA' in x else x for x in cc)[:18]}={mf1(y, pred):.3f}"
        print(line)

    print("\n" + "=" * 78, "\nF. Issues that strong, diverse methods ALL get wrong (label-noise / unrecoverable proxy)\n" + "=" * 78)
    strong = ["setfit_issues_PS", "setfit_mpnet_PS", "roberta_PA", "ft_PA_14B", "ft_PA_32B",
              "bragtag_PS_32B_k12", "ragtag_PS_32B_k12", "bragtag_PS_14B_k15"]
    R = np.stack([m[c].to_numpy() == y for c in strong], 1)
    allw = ~R.any(1)
    print(f"  wrong under all {len(strong)} strong methods: {allw.sum()} ({allw.mean():.3f}); label mix "
          f"{pd.Series(y[allw]).value_counts().to_dict()}")
    maj = []
    for i in np.where(allw)[0]:
        maj.append(pd.Series([m[c].to_numpy()[i] for c in strong]).value_counts().index[0])
    print(pd.crosstab(pd.Series(y[allw], name="true"), pd.Series(maj, name="consensus_pred")))
    for kk in [1, 2]:
        few = R.sum(1) <= kk
        print(f"  right under <= {kk} of {len(strong)}: {few.sum()} ({few.mean():.3f})")
    ceil = np.where(allw, S, y)
    print(f"  ceiling if every issue some strong method gets right were fixed: macro F1 {mf1(y, ceil):.4f}")
    pd.DataFrame({"gidx": m["gidx"], "n_strong_right": R.sum(1)}).to_parquet(H / "strong_right_counts.parquet")

    print("\n" + "=" * 78, "\nG. Per project: SetFit-PS vs best LLM and their oracle\n" + "=" * 78)
    rows = []
    for proj, g in m.groupby("proj"):
        yy = g["label"].to_numpy()
        s = g["setfit_issues_PS"].to_numpy()
        r = dict(proj=proj, setfit=mf1(yy, s))
        for k, c in [("bragtag32", "bragtag_PS_32B_k12"), ("ft14", "ft_PA_14B"), ("bragtag7", "bragtag_PS_7B_k12")]:
            r[k] = mf1(yy, g[c].to_numpy())
        l = g["bragtag_PS_32B_k12"].to_numpy()
        r["dis_S_B32"] = (s != l).mean()
        r["oracle_S_B32"] = mf1(yy, np.where((s == yy) | (l == yy), yy, s))
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score the frontier-reader audit (Claude subagents, blind to gold) on the routed dev sample.

Inputs in results/issues11k/exploration/agentic/audit/:
  key.csv                      id, uid, proj, label (gold), setfit, margin
  ann_text_A.jsonl, ann_text_B.jsonl, ann_ctx_A.jsonl   one JSON object per item (annotator output)
Pilot arms (optional) from exploration/agentic/pilot/<tag>/p*.csv are scored on the same items.

Usage (lab machine): venv/bin/python scripts/experiments/agentic/score_audit.py --tag q14
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, LABELS, macro_f1  # noqa: E402


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == l) * np.mean(b == l) for l in LABELS)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def boot_acc(y, p, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    c = (np.asarray(y) == np.asarray(p)).astype(float)
    v = [c[rng.integers(0, len(c), len(c))].mean() for _ in range(B)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="q14")
    args = ap.parse_args()
    d = EXP / "audit"
    key = pd.read_csv(d / "key.csv").set_index("id")
    ann = {}
    for name in ["text_A", "text_B", "ctx_A"]:
        f = d / f"ann_{name}.jsonl"
        if f.exists():
            rows = [json.loads(l) for l in open(f) if l.strip()]
            a = pd.DataFrame(rows).set_index("id")
            missing = set(key.index) - set(a.index)
            if missing:
                print(f"{name}: {len(missing)} items missing")
            ann[name] = a.reindex(key.index)
    y = key["label"].to_numpy()
    print(f"items: {len(key)}; gold mix {key.label.value_counts().to_dict()}")
    rows = []

    def add(name, p):
        p = np.asarray(p, dtype=object)
        ok = pd.notna(p)
        lo, hi = boot_acc(y[ok], p[ok])
        rows.append(dict(reader=name, n=int(ok.sum()), acc=np.mean(y[ok] == p[ok]), acc_lo=lo, acc_hi=hi,
                         macroF1=macro_f1(y[ok], p[ok]), q2bug=np.mean(p[ok][y[ok] == "question"] == "bug")))

    add("SetFit-PS (dev)", key["setfit"].to_numpy())
    for name, a in ann.items():
        add(f"Claude {name}", a["label"].to_numpy())
    pil = EXP / "pilot" / args.tag
    for arm in ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"]:
        f = pil / f"{arm}.csv"
        if f.exists():
            pr = pd.read_csv(f).set_index("uid")
            add(f"Qwen-14B {arm}", pr.loc[key["uid"], "pred"].to_numpy())
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))

    if "text_A" in ann and "text_B" in ann:
        a, b = ann["text_A"]["label"].to_numpy(), ann["text_B"]["label"].to_numpy()
        ok = pd.notna(a) & pd.notna(b)
        print(f"\ntext A vs B: agreement {np.mean(a[ok] == b[ok]):.3f}, Cohen kappa {kappa(a[ok], b[ok]):.3f}")
        both = ok & (a == b)
        print(f"  both agree on {both.sum()} items; of those, agree with gold {np.mean(a[both] == y[both]):.3f}")
        agree_wrong = both & (a != y)
        print(f"  both agree AND differ from gold: {agree_wrong.sum()} items "
              f"(label-noise / convention candidates); gold->reader: "
              f"{pd.Series([f'{g}->{r}' for g, r in zip(y[agree_wrong], a[agree_wrong])]).value_counts().to_dict()}")
        amb = ann["text_A"]["ambiguous"].astype(bool).to_numpy() | ann["text_B"]["ambiguous"].astype(bool).to_numpy()
        print(f"  flagged ambiguous by either: {amb.mean():.3f}; among agree-and-wrong: {amb[agree_wrong].mean() if agree_wrong.any() else float('nan'):.3f}")
        tm = ann["text_A"]["template_mismatch"].astype(bool).to_numpy()
        print(f"  template_mismatch (A): {tm.mean():.3f}; gold mix of those {pd.Series(y[tm]).value_counts().to_dict()}")
    for name, a in ann.items():
        conf = a["confidence"].to_numpy()
        s = f"{name}: accuracy by confidence"
        for c in ["high", "medium", "low"]:
            m = conf == c
            if m.any():
                s += f" | {c} n={m.sum()} acc={np.mean(a['label'].to_numpy()[m] == y[m]):.2f}"
        print(s)
    if "ctx_A" in ann:
        c = ann["ctx_A"]["label"].to_numpy()
        s = key["setfit"].to_numpy()
        print(f"\nctx reader vs SetFit: agrees with SetFit on {np.mean(c == s):.3f}; "
              f"fixes {np.sum((s != y) & (c == y))}, breaks {np.sum((s == y) & (c != y))}")
    t.to_csv(d / "audit_scores.csv", index=False)


if __name__ == "__main__":
    main()

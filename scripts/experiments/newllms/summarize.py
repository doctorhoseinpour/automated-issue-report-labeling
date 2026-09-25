#!/usr/bin/env python3
"""Collect the newllms test evaluations and validation curves into markdown tables (lab machine).

  python summarize.py            -> prints the tables used in docs/NEWLLMS_STUDY.md
"""
from __future__ import annotations

import json

import pandas as pd

from nm_common import MODELS, NM, NM_FEATS, NM_GEN

ROWS = [("zs", "ZS (K=0, generation)"), ("zsc", "ZS, constrained"), ("rag", "RAG@K* (generation)"),
        ("ragc", "RAG@K*, constrained"), ("stateknn", "K: decision-state kNN@9"), ("readout", "R: read-out")]


def cis(name):
    f = NM / "test_eval" / f"{name}_cis.csv"
    return pd.read_csv(f).set_index("baseline") if f.exists() else None


def fmt_ci(c, base):
    if c is None or base not in c.index:
        return ""
    r = c.loc[base]
    return f"{r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"


def md(df):
    """Markdown table without the optional tabulate dependency."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    out = []
    for tag, m in MODELS.items():
        kstar = None
        for f in sorted(NM_GEN.glob(f"{tag}_test_k*.json")):
            k = int(f.stem.split("_test_k")[1])
            if k > 0:
                kstar = k
        out.append(f"\n**{m['name']}** (K* = {kstar})\n")
        out.append("| Method | Macro F1 | F1 bug / feat / question | q→bug | Invalid | vs RAG@K* | vs SetFit-PS | projects ≥ SetFit |")
        out.append("|---|---|---|---|---|---|---|---|")
        for key, label in ROWS:
            name = f"{key}_{tag}"
            f = NM / "test_eval" / f"{name}_eval.json"
            if not f.exists():
                continue
            e = json.load(open(f))
            c = cis(name)
            pc = e["per_class_f1"]
            vs_rag = "—" if key == "rag" else fmt_ci(c, f"RAG@K* {tag}")
            out.append(f"| {label} | {e['f1_macro']:.4f} | {pc[0]:.3f} / {pc[1]:.3f} / {pc[2]:.3f} | {e['q_to_bug']:.3f} | "
                       f"{e['invalid_rate']:.4f} | {vs_rag} | {fmt_ci(c, 'SetFit-PS (issues)')} | {e['projects_ge_setfit']}/11 |")
    print("\n".join(out))

    print("\n**Validation curves (val495, macro F1, generation; constrained in parentheses)**\n")
    rows = []
    for tag, m in MODELS.items():
        f = NM / "val" / f"{tag}_curve.csv"
        if f.exists():
            c = pd.read_csv(f)
            rows.append({"model": m["name"], **{f"K={int(r.k)}": f"{r.macro_f1:.3f} ({r.constrained_f1:.3f})" for r in c.itertuples()}})
    if rows:
        print(md(pd.DataFrame(rows)))

    print("\n**GPU cost (A100-40GB, per-item compute summed over shards, model load excluded)**\n")
    rows = []
    for tag, m in MODELS.items():
        r = {"model": m["name"]}
        f = NM_FEATS / f"{tag}_k0.json"
        if f.exists():
            j = json.load(open(f))
            r["states (6,600)"] = f"{j['compute_s'] / 60:.1f} min, {j['gpu_peak_mb'] / 1024:.1f} GB"
        for run in sorted(NM_GEN.glob(f"{tag}_*.json")):
            j = json.load(open(run))
            r[run.stem.replace(f'{tag}_', '')] = f"{j['compute_s'] / 60:.1f} min, {j['gpu_peak_mb'] / 1024:.1f} GB"
        rows.append(r)
    print(md(pd.DataFrame(rows)))


if __name__ == "__main__":
    main()

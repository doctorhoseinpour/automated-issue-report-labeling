#!/usr/bin/env python3
"""Final TEST evaluation of a candidate (reads test labels; every call is logged).

Input: a predictions CSV with one row per test issue in the paper's global test
order (test_idx 0..3299 = agnostic test_split.csv order) in the RAGTAG schema.
Scores it with evaluate.py's own evaluate_predictions(), then paired issue-level
bootstrap CIs (2,000 resamples) of candidate minus each baseline, per-class F1,
per-project macro F1 and wins/losses vs SetFit-PS per project.

Every run appends to results/issues11k/exploration/rag_next/test_eval_log.csv.

Usage (lab machine, repo root):
  venv/bin/python scripts/experiments/rag_next/final_eval.py <preds.csv> <name> [--size 7B]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2]))
from common import EXP, LABELS, RES, boot_diff, macro_f1, per_class_f1  # noqa: E402
from evaluate import evaluate_predictions  # noqa: E402

L2I = {l: i for i, l in enumerate(LABELS)}
BEST_BRAG = {"3B": 6, "7B": 12, "14B": 15, "32B": 12}
BEST_RAG = {"3B": 3, "7B": 6, "14B": 12, "32B": 12}


def enc(series):
    return np.array([L2I.get(str(x).strip().lower(), -1) for x in series])


def centering_preds(test_split):
    """The other session's centered RAGTAG-PS (Qwen-7B, k=9) run, pooled into global order."""
    base = Path.home() / "center_probe_20260924" / "preds_centered"
    if not base.exists():
        return None
    proj = test_split["repo"].str.replace("/", "_", n=1)
    out = np.full(len(test_split), -1)
    for p in sorted(proj.unique()):
        f = base / p / "preds_k9.csv"
        if not f.exists():
            return None
        d = pd.read_csv(f, usecols=["test_idx", "predicted_label"]).sort_values("test_idx")
        idx = np.where(proj.to_numpy() == p)[0]
        assert len(idx) == len(d)
        out[idx] = enc(d["predicted_label"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("name")
    ap.add_argument("--size", default=None, help="Qwen size for matched BRAGTAG/RAGTAG baselines")
    args = ap.parse_args()

    test_split = pd.read_csv(RES / "agnostic" / "neighbors" / "test_split.csv", keep_default_na=False)
    d = pd.read_csv(args.preds, keep_default_na=False)
    assert len(d) == 3300 and (d["test_idx"].to_numpy() == np.arange(3300)).all()
    y = enc(test_split["labels"])
    assert (enc(d["ground_truth"]) == y).all()
    p = enc(d["predicted_label"])
    ev = evaluate_predictions(d, model_name=args.name)

    m = pd.read_parquet(EXP / "headroom" / "master_preds.parquet")
    base = {"SetFit-PS (issues)": m["setfit_issues_PS"], "SetFit-PS (mpnet)": m["setfit_mpnet_PS"],
            "FT-PA-14B": m["ft_PA_14B"], "BRAGTAG-PS-32B (k12)": m["bragtag_PS_32B_k12"]}
    if args.size:
        base[f"BRAGTAG-PS-{args.size} (k{BEST_BRAG[args.size]})"] = m[f"bragtag_PS_{args.size}_k{BEST_BRAG[args.size]}"]
        base[f"RAGTAG-PS-{args.size} (k{BEST_RAG[args.size]})"] = m[f"ragtag_PS_{args.size}_k{BEST_RAG[args.size]}"]
    cen = centering_preds(test_split)
    rows = []
    print(f"\n=== {args.name}: macro F1 {ev['f1_macro']:.4f} (evaluate.py) | recomputed {macro_f1(y, p):.4f} | "
          f"invalid {ev['invalid_rate']:.4f}")
    f = per_class_f1(y, p)
    print(f"    per-class F1 bug/feat/q: {f[0]:.3f} / {f[1]:.3f} / {f[2]:.3f}; "
          f"q->bug {np.mean(p[y == 2] == 0):.3f}; pred bug share {np.mean(p == 0):.3f}")
    comps = list(base.items()) + ([("Centered RAGTAG-PS-7B (k9, other session)", cen)] if cen is not None else [])
    for bname, bp in comps:
        bp = enc(bp) if not isinstance(bp, np.ndarray) else bp
        diff, lo, hi = boot_diff(y, bp, p, B=2000, seed=0)
        print(f"    vs {bname:42s} base={macro_f1(y, bp):.4f}  diff={diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        rows.append({"baseline": bname, "base_f1": macro_f1(y, bp), "diff": diff, "ci_lo": lo, "ci_hi": hi})
    # per project
    proj = test_split["repo"].to_numpy()
    sf = enc(m["setfit_issues_PS"])
    pp = []
    for pr in sorted(np.unique(proj)):
        k = proj == pr
        pp.append({"project": pr, "cand": macro_f1(y[k], p[k]), "setfit_ps": macro_f1(y[k], sf[k])})
    pp = pd.DataFrame(pp)
    pp["diff"] = pp["cand"] - pp["setfit_ps"]
    print(pp.round(3).to_string(index=False))
    print(f"    projects >= SetFit-PS: {(pp['diff'] >= 0).sum()}/11")

    out_dir = EXP / "test_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"{args.name}_cis.csv", index=False)
    pp.to_csv(out_dir / f"{args.name}_per_project.csv", index=False)
    pd.DataFrame([ev]).to_csv(out_dir / f"{args.name}_eval.csv", index=False)
    log = EXP / "test_eval_log.csv"
    entry = pd.DataFrame([{"time": dt.datetime.now().isoformat(timespec="seconds"), "name": args.name,
                           "preds": args.preds, "macro_f1": ev["f1_macro"], "invalid_rate": ev["invalid_rate"]}])
    entry.to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"    logged test evaluation #{len(pd.read_csv(log))} -> {log}")


if __name__ == "__main__":
    main()

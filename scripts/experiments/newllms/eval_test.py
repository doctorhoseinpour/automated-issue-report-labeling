#!/usr/bin/env python3
"""Final TEST evaluation for the newllms study (reads test labels; every call is logged).

Modeled on rag_next/final_eval.py (not modified). Input: a 3,300-row predictions CSV in the
paper's global test order and RAGTAG schema. Scores it with evaluate.py's own
evaluate_predictions(), then per-class F1, q->bug, invalid rate, paired issue-level bootstrap
CIs (2,000 resamples) of candidate minus each baseline, and per-project macro F1 vs SetFit-PS.

Fixed baselines: SetFit-PS (issue-adapted MPNet), LoRA FT-PA-14B, BRAGTAG-PS-32B (k12),
RAGTAG-PS-32B (k12) from the archival predictions; rag_next R-32B and K-32B. Extra baselines
(e.g. the same model's RAG@K* and zero-shot) via --vs NAME=path.csv.

Appends to results/issues11k/exploration/newllms/test_eval_log.csv.

  python eval_test.py <preds.csv> <name> [--vs "RAG-qw35_9b=.../rag_qw35_9b.csv" ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd

from nm_common import LABELS, NM, RES, RN, boot_diff, macro_f1, per_class_f1
from evaluate import evaluate_predictions

L2I = {l: i for i, l in enumerate(LABELS)}


def enc(series):
    return np.array([L2I.get(str(x).strip().lower(), -1) for x in series])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("name")
    ap.add_argument("--vs", action="append", default=[], help="NAME=path.csv extra baseline")
    args = ap.parse_args()

    test_split = pd.read_csv(RES / "agnostic" / "neighbors" / "test_split.csv", keep_default_na=False)
    d = pd.read_csv(args.preds, keep_default_na=False)
    assert len(d) == 3300 and (d["test_idx"].to_numpy() == np.arange(3300)).all()
    y = enc(test_split["labels"])
    assert (enc(d["ground_truth"]) == y).all()
    p = enc(d["predicted_label"])
    ev = evaluate_predictions(d, model_name=args.name)

    m = pd.read_parquet(RN / "headroom" / "master_preds.parquet")
    base = {"SetFit-PS (issues)": enc(m["setfit_issues_PS"]), "FT-PA-14B": enc(m["ft_PA_14B"]),
            "BRAGTAG-PS-32B (k12)": enc(m["bragtag_PS_32B_k12"]), "RAGTAG-PS-32B (k12)": enc(m["ragtag_PS_32B_k12"])}
    for nm, f in [("R-32B (Qwen2.5 read-out)", "readout_32B.csv"), ("K-32B (Qwen2.5 kNN vote)", "stateknn_32B.csv")]:
        base[nm] = enc(pd.read_csv(RN / "test_preds" / f, keep_default_na=False)["predicted_label"])
    for s in args.vs:
        nm, path = s.split("=", 1)
        base[nm] = enc(pd.read_csv(path, keep_default_na=False)["predicted_label"])

    f = per_class_f1(y, p)
    print(f"\n=== {args.name}: macro F1 {ev['f1_macro']:.4f} (evaluate.py) | recomputed {macro_f1(y, p):.4f} | "
          f"invalid {ev['invalid_rate']:.4f}")
    print(f"    per-class F1 bug/feat/q: {f[0]:.3f} / {f[1]:.3f} / {f[2]:.3f}; q->bug {np.mean(p[y == 2] == 0):.3f}; "
          f"pred bug share {np.mean(p == 0):.3f}")
    rows = []
    for bname, bp in base.items():
        diff, lo, hi = boot_diff(y, bp, p, B=2000, seed=0)
        print(f"    vs {bname:34s} base={macro_f1(y, bp):.4f}  diff={diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        rows.append({"baseline": bname, "base_f1": macro_f1(y, bp), "diff": diff, "ci_lo": lo, "ci_hi": hi})
    proj = test_split["repo"].to_numpy()
    sf = base["SetFit-PS (issues)"]
    pp = pd.DataFrame([{"project": pr, "cand": macro_f1(y[proj == pr], p[proj == pr]),
                        "setfit_ps": macro_f1(y[proj == pr], sf[proj == pr])} for pr in sorted(np.unique(proj))])
    pp["diff"] = pp["cand"] - pp["setfit_ps"]
    print(pp.round(3).to_string(index=False))
    print(f"    projects >= SetFit-PS: {(pp['diff'] >= 0).sum()}/11")

    out = NM / "test_eval"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f"{args.name}_cis.csv", index=False)
    pp.to_csv(out / f"{args.name}_per_project.csv", index=False)
    summ = dict(ev, per_class_f1=f, q_to_bug=float(np.mean(p[y == 2] == 0)), projects_ge_setfit=int((pp["diff"] >= 0).sum()))
    json.dump(summ, open(out / f"{args.name}_eval.json", "w"), indent=1, default=float)
    log = NM / "test_eval_log.csv"
    pd.DataFrame([{"time": dt.datetime.now().isoformat(timespec="seconds"), "name": args.name, "preds": args.preds,
                   "macro_f1": ev["f1_macro"], "invalid_rate": ev["invalid_rate"]}]).to_csv(
        log, mode="a", header=not log.exists(), index=False)
    print(f"    logged test evaluation #{len(pd.read_csv(log))} -> {log}")


if __name__ == "__main__":
    main()

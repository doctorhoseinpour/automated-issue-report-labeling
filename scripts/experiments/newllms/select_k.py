#!/usr/bin/env python3
"""Choose RAGTAG's K per model on the validation split (val495; lab machine).

Rule (pre-declared in the notebook before any run): K* = argmax of validation macro F1 of the
paper's protocol (greedy generation + parse_label, invalid = wrong) over K >= 1; ties -> the
smaller K. K = 0 is the zero-shot baseline and is not a candidate.

Also reports, per K: per-class F1, invalid rate, the constrained-decoding F1 (argmax of the
first-step label log-probs), prompt length, GPU seconds; and a paired bootstrap over the 495
issues (2,000 resamples): how often each K is the argmax, and the 95% CI of F1(K*) - F1(K).
Writes val/<tag>_curve.csv and configs/rag_<tag>.json.

  python select_k.py qw35_9b [gm4_12b mi3_8b]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nm_common import LAB2ID, LABELS, NM, NM_GEN, VAL_FILE, load_pool, macro_f1, per_class_f1

HERE = Path(__file__).resolve().parent
B = 2000


def enc(s):
    return np.array([LAB2ID.get(str(x).strip().lower(), -1) for x in s])


def main():
    pool = load_pool().set_index("uid")
    val = pd.read_csv(VAL_FILE)
    out_dir = NM / "val"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag in sys.argv[1:]:
        df = pd.read_parquet(NM_GEN / f"{tag}_val_sweep.parquet")
        uids = np.sort(val["uid"].to_numpy())
        y = pool.loc[uids, "label"].map(LAB2ID).to_numpy()
        P, Pc, rows = {}, {}, []
        for k, g in df.groupby("k"):
            g = g.set_index("uid").loc[uids]
            p = enc(g["predicted_label"])
            pc = g[["lp_bug", "lp_feature", "lp_question"]].to_numpy().argmax(1)
            P[int(k)], Pc[int(k)] = p, pc
            f = per_class_f1(y, p)
            rows.append({"k": int(k), "macro_f1": macro_f1(y, p), "f1_bug": f[0], "f1_feature": f[1], "f1_question": f[2],
                         "invalid": float(np.mean(p < 0)), "q_to_bug": float(np.mean(p[y == 2] == 0)),
                         "constrained_f1": macro_f1(y, pc), "prompt_tokens": float(g.prompt_tokens.mean()),
                         "truncated": float(g.truncated.mean()), "gpu_s": float(g.time_s.sum())})
        cur = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
        cand = [k for k in sorted(P) if k >= 1]
        best = max(cand, key=lambda k: (round(macro_f1(y, P[k]), 12), -k))
        # paired bootstrap over validation issues
        rng = np.random.default_rng(0)
        wins = {k: 0 for k in cand}
        diffs = {k: [] for k in sorted(P)}
        for _ in range(B):
            ii = rng.integers(0, len(y), len(y))
            s = {k: macro_f1(y[ii], P[k][ii]) for k in sorted(P)}
            wins[max(cand, key=lambda k: (round(s[k], 12), -k))] += 1
            for k in P:
                diffs[k].append(s[best] - s[k])
        cur["boot_win_freq"] = [wins.get(k, np.nan) / B if k >= 1 else np.nan for k in cur.k]
        cur["d_best_lo"] = [np.percentile(diffs[k], 2.5) for k in cur.k]
        cur["d_best_hi"] = [np.percentile(diffs[k], 97.5) for k in cur.k]
        cur.to_csv(out_dir / f"{tag}_curve.csv", index=False)
        print(f"\n=== {tag}: validation curve (n={len(y)})")
        print(cur.round(4).to_string(index=False))
        print(f"K* = {best} (val macro F1 {macro_f1(y, P[best]):.4f}; bootstrap win frequency {wins[best] / B:.2f})")
        cfg = {"name": f"rag_{tag}", "tag": tag, "k_star": int(best), "rule": "argmax val macro F1 over K>=1, ties -> smaller K",
               "val_file": str(VAL_FILE.name), "val_n": int(len(y)), "val_macro_f1": macro_f1(y, P[best]),
               "val_zero_shot_f1": macro_f1(y, P[0]) if 0 in P else None,
               "boot_win_freq": {int(k): wins[k] / B for k in cand}}
        (HERE / "configs").mkdir(exist_ok=True)
        json.dump(cfg, open(HERE / "configs" / f"rag_{tag}.json", "w"), indent=1)
        print("wrote", HERE / "configs" / f"rag_{tag}.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the frontier-reader audit on routed dev issues (no gold labels in the item files).

Two conditions over the same sample of routed dev issues:
  text : rubric + target issue only (what a careful reader recovers from the report)
  ctx  : exactly the p2 prompt of adjudicate.py (rubric + 3 most similar past issues per
         label + SetFit scores + target), i.e. the single-prompt adjudicator's information
Writes results/issues11k/exploration/agentic/audit/{items_text.jsonl, items_ctx.jsonl, key.csv}.
The annotators (Claude subagents, no tools other than reading the item file) never see key.csv.

Usage (lab machine):  venv/bin/python scripts/experiments/agentic/audit_prep.py --n 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
from common import EXP, load_pool, load_setfit_dev, routed_uids, snapshot_inputs  # noqa: E402


class _TokOnly:
    """Just enough of adjudicate.LM for Ctx: a tokenizer and a chat renderer."""

    def __init__(self, name):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_route", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tokenizer", default="unsloth/Qwen2.5-14B-Instruct-bnb-4bit")
    args = ap.parse_args()
    snapshot_inputs()
    from adjudicate import Ctx, rubric

    out = EXP / "audit"
    out.mkdir(parents=True, exist_ok=True)
    lm = _TokOnly(args.tokenizer)
    ctx = Ctx(load_pool(), lm, args.n_route)
    rng = np.random.default_rng(args.seed)
    uids = rng.choice(ctx.uids, size=args.n, replace=False)
    sf = load_setfit_dev().set_index("uid")
    key = []
    with open(out / "items_text.jsonl", "w") as ft, open(out / "items_ctx.jsonl", "w") as fc:
        for i, u in enumerate(uids, 1):
            r = ctx.row(u)
            iid = f"A{i:03d}"
            tgt, _ = ctx.v.target(r["title"], r["body"])
            ft.write(json.dumps({"id": iid, "repo": r["repo"], "instructions": rubric(r["repo"]), "issue": tgt}) + "\n")
            fc.write(json.dumps({"id": iid, "repo": r["repo"], "instructions": rubric(r["repo"]),
                                 "context_and_issue": ctx.user_msg(u, with_hint=True)}) + "\n")
            key.append(dict(id=iid, uid=int(u), proj=r["proj"], label=r["label"], setfit=sf.loc[int(u), "pred"],
                            margin=float(sf.loc[int(u), "margin"])))
    pd.DataFrame(key).to_csv(out / "key.csv", index=False)
    print(f"wrote {len(key)} items to {out}; label mix {pd.DataFrame(key).label.value_counts().to_dict()}")


if __name__ == "__main__":
    main()

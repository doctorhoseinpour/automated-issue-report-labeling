#!/usr/bin/env python3
"""Per-project train/eval CSVs for the dev phase (fit on inner, score on dev) in the
schema run_setfit.py / run_transformer_ft.py expect (title, body, labels, + uid)."""
from common import SPLITS, load_pool

pool = load_pool()
for proj, g in pool.groupby("proj"):
    d = SPLITS / "ps" / proj
    d.mkdir(parents=True, exist_ok=True)
    for role in ["inner", "dev"]:
        x = g[g.role == role][["uid", "repo", "created_at", "label", "title", "body"]].rename(columns={"label": "labels"})
        x.to_csv(d / f"{role}.csv", index=False)
for role in ["inner", "dev"]:
    x = pool[pool.role == role][["uid", "repo", "created_at", "label", "title", "body"]].rename(columns={"label": "labels"})
    x.to_csv(SPLITS / f"{role}.csv", index=False)
print("ok")

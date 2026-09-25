#!/usr/bin/env python3
"""Dev-phase SetFit (project-specific): train on role == inner, predict role == dev.

Same recipe as run_setfit.py (the 0.810 bar): body Collab-uniba/github-issues-mpnet-st-e10,
batch 16, 1 epoch, num_iterations 20, LogisticRegression head, seed 42, "Title:\\nBody:" text.
One change for the shared 24 GB GPU: gradient checkpointing on the body, which recomputes
activations in the backward pass instead of storing them (same math; peak memory drops from
~21.5 GB to a few GB).

Reads the dev protocol of the rag_next study (splits/pool.csv: dev = newest 30 train issues
per (repo, label)). Never touches role == test.

Writes results/issues11k/exploration/agentic/dev/setfit_issues_PS/{preds.csv, cost.json}.

Usage (lab machine, repo root):
  venv-setfit/bin/python scripts/experiments/agentic/setfit_dev.py [--projects a,b] [--no_ckpt]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
RES = Path(os.environ.get("RESULTS_DIR", REPO / "results" / "issues11k"))
POOL = RES / "exploration" / "rag_next" / "splits" / "pool.csv"
OUT = RES / "exploration" / "agentic" / "dev" / "setfit_issues_PS"
LABELS = ["bug", "feature", "question"]


def text(t, b):
    return f"Title: {t}\nBody: {b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body_model", default="Collab-uniba/github-issues-mpnet-st-e10")
    ap.add_argument("--projects", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_ckpt", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    from transformers import set_seed

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(POOL, keep_default_na=False)
    pool = pool[pool.role.isin(["inner", "dev"])]
    projs = sorted(pool.proj.unique()) if not args.projects else args.projects.split(",")

    rows, cost = [], {}
    for proj in projs:
        f = out / f"preds_{proj}.csv"
        if f.exists():
            print("SKIP", proj)
            rows.append(pd.read_csv(f))
            continue
        tr = pool[(pool.proj == proj) & (pool.role == "inner")]
        dv = pool[(pool.proj == proj) & (pool.role == "dev")]
        set_seed(args.seed)
        model = SetFitModel.from_pretrained(args.body_model, labels=LABELS)
        if not args.no_ckpt:
            model.model_body[0].auto_model.gradient_checkpointing_enable()
        torch.cuda.reset_peak_memory_stats()
        ds = Dataset.from_dict({"text": [text(t, b) for t, b in zip(tr.title, tr.body)], "label": tr.label.tolist()})
        targs = TrainingArguments(output_dir=str(out / "_trainer_output"), batch_size=16, num_epochs=1,
                                  num_iterations=20, seed=args.seed, report_to="none")
        t0 = time.time()
        Trainer(model=model, args=targs, train_dataset=ds, column_mapping={"text": "text", "label": "label"}).train()
        t_train = time.time() - t0
        t1 = time.time()
        P = model.predict_proba([text(t, b) for t, b in zip(dv.title, dv.body)], batch_size=32)
        P = np.asarray(P.cpu() if torch.is_tensor(P) else P)
        t_inf = time.time() - t1
        cls = [str(c) for c in np.asarray(model.model_head.classes_).tolist()]
        P = P[:, [cls.index(l) for l in LABELS]]
        d = pd.DataFrame({"uid": dv.uid.values, "proj": proj, "label": dv.label.values,
                          "pred": [LABELS[i] for i in P.argmax(1)],
                          **{f"p_{l}": P[:, i] for i, l in enumerate(LABELS)}})
        d.to_csv(f, index=False)
        rows.append(d)
        cost[proj] = dict(n_train=len(tr), n_dev=len(dv), train_s=round(t_train, 1), infer_s=round(t_inf, 2),
                          peak_mb=round(torch.cuda.max_memory_allocated() / 2**20))
        acc = (d.pred == d.label).mean()
        print(f"{proj}: train {t_train:.0f}s infer {t_inf:.1f}s peak {cost[proj]['peak_mb']} MB dev acc {acc:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    allp = pd.concat(rows, ignore_index=True)
    allp.to_csv(out / "preds.csv", index=False)
    old = json.load(open(out / "cost.json")) if (out / "cost.json").exists() else {}
    old.update(cost)
    json.dump(dict(old, body_model=args.body_model, grad_ckpt=not args.no_ckpt, seed=args.seed), open(out / "cost.json", "w"), indent=1)
    from sklearn.metrics import f1_score
    print("dev macro F1:", round(f1_score(allp.label, allp.pred, labels=LABELS, average="macro"), 4), "n", len(allp))


if __name__ == "__main__":
    main()

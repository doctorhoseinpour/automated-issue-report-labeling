#!/usr/bin/env python3
"""Collect every existing per-issue prediction on the 11k test split into one
aligned table (global test order of agnostic/neighbors/test_split.csv).

Read-only over results/issues11k; writes under
results/issues11k/exploration/rag_next/headroom/.

Outputs
  master_preds.parquet   one row per test issue (3,300), one column per
                         (method, setting, model, k) prediction
  setfit_probs.parquet   SetFit class probabilities (PS/PA x 2 bodies)
  neighbors_ps.parquet   top-30 PS neighbor labels/sims per test issue
  neighbors_pa.parquet   top-30 PA neighbor labels/sims per test issue

Usage (lab machine, repo root):
  venv/bin/python scripts/experiments/rag_next/collect_preds.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RES = Path(os.environ.get("RESULTS_DIR", REPO / "results" / "issues11k"))
OUT = RES / "exploration" / "rag_next" / "headroom"
LABELS = ["bug", "feature", "question"]

QWEN = {
    "3B": "unsloth_Qwen2_5_3B_Instruct_bnb_4bit",
    "7B": "unsloth_Qwen2_5_7B_Instruct_bnb_4bit",
    "14B": "unsloth_Qwen2_5_14B_Instruct_bnb_4bit",
    "32B": "unsloth_Qwen2_5_32B_Instruct_bnb_4bit",
}
KS = [0, 1, 3, 6, 9, 12, 15]


def projects() -> list[str]:
    return sorted(p.name for p in (RES / "project_specific").iterdir() if p.is_dir())


def base_frame() -> pd.DataFrame:
    df = pd.read_csv(RES / "agnostic" / "neighbors" / "test_split.csv")
    df["proj"] = df["repo"].str.replace("/", "_", n=1)
    df["local_idx"] = df.groupby("proj").cumcount()
    df["gidx"] = np.arange(len(df))
    df["label"] = df["labels"].astype(str).str.lower().str.strip()
    return df


def canon(x) -> str:
    x = str(x).strip().lower()
    return x if x in LABELS else "invalid"


def load_ps(rel: str, fname: str, base: pd.DataFrame, extra: list[str] | None = None):
    """rel is the per-project subpath, e.g. '<model>/ragtag/predictions'."""
    parts = []
    for proj in projects():
        f = RES / "project_specific" / proj / rel / fname
        if not f.exists():
            return None
        cols = ["test_idx", "predicted_label"] + (extra or [])
        d = pd.read_csv(f, usecols=lambda c: c in cols)
        d["proj"] = proj
        parts.append(d)
    d = pd.concat(parts, ignore_index=True).rename(columns={"test_idx": "local_idx"})
    m = base[["proj", "local_idx", "gidx"]].merge(d, on=["proj", "local_idx"], how="left")
    assert len(m) == len(base) and m["predicted_label"].notna().all(), rel
    return m.sort_values("gidx").reset_index(drop=True)


def load_pa(rel: str, fname: str, base: pd.DataFrame, extra: list[str] | None = None):
    f = RES / "agnostic" / rel / fname
    if not f.exists():
        return None
    cols = ["test_idx", "predicted_label", "ground_truth"] + (extra or [])
    d = pd.read_csv(f, usecols=lambda c: c in cols)
    assert len(d) == len(base), f
    d = d.sort_values("test_idx").reset_index(drop=True)
    assert (d["test_idx"].values == base["gidx"].values).all()
    if "ground_truth" in d:
        assert (d["ground_truth"].astype(str).str.lower().values == base["label"].values).all(), f
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = base_frame()
    cols = {}

    def put(name, d):
        if d is None:
            print(f"  missing: {name}")
            return
        cols[name] = d["predicted_label"].map(canon).values

    for size, tag in QWEN.items():
        for k in KS:
            kl = "zero_shot" if k == 0 else f"k{k}"
            put(f"ragtag_PS_{size}_k{k}", load_ps(f"{tag}/ragtag/predictions", f"preds_{kl}.csv", base))
            put(f"ragtag_PA_{size}_k{k}", load_pa(f"{tag}/ragtag/predictions", f"preds_{kl}.csv", base))
            if k > 0:
                put(f"bragtag_PS_{size}_k{k}",
                    load_ps(f"{tag}/ragtag_debias_m3/predictions", f"preds_{kl}.csv", base))
        put(f"ft_PA_{size}", load_pa(f"{tag}/finetune_fixed", "preds_finetune_fixed.csv", base))
        put(f"ft_PS_{size}", load_ps(f"{tag}/finetune_fixed", "preds_finetune_fixed.csv", base))

    for k in range(1, 31):
        put(f"vtag_PS_k{k}", load_ps("vtag/predictions", f"preds_k{k}.csv", base))
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30]:
        put(f"vtag_PA_k{k}", load_pa("vtag/predictions", f"preds_k{k}.csv", base))

    probs = {}
    for body, short in [("sentence-transformers_all-mpnet-base-v2", "mpnet"),
                        ("Collab-uniba_github-issues-mpnet-st-e10", "issues")]:
        for setting, loader in [("PS", load_ps), ("PA", load_pa)]:
            d = loader(f"{body}/setfit/predictions", "preds_setfit.csv", base, extra=["raw_output"])
            put(f"setfit_{short}_{setting}", d)
            if d is not None:
                p = d["raw_output"].map(json.loads)
                for lab in LABELS:
                    probs[f"setfit_{short}_{setting}_p_{lab}"] = p.map(lambda x: x[lab]).values
    for setting, loader in [("PS", load_ps), ("PA", load_pa)]:
        put(f"roberta_{setting}",
            loader("roberta-base/finetune_transformer/predictions", "preds_finetune_transformer.csv", base))

    for tag, short in [("unsloth_Meta_Llama_3_1_8B_Instruct_bnb_4bit", "llama8B"),
                       ("unsloth_Llama_3_2_3B_Instruct", "llama3B")]:
        for k in [1, 3, 6, 9]:
            put(f"ragtag_PS_{short}_k{k}", load_ps(f"{tag}/ragtag/predictions", f"preds_k{k}.csv", base))

    master = base[["gidx", "repo", "proj", "local_idx", "created_at", "label"]].copy()
    master["title_len"] = base["title"].fillna("").str.len()
    master["body_len"] = base["body"].fillna("").str.len()
    master = pd.concat([master, pd.DataFrame(cols)], axis=1)
    master.to_parquet(OUT / "master_preds.parquet", index=False)
    pd.concat([master[["gidx"]], pd.DataFrame(probs)], axis=1).to_parquet(
        OUT / "setfit_probs.parquet", index=False)
    print(f"master: {master.shape} -> {OUT/'master_preds.parquet'}")

    # --- neighbors (top-30) ---
    def nb_table(frame: pd.DataFrame, n_test: int) -> pd.DataFrame:
        frame = frame.sort_values(["test_idx", "neighbor_rank"])
        lab = np.asarray(frame["neighbor_label"].astype(str).str.lower().tolist(), dtype=object).reshape(n_test, 30)
        sim = np.asarray(frame["neighbor_similarity"].to_numpy(dtype=float)).reshape(n_test, 30)
        out = {}
        for r in range(30):
            out[f"l{r}"] = lab[:, r]
            out[f"s{r}"] = sim[:, r]
        return pd.DataFrame(out)

    pa = pd.read_csv(RES / "agnostic" / "neighbors" / "neighbors_k30.csv",
                     usecols=["test_idx", "neighbor_rank", "neighbor_label", "neighbor_similarity"])
    npa = nb_table(pa, len(base))
    npa.insert(0, "gidx", base["gidx"].values)
    npa.to_parquet(OUT / "neighbors_pa.parquet", index=False)

    parts = []
    for proj in projects():
        f = RES / "project_specific" / proj / "neighbors" / "neighbors_k30.csv"
        d = pd.read_csv(f, usecols=["test_idx", "neighbor_rank", "neighbor_label", "neighbor_similarity"])
        n = d["test_idx"].nunique()
        t = nb_table(d, n)
        t["proj"] = proj
        t["local_idx"] = np.arange(n)
        parts.append(t)
    nps = pd.concat(parts, ignore_index=True)
    nps = base[["proj", "local_idx", "gidx"]].merge(nps, on=["proj", "local_idx"]).sort_values("gidx")
    nps.drop(columns=["proj", "local_idx"]).to_parquet(OUT / "neighbors_ps.parquet", index=False)
    print("neighbors written")


if __name__ == "__main__":
    main()

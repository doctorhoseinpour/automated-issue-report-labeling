#!/usr/bin/env python3
"""Retrieval for the exploration, reproducing the paper's recipe
(all-MiniLM-L6-v2 on clean_text(title + " " + body), L2-normalised, cosine).

For each phase the "universe" and the index differ:
  phase dev : queries = inner (leave-one-out) + dev ; index = inner
  phase test: queries = train (leave-one-out) + test; index = train (= paper index)
Settings: PS (index restricted to the query's project) and PA (all projects).
Variants: raw (paper) and center (per-project mean-centering of query and index
with the index's project mean, then re-normalisation; the existing baseline
component from the 2026-09-22 probe, not a contribution of this study).

Writes features/nb_{setting}_{variant}_{phase}.npz with uids, nb (index uids,
ranked), sim (cosine in the variant's space), K = 50.
Also verifies that phase-test PS/PA raw neighbours for the test queries match
the paper's neighbors_k30.csv files.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import FEATS, RES, load_pool

K = 50
_ws = re.compile(r"\s+")


def clean(t: str) -> str:
    return _ws.sub(" ", str(t)).strip()


def embed_minilm(texts):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(16)
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    E = m.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def l2n(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def search(Q, X, q_uids, x_uids, k, exclude_self=True, q_ts=None, x_ts=None, causal_mask=None):
    """causal_mask (bool per query): for those queries only index items created strictly
    before the query are eligible (others get -inf); missing slots come back as uid -1."""
    S = Q @ X.T
    if exclude_self:
        same = q_uids[:, None] == x_uids[None, :]
        S = np.where(same, -np.inf, S)
    if causal_mask is not None:
        later = (x_ts[None, :] >= q_ts[:, None]) & causal_mask[:, None]
        S = np.where(later, -np.inf, S)
    k = min(k, X.shape[0] - 1)
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    ps = np.take_along_axis(S, part, 1)
    order = np.argsort(-ps, axis=1, kind="stable")
    idx = np.take_along_axis(part, order, 1)
    sims = np.take_along_axis(S, idx, 1)
    out = np.where(np.isfinite(sims), x_uids[idx], -1)
    return out, np.where(np.isfinite(sims), sims, 0.0)


def main():
    FEATS.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    emb_path = FEATS / "minilm_raw.npy"
    if emb_path.exists():
        E = np.load(emb_path)
    else:
        E = embed_minilm([clean(t + " " + b) for t, b in zip(pool.title, pool.body)])
        np.save(emb_path, E)
    uid = pool.uid.to_numpy()
    proj = pool.proj.to_numpy()

    phases = {
        "dev": (pool.role.isin(["inner", "dev"]).to_numpy(), (pool.role == "inner").to_numpy()),
        "test": (np.ones(len(pool), bool), (pool.split == "train").to_numpy()),
    }
    ts = pd.to_datetime(pool["created_at"], utc=True).astype("int64").to_numpy()
    train_role = {"dev": (pool.role == "inner").to_numpy(), "test": (pool.split == "train").to_numpy()}
    for phase, (qmask, imask) in phases.items():
      for causal in [False, True]:
        for variant in ["raw", "center"]:
            for setting in ["PS", "PA"]:
                out = FEATS / f"nb_{setting}_{variant}_{phase}{'_causal' if causal else ''}.npz"
                if out.exists():
                    continue
                NB = np.zeros((qmask.sum(), K), dtype=np.int64)
                SIM = np.zeros((qmask.sum(), K), dtype=np.float32)
                q_idx = np.where(qmask)[0]
                pos = {u: i for i, u in enumerate(uid[q_idx])}
                Ev = E.copy()
                if variant == "center":
                    for p in np.unique(proj):
                        mu = E[imask & (proj == p)].mean(0)
                        Ev[proj == p] = E[proj == p] - mu
                    Ev = l2n(Ev)
                groups = [(p, proj[q_idx] == p, imask & (proj == p)) for p in np.unique(proj)] \
                    if setting == "PS" else [("all", np.ones(len(q_idx), bool), imask)]
                for p, qsel, isel in groups:
                    qi = q_idx[qsel]
                    ii = np.where(isel)[0]
                    nb, s = search(Ev[qi], Ev[ii], uid[qi], uid[ii], K, q_ts=ts[qi], x_ts=ts[ii],
                                   causal_mask=train_role[phase][qi] if causal else None)
                    rows = [pos[u] for u in uid[qi]]
                    NB[rows], SIM[rows] = nb, s
                np.savez(out, uids=uid[q_idx], nb=NB, sim=SIM)
                print("wrote", out.name, NB.shape)

    # --- verify against the paper's neighbor files (test queries, raw) ---
    z = np.load(FEATS / "nb_PS_raw_test.npz")
    m = dict(zip(z["uids"], range(len(z["uids"]))))
    train_uid_by_tidx = pool[pool.split == "train"].set_index("tidx")["uid"]
    lab = pool.set_index("uid")["label"]
    agree = tot = 0
    for p in sorted(pool.proj.unique()):
        f = RES / "project_specific" / p / "neighbors" / "neighbors_k30.csv"
        d = pd.read_csv(f, usecols=["test_idx", "neighbor_rank", "neighbor_label", "neighbor_title"])
        te = pool[(pool.split == "test") & (pool.proj == p)].reset_index(drop=True)
        for ti, g in d.groupby("test_idx"):
            u = te.loc[ti, "uid"]
            mine = z["nb"][m[u], :30]
            titles = pool.set_index("uid").loc[mine, "title"].tolist()
            agree += sum(a == b for a, b in zip(titles, g.sort_values("neighbor_rank")["neighbor_title"].astype(str)))
            tot += 30
    print(f"PS raw test-neighbour title agreement with paper files (rank-exact): {agree/tot:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge run_llm.py shard parts (lab machine, after pull_nm.sh).

  states -> features/<tag>_k0.npz      rag_next features format (uids, logp, top1, h_L{l} of the
                                       read-out layers, prompt_tokens, truncated, query_truncated,
                                       n_demos) -- what components.probe / state_neighbors read
            features/<tag>_k0_all.npz  h_L{l} of every layer (dev-only depth analysis)
  gen    -> gen/<tag>_<run>.parquet    one row per (uid, k)
Plus a .json with the summed GPU compute time (per-item, excludes model load), the shards'
metadata and the peak memory. Asserts completeness: every expected item exactly once.

  python merge_parts.py qw35_9b states
  python merge_parts.py qw35_9b val_sweep
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from nm_common import MODELS, NM_FEATS, NM_GEN, NM_RAW, VAL_FILE, K_GRID, load_pool, readout_layers

EXPECTED = {  # run -> (query uids, ks)
    "states": ("inner,dev,test", [0]),
    "val_sweep": ("val", K_GRID),
    "test_k0": ("test", [0]),
}


def expected_items(run, kstar=None):
    pool = load_pool()
    if run.startswith("test_k") and run != "test_k0":
        uids, ks = pool.uid[pool.role == "test"], [int(run[len("test_k"):])]
    else:
        roles, ks = EXPECTED[run]
        uids = pd.read_csv(VAL_FILE)["uid"] if roles == "val" else pool.uid[pool.role.isin(roles.split(","))]
    return {(int(u), int(k)) for u in uids for k in ks}


def main():
    tag, run = sys.argv[1], sys.argv[2]
    d = NM_RAW / tag / run
    dones = sorted(d.glob("done_*.json"))
    metas = [json.load(open(f)) for f in dones]
    assert metas, f"no done markers in {d}"
    nsh = metas[0]["nshards"]
    assert len(metas) == nsh and all(m["nshards"] == nsh for m in metas), f"{len(metas)} of {nsh} shards done"
    want = expected_items(run)
    info = {k: v for k, v in metas[0].items() if k not in ("shard", "n_items", "n_parts", "gpu_peak_mb", "finished",
                                                          "node", "slurm_job", "load_s")}
    summary = dict(info, n_shards=nsh, gpus=sorted({m["gpu"] for m in metas}),
                   gpu_peak_mb=max(m["gpu_peak_mb"] for m in metas),
                   load_s_per_shard=[m["load_s"] for m in metas], shards=metas)

    if run == "states":
        parts = sorted(d.glob("part_*.npz"))
        zs = [np.load(f) for f in parts]
        uids = np.concatenate([z["uid"] for z in zs]).astype(np.int64)
        got = {(int(u), 0) for u in uids}
        assert len(uids) == len(got) == len(want) and got == want, (len(uids), len(got), len(want))
        o = np.argsort(uids)
        H = np.concatenate([z["H"].astype(np.float32 if z["H"].dtype == np.float32 else np.float16) for z in zs])[o]
        col = lambda c: np.concatenate([z[c] for z in zs])[o]  # noqa: E731
        n = MODELS[tag]["n_layers"]
        assert H.shape[1] == n
        base = dict(uids=uids[o], logp=np.stack([col("lp_bug"), col("lp_feature"), col("lp_question")], 1).astype(np.float32),
                    top1=col("top1").astype(np.int64), prompt_tokens=col("prompt_tokens"), truncated=col("truncated"),
                    query_truncated=col("query_truncated"), n_demos=col("n_demos"), capped=col("capped"))
        NM_FEATS.mkdir(parents=True, exist_ok=True)
        np.savez(NM_FEATS / f"{tag}_k0.npz", **base, **{f"h_L{l}": H[:, l - 1] for l in readout_layers(n)})
        np.savez(NM_FEATS / f"{tag}_k0_all.npz", uids=uids[o], **{f"h_L{l}": H[:, l - 1] for l in range(1, n + 1)})
        summary.update(n=int(len(uids)), compute_s=float(col("time_s").sum()), readout_layers=readout_layers(n),
                       dtype=str(H.dtype), top1_is_label=float(col("top1_is_label").mean()),
                       mean_prompt_tokens=float(col("prompt_tokens").mean()))
        json.dump(summary, open(NM_FEATS / f"{tag}_k0.json", "w"), indent=1)
        print(f"{tag} states: n={len(uids)} layers={n} dtype={H.dtype} compute={summary['compute_s']:.0f}s "
              f"top1_is_label={summary['top1_is_label']:.3f} peak={summary['gpu_peak_mb']:.0f}MB gpus={summary['gpus']}")
    else:
        df = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("part_*.parquet"))], ignore_index=True)
        got = set(zip(df.uid.astype(int), df.k.astype(int)))
        assert len(df) == len(got) == len(want) and got == want, (len(df), len(got), len(want))
        df = df.sort_values(["k", "uid"]).reset_index(drop=True)
        NM_GEN.mkdir(parents=True, exist_ok=True)
        df.to_parquet(NM_GEN / f"{tag}_{run}.parquet", index=False)
        summary.update(n=int(len(df)), compute_s=float(df.time_s.sum()),
                       per_k={int(k): {"n": int(len(g)), "compute_s": float(g.time_s.sum()),
                                       "mean_prompt_tokens": float(g.prompt_tokens.mean()),
                                       "invalid_rate": float((g.predicted_label == "invalid").mean()),
                                       "truncated": float(g.truncated.mean())} for k, g in df.groupby("k")})
        json.dump(summary, open(NM_GEN / f"{tag}_{run}.json", "w"), indent=1)
        print(f"{tag} {run}: n={len(df)} compute={summary['compute_s']:.0f}s peak={summary['gpu_peak_mb']:.0f}MB "
              f"gpus={summary['gpus']}")
        for k, v in summary["per_k"].items():
            print(f"   k={k:2d} n={v['n']} tok={v['mean_prompt_tokens']:.0f} invalid={v['invalid_rate']:.3f} "
                  f"trunc={v['truncated']:.2f} compute={v['compute_s']:.0f}s")


if __name__ == "__main__":
    main()

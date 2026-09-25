#!/usr/bin/env python3
"""Merge chunked llm_features.py outputs (same model/prompt, disjoint issues) into one file.
  python merge_features.py q32_k0 q32_k0_innerdev q32_k0_test
The merged .json keeps per-chunk metadata and sums the wall time."""
import json
import sys

import numpy as np

from common import FEATS

out, parts = sys.argv[1], sys.argv[2:]
zs = [np.load(FEATS / f"{p}.npz") for p in parts]
keys = zs[0].files
assert all(z.files == keys for z in zs)
merged = {k: np.concatenate([z[k] for z in zs]) for k in keys}
assert len(np.unique(merged["uids"])) == len(merged["uids"]), "overlapping chunks"
np.savez(FEATS / f"{out}.npz", **merged)
metas = [json.load(open(FEATS / f"{p}.json")) for p in parts]
meta = dict(metas[0])
meta.update(n=int(len(merged["uids"])), roles=sorted({r for m in metas for r in m["roles"]}),
            wall_time_s=round(sum(m["wall_time_s"] for m in metas), 1),
            gpu_peak_mb=max(m["gpu_peak_mb"] for m in metas), chunks=parts, chunk_meta=metas)
json.dump(meta, open(FEATS / f"{out}.json", "w"), indent=1)
print(f"merged {parts} -> {out}: n={meta['n']} wall={meta['wall_time_s']}s peak={meta['gpu_peak_mb']}MB")

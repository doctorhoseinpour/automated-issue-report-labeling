#!/usr/bin/env python3
"""Turn merged RAGTAG test generations into evaluate.py-ready CSVs (lab machine).

For gen/<tag>_<run>.parquet (run = test_k0 or test_k<K*>) writes, in the paper's global test
order and RAGTAG schema:
  test_preds/<zs|rag>_<tag>.csv    the paper's protocol: parsed greedy generation (invalid kept)
  test_preds/<zs|rag>c_<tag>.csv   constrained decoding: argmax of the first-step label log-probs
ground_truth is copied from the pool only to fill the schema. Also checks that the zero-shot
generation prompt is the states prompt (first-step log-probs = states log-probs).

  python make_test_csvs.py qw35_9b test_k0
  python make_test_csvs.py qw35_9b test_k12
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from nm_common import LABELS, NM, NM_FEATS, NM_GEN, load_pool


def main():
    tag, run = sys.argv[1], sys.argv[2]
    kind = "zs" if run == "test_k0" else "rag"
    df = pd.read_parquet(NM_GEN / f"{tag}_{run}.parquet")
    pool = load_pool().set_index("uid")
    te = pool[pool.role == "test"].sort_values("gidx")
    assert (te["gidx"].to_numpy() == np.arange(3300)).all()
    g = df.set_index("uid").loc[te.index]
    lp = g[["lp_bug", "lp_feature", "lp_question"]].to_numpy()
    base = {"test_idx": np.arange(3300), "title": te["title"].to_numpy(), "body": te["body"].to_numpy(),
            "ground_truth": te["label"].to_numpy()}
    trunc = {c: g[c].to_numpy() for c in ["truncated", "neighbors_truncated", "query_truncated", "tokens_removed"]}
    out = NM / "test_preds"
    out.mkdir(parents=True, exist_ok=True)
    free = pd.DataFrame({**base, "predicted_label": g["predicted_label"].to_numpy(), "raw_output": g["raw_output"].to_numpy(),
                         **trunc, "parsed_via": g["parsed_via"].to_numpy(), "prompt_tokens": g["prompt_tokens"].to_numpy(),
                         "generated_tokens": g["generated_tokens"].to_numpy()})
    cons = pd.DataFrame({**base, "predicted_label": np.array(LABELS)[lp.argmax(1)],
                         "raw_output": [json.dumps(dict(zip(LABELS, np.round(r, 4).tolist()))) for r in lp], **trunc,
                         "parsed_via": "constrained", "prompt_tokens": g["prompt_tokens"].to_numpy(), "generated_tokens": 0})
    free.to_csv(out / f"{kind}_{tag}.csv", index=False)
    cons.to_csv(out / f"{kind}c_{tag}.csv", index=False)
    k = sorted(df.k.unique())
    print(f"{tag} {run} (k={k}): wrote {kind}_{tag}.csv (invalid {np.mean(free.predicted_label == 'invalid'):.4f}) "
          f"and {kind}c_{tag}.csv")
    if run == "test_k0" and (NM_FEATS / f"{tag}_k0.npz").exists():
        z = np.load(NM_FEATS / f"{tag}_k0.npz")
        pos = {int(u): i for i, u in enumerate(z["uids"])}
        ls = z["logp"][[pos[int(u)] for u in te.index]]
        d = np.abs(ls - lp).max()
        agree = np.mean(ls.argmax(1) == lp.argmax(1))
        print(f"   consistency with states pass: max |dlogp| {d:.3f}, constrained-label agreement {agree:.4f}")


if __name__ == "__main__":
    main()

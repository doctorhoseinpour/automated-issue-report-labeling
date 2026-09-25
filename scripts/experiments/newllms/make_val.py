#!/usr/bin/env python3
"""Build splits/val495.csv: the validation queries for tuning RAGTAG's K.

The newest VAL_PER_GROUP (15) issues of every (repo, label) group of the rag_next *dev*
split (itself the newest 30 train issues per group), i.e. 495 issues = 15% of train, all 33
cells. Their retrieval index is *inner* (the older 70 per group), so validation mirrors the
paper's temporal train -> test split. The file carries no labels (the GPU runner never reads
query labels); labels are joined at scoring time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nm_common import NM_SPLITS, VAL_FILE, VAL_PER_GROUP, load_pool


def main():
    pool = load_pool()
    dev = pool[pool.role == "dev"].copy()
    dev["ts"] = pd.to_datetime(dev["created_at"], utc=True)
    dev["order"] = np.arange(len(dev))
    dev = dev.sort_values(["repo", "label", "ts", "order"])
    dev["rank_new"] = dev.groupby(["repo", "label"]).cumcount(ascending=False)  # 0 = newest
    val = dev[dev["rank_new"] < VAL_PER_GROUP]

    cells = val.groupby(["repo", "label"]).size()
    assert len(cells) == 33 and (cells == VAL_PER_GROUP).all(), cells
    assert set(val.role) == {"dev"} and set(val.split) == {"train"}
    rest = dev[dev["rank_new"] >= VAL_PER_GROUP]
    for (r, l), g in val.groupby(["repo", "label"]):  # val is the newer half of dev in every cell
        assert g["ts"].min() >= rest[(rest.repo == r) & (rest.label == l)]["ts"].max(), (r, l)

    NM_SPLITS.mkdir(parents=True, exist_ok=True)
    out = val.sort_values("uid")[["uid", "proj"]]
    out.to_csv(VAL_FILE, index=False)
    print(f"wrote {VAL_FILE}: {len(out)} issues, {out.proj.nunique()} projects, "
          f"{VAL_PER_GROUP} per (project, label)")


if __name__ == "__main__":
    main()

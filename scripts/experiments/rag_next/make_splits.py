#!/usr/bin/env python3
"""Build splits/pool.csv: the paper's train/test split plus a temporal dev slice.

Train/test come verbatim from results/issues11k/agnostic/neighbors/{train,test}_split.csv
(the files every paper method used). Dev = newest DEV_PER_GROUP train issues of
each (repo, label) group by created_at; ties keep file order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DEV_PER_GROUP, RES, SPLITS


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    nb = RES / "agnostic" / "neighbors"
    tr = pd.read_csv(nb / "train_split.csv", keep_default_na=False)
    te = pd.read_csv(nb / "test_split.csv", keep_default_na=False)
    tr["split"], te["split"] = "train", "test"
    tr["tidx"], te["tidx"] = np.arange(len(tr)), -1
    tr["gidx"], te["gidx"] = -1, np.arange(len(te))
    df = pd.concat([tr, te], ignore_index=True)
    df["label"] = df["labels"].astype(str).str.lower().str.strip()
    df["proj"] = df["repo"].str.replace("/", "_", n=1)
    df["uid"] = np.arange(len(df))
    df["ts"] = pd.to_datetime(df["created_at"], utc=True)

    df["role"] = np.where(df["split"] == "test", "test", "inner")
    t = df[df.split == "train"].copy()
    t["order"] = np.arange(len(t))
    t = t.sort_values(["repo", "label", "ts", "order"])
    t["rank_new"] = t.groupby(["repo", "label"]).cumcount(ascending=False)  # 0 = newest
    dev_uids = t.loc[t["rank_new"] < DEV_PER_GROUP, "uid"]
    df.loc[df["uid"].isin(dev_uids), "role"] = "dev"

    # sanity: dev issues are newer than inner issues within each group
    for (r, l), g in df[df.split == "train"].groupby(["repo", "label"]):
        assert g[g.role == "dev"]["ts"].min() >= g[g.role == "inner"]["ts"].max(), (r, l)
        assert (g.role == "dev").sum() == DEV_PER_GROUP
    # and train issues are older than test issues (the paper's temporal split)
    viol = 0
    for (r, l), g in df.groupby(["repo", "label"]):
        viol += int(g[g.split == "train"]["ts"].max() > g[g.split == "test"]["ts"].min())
    print("groups where train is not strictly older than test:", viol)

    cols = ["uid", "split", "role", "tidx", "gidx", "repo", "proj", "label", "created_at", "title", "body"]
    df[cols].to_csv(SPLITS / "pool.csv", index=False)
    print(df.groupby(["split", "role"]).size())


if __name__ == "__main__":
    main()

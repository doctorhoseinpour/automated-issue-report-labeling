"""Stratified-random sample of Qwen-32B prediction errors and invalid outputs.

Sampling protocol (frozen seed = 20260509):

A. MISCLASSIFICATION SAMPLE
   For each method (RAGTAG, BRAGTAG, FT), pick Qwen-32B at its best
   configuration (RAGTAG-PS k=12, BRAGTAG-PS k=12, FT-PA). Within each
   method, restrict to rows where predicted_label != ground_truth and
   predicted_label != "invalid". Stratify by ground_truth label and sample:
       - 30 true-question
       - 10 true-bug
       - 10 true-feature
   Total: 50 / method x 3 methods = 150 misclassifications.

B. INVALID-OUTPUT SAMPLE
   For each method (RAGTAG, BRAGTAG only -- FT has too few invalids), sample
   30 rows where predicted_label == "invalid". Total: 60.

For RAGTAG and BRAGTAG, attach the top-12 neighbors the LLM actually saw
(label, similarity, title) so we can diagnose retrieval-misled errors. FT
has no neighbors.

Output:
   scripts/audit/qual/sample_misclassifications.csv  (150 rows)
   scripts/audit/qual/sample_invalids.csv            (60 rows)
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results" / "issues11k"
sys.path.insert(0, str(REPO / "scripts" / "paper"))
from _rescue import _project_list  # noqa

SEED = 20260509
MODEL_TAG = "unsloth_Qwen2_5_32B_Instruct_bnb_4bit"
RAGTAG_BEST_K = 12
BRAGTAG_BEST_K = 12

OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)


def load_pooled_ps(approach: str, k: int) -> pd.DataFrame:
    parts = []
    for proj in _project_list():
        path = (RESULTS / "project_specific" / proj / MODEL_TAG / approach
                / "predictions" / f"preds_k{k}.csv")
        df = pd.read_csv(path)
        df["__proj"] = proj
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["__method"] = "RAGTAG" if approach == "ragtag" else "BRAGTAG"
    out["__k"] = k
    return out


def load_ft_pa() -> pd.DataFrame:
    path = RESULTS / "agnostic" / MODEL_TAG / "finetune_fixed" / "preds_finetune_fixed.csv"
    df = pd.read_csv(path)
    # Add project from test_split
    test_split = pd.read_csv(RESULTS / "agnostic" / "neighbors" / "test_split.csv")
    df["__proj"] = test_split["repo"].str.replace("/", "_", n=1)
    df["__method"] = "FT"
    df["__k"] = -1
    return df


def attach_neighbors(df: pd.DataFrame, approach: str) -> pd.DataFrame:
    """Attach top-12 neighbor labels/similarities/titles as a single string column."""
    nb_cache: dict[str, pd.DataFrame] = {}
    for proj in _project_list():
        path = RESULTS / "project_specific" / proj / "neighbors" / "neighbors_k30.csv"
        nb_cache[proj] = pd.read_csv(path,
            usecols=["test_idx", "neighbor_rank", "neighbor_similarity",
                     "neighbor_title", "neighbor_label"])
    summaries = []
    for _, row in df.iterrows():
        proj = row["__proj"]
        tid = int(row["test_idx"])
        nb = nb_cache[proj]
        nb = nb[(nb["test_idx"] == tid) & (nb["neighbor_rank"] < 12)] \
            .sort_values("neighbor_rank")
        # Compact summary string
        labels = nb["neighbor_label"].tolist()
        sims = nb["neighbor_similarity"].tolist()
        bug_n = sum(1 for l in labels if l == "bug")
        feat_n = sum(1 for l in labels if l == "feature")
        q_n = sum(1 for l in labels if l == "question")
        summary = f"bug={bug_n}/feat={feat_n}/q={q_n}; sim_top={sims[0]:.3f} sim_bot={sims[-1]:.3f}; labels={labels}"
        summaries.append(summary)
    df = df.copy()
    df["neighbors_top12"] = summaries
    return df


def stratified_sample(df: pd.DataFrame, by: str, n_per: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for val, n in n_per.items():
        sub = df[df[by] == val]
        if len(sub) < n:
            print(f"   WARN: only {len(sub)} rows for {by}={val}; taking all")
            parts.append(sub)
        else:
            idx = rng.choice(sub.index.values, size=n, replace=False)
            parts.append(sub.loc[idx])
    return pd.concat(parts).sort_index()


def main():
    print(f"Loading Qwen-32B predictions (RAGTAG-PS k={RAGTAG_BEST_K}, BRAGTAG-PS k={BRAGTAG_BEST_K}, FT-PA)...")
    rag = load_pooled_ps("ragtag", RAGTAG_BEST_K)
    brag = load_pooled_ps("ragtag_debias_m3", BRAGTAG_BEST_K)
    ft = load_ft_pa()

    print(f"   RAGTAG: {len(rag)} rows")
    print(f"   BRAGTAG: {len(brag)} rows")
    print(f"   FT: {len(ft)} rows")

    # --- Misclassifications ---
    print("\n=== A. Misclassifications ===")
    miss_parts = []
    for df, name in [(rag, "RAGTAG"), (brag, "BRAGTAG"), (ft, "FT")]:
        miss = df[(df["predicted_label"] != df["ground_truth"]) &
                  (df["predicted_label"] != "invalid")].copy()
        print(f"\n   {name} total errors: {len(miss)}")
        by_gt = miss["ground_truth"].value_counts().to_dict()
        print(f"      by ground_truth: {by_gt}")
        sampled = stratified_sample(miss, "ground_truth",
            {"question": 30, "bug": 10, "feature": 10}, SEED + hash(name) % 1000)
        miss_parts.append(sampled)
    miss_df = pd.concat(miss_parts, ignore_index=True)

    # Attach neighbors for RAG/BRAG
    print("\n   Attaching neighbors for RAGTAG/BRAGTAG...")
    rag_brag_mask = miss_df["__method"].isin(["RAGTAG", "BRAGTAG"])
    rb_part = miss_df[rag_brag_mask].copy()
    rb_part_with_nb = []
    for method in ["RAGTAG", "BRAGTAG"]:
        m = rb_part[rb_part["__method"] == method]
        approach = "ragtag" if method == "RAGTAG" else "ragtag_debias_m3"
        rb_part_with_nb.append(attach_neighbors(m, approach))
    miss_df_rb = pd.concat(rb_part_with_nb)
    miss_df_ft = miss_df[~rag_brag_mask].copy()
    miss_df_ft["neighbors_top12"] = ""
    miss_df = pd.concat([miss_df_rb, miss_df_ft]).sort_values(["__method", "ground_truth", "__proj", "test_idx"])

    # Truncate body for readability
    miss_df["body_short"] = miss_df["body"].astype(str).str.slice(0, 1500)
    cols = ["__method", "__proj", "test_idx", "ground_truth", "predicted_label",
            "title", "body_short", "raw_output", "neighbors_top12",
            "truncated", "tokens_removed", "parsed_via",
            "prompt_tokens", "generated_tokens"]
    miss_df[cols].to_csv(OUT_DIR / "sample_misclassifications.csv", index=False)
    print(f"\n   wrote {OUT_DIR / 'sample_misclassifications.csv'} ({len(miss_df)} rows)")
    print(f"   per-cell counts:")
    print(miss_df.groupby(["__method", "ground_truth"]).size())

    # --- Invalid outputs ---
    print("\n=== B. Invalid outputs ===")
    inv_parts = []
    for df, name in [(rag, "RAGTAG"), (brag, "BRAGTAG")]:
        inv = df[df["predicted_label"] == "invalid"].copy()
        print(f"   {name} total invalids: {len(inv)}")
        rng = np.random.default_rng(SEED + hash(name) % 1000)
        n = min(30, len(inv))
        sampled = inv.iloc[rng.choice(len(inv), size=n, replace=False)]
        inv_parts.append(sampled)
    inv_df = pd.concat(inv_parts, ignore_index=True)

    # Attach neighbors
    inv_with_nb = []
    for method in ["RAGTAG", "BRAGTAG"]:
        m = inv_df[inv_df["__method"] == method]
        approach = "ragtag" if method == "RAGTAG" else "ragtag_debias_m3"
        inv_with_nb.append(attach_neighbors(m, approach))
    inv_df = pd.concat(inv_with_nb).sort_values(["__method", "__proj", "test_idx"])
    inv_df["body_short"] = inv_df["body"].astype(str).str.slice(0, 1500)
    inv_df["raw_output_short"] = inv_df["raw_output"].astype(str).str.slice(0, 800)
    cols = ["__method", "__proj", "test_idx", "ground_truth", "predicted_label",
            "title", "body_short", "raw_output_short", "neighbors_top12",
            "truncated", "tokens_removed", "parsed_via",
            "prompt_tokens", "generated_tokens"]
    inv_df[cols].to_csv(OUT_DIR / "sample_invalids.csv", index=False)
    print(f"   wrote {OUT_DIR / 'sample_invalids.csv'} ({len(inv_df)} rows)")
    print(f"   per-cell counts:")
    print(inv_df.groupby("__method").size())


if __name__ == "__main__":
    main()

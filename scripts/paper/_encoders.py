"""Shared loader for the encoder baselines (SetFit x2, RoBERTa-base).

Added for the SANER revision (ESEM Review #126C): cheap encoder baselines to
compare against RAGTAG / BRAGTAG / LoRA fine-tuning. See
docs/SANER_REVISION_PLAN.md P0.1 and run_encoder_baselines.sh.

Result locations (written by run_setfit.py / run_transformer_ft.py):
  PA: results/issues11k/agnostic/<model_tag>/<approach>/...
  PS: results/issues11k/project_specific/<proj>/<model_tag>/<approach>/...

Pooled aggregation convention: PS is concat-then-evaluate across the 11
per-project prediction CSVs in sorted project order (same as _rescue.py);
PA is the single agnostic file. Encoders emit no invalid predictions by
construction (argmax over a fixed 3-class head).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _rescue import RESULTS, _project_list

# (model_tag, approach_dir, preds_filename, display_label)
ENCODERS = [
    ("sentence-transformers_all-mpnet-base-v2", "setfit",
     "preds_setfit.csv", "SetFit (mpnet)"),
    ("Collab-uniba_github-issues-mpnet-st-e10", "setfit",
     "preds_setfit.csv", "SetFit (issues)"),
    ("roberta-base", "finetune_transformer",
     "preds_finetune_transformer.csv", "RoBERTa-base"),
]

USECOLS = ["test_idx", "ground_truth", "predicted_label"]


def _encoder_dir(tag: str, approach: str, setting: str, proj: str | None = None) -> Path:
    if setting == "PA":
        return RESULTS / "agnostic" / tag / approach
    return RESULTS / "project_specific" / proj / tag / approach


def load_encoder_preds(tag: str, approach: str, fname: str, setting: str) -> pd.DataFrame:
    """Pooled predictions frame (3,300 rows). PS pools in sorted project order."""
    if setting == "PA":
        return pd.read_csv(_encoder_dir(tag, approach, "PA") / "predictions" / fname,
                           usecols=USECOLS)
    parts = [
        pd.read_csv(_encoder_dir(tag, approach, "PS", proj) / "predictions" / fname,
                    usecols=USECOLS)
        for proj in _project_list()
    ]
    return pd.concat(parts, ignore_index=True)


def load_encoder_cost(tag: str, approach: str, setting: str) -> dict:
    """Cost summary. PS sums train/infer time across the 11 projects and takes
    the max peak GPU memory (each project run is a fresh process). Excludes
    model load time per project convention."""
    if setting == "PA":
        files = [_encoder_dir(tag, approach, "PA") / "cost_metrics.csv"]
    else:
        files = [_encoder_dir(tag, approach, "PS", proj) / "cost_metrics.csv"
                 for proj in _project_list()]
    train_s = infer_s = 0.0
    ram_mb = 0.0
    for f in files:
        row = pd.read_csv(f).iloc[0]
        train_s += float(row["training_time_s"])
        infer_s += float(row["wall_time_s"])
        ram_mb = max(ram_mb, float(row["gpu_peak_memory_mb"]))
    return {"train_time_s": train_s, "infer_time_s": infer_s,
            "total_time_s": train_s + infer_s, "gpu_ram_mb": ram_mb}


def ps_pooled_to_global(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder a pooled-PS frame (sorted-project concat, local test_idx) into
    the agnostic global test ordering, for per-issue pairing against PA frames.

    Uses the test_split.csv repo column as the canonical agnostic ordering
    (same walk as significance_method_comparison._aligned_pair).
    """
    test_split = pd.read_csv(RESULTS / "agnostic" / "neighbors" / "test_split.csv",
                             usecols=["repo"])
    proj_tags = test_split["repo"].str.replace("/", "_", n=1).tolist()
    if len(df) != len(proj_tags):
        raise RuntimeError(f"pooled PS frame has {len(df)} rows, expected {len(proj_tags)}")

    per_proj = {}
    start = 0
    for proj in _project_list():
        block = df.iloc[start:start + 300].reset_index(drop=True)
        per_proj[proj] = block
        start += 300

    local_counter = {p: 0 for p in per_proj}
    rows = []
    for proj in proj_tags:
        li = local_counter[proj]
        local_counter[proj] += 1
        rows.append(per_proj[proj].iloc[li])
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["test_idx"] = np.arange(len(out))
    return out

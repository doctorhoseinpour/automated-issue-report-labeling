"""Shared configuration for the newllms study (new paper, started 2026-09-24).

The rag_next decision-state read-out, carried unchanged to three new open LLMs, against a
RAGTAG-PS baseline whose K is tuned on a validation split carved from train.

Layout (lab machine; OSC mirrors it under /fs/ess/PCS0289/rag_next/repo):
  results/issues11k/exploration/newllms/
      splits/val495.csv     validation queries (uid, proj): newest 15 per (project, label) of dev
      raw/<tag>/<run>/      GPU runner parts (run_llm.py)
      features/<tag>_k0.npz merged zero-shot states (rag_next features format)
      gen/<tag>_<run>.parquet merged RAGTAG generations
Read-only inputs from rag_next: splits/pool.csv, features/nb_PS_raw_{dev,test}.npz.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rag_next"))
sys.path.insert(0, str(HERE.parents[2]))  # repo root: llm_labeler, evaluate

from common import (EXP as RN, FEATS as RN_FEATS, LAB2ID, LABELS, RES, boot_diff,  # noqa: E402,F401
                    load_pool, macro_f1, per_class_f1)

NM = RES / "exploration" / "newllms"
NM_SPLITS = NM / "splits"
NM_RAW = NM / "raw"
NM_FEATS = NM / "features"
NM_GEN = NM / "gen"
VAL_FILE = NM_SPLITS / "val495.csv"

VAL_PER_GROUP = 15
K_GRID = [0, 1, 3, 5, 7, 9, 10, 12, 15]
MAX_SEQ = 8192          # the paper's context
MAX_NEW = 50            # the paper's llm_labeler --max_new_tokens (prompt budget = ctx - 50 - 20)
DEPTH_FRACS = (0.625, 0.75, 0.875, 1.0)   # frozen rag_next read-out recipe

MODELS = {
    "qw35_9b": {"hf": "Qwen/Qwen3.5-9B", "name": "Qwen3.5-9B", "n_layers": 32},
    "gm4_12b": {"hf": "google/gemma-4-12B-it", "name": "Gemma-4-12B-it", "n_layers": 48},
    "mi3_8b": {"hf": "mistralai/Ministral-3-8B-Instruct-2512-BF16", "name": "Ministral-3-8B-Instruct-2512",
               "n_layers": 34},
}


def readout_layers(n_layers: int) -> list[int]:
    """Depth fractions -> 1-based layer indices, rounded exactly as rag_next/llm_features.py."""
    return sorted(set(min(n_layers, max(1, int(round(f * n_layers)))) for f in DEPTH_FRACS))

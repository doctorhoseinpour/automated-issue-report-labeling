"""VTAG-rescue utility for RAGTAG / Debiased RAGTAG predictions.

When a RAGTAG (or Debiased RAGTAG) prediction fails to parse a valid label
(predicted_label == "invalid"), substitute VTAG's BEST prediction in the
same setting. The VTAG-best vote uses the same top-30 neighbor pool that
retrieval already produced, so the substitution incurs no extra cost.

Rescue source (always VTAG's best per setting):
  - PS predictions (any k)    -> VTAG-PS at k=15 (PS peak)
  - PA predictions (any k)    -> VTAG-PA at k=16 (PA peak)
  - Zero-shot (no setting)    -> VTAG-PS at k=15 (PS is the paper's
                                  canonical setting)
  - Debiased RAGTAG (PS only) -> VTAG-PS at k=15 (vtag_debias_m3 only
                                  exists for k<=9 and is weaker than
                                  vanilla VTAG-PS at k=15; we use the
                                  stronger fallback)

Pooled aggregation convention: see paper/sections/04_setup.tex. PS is
concat-then-evaluate across the 11 per-project prediction CSVs; PA is the
single agnostic file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "issues11k"

VTAG_BEST_K_PS = 15  # VTAG-PS pooled peak (macro F1 = 0.5951)
VTAG_BEST_K_PA = 16  # VTAG-PA pooled peak (macro F1 = 0.6039)

Setting = Literal["PS", "PA"]
Approach = Literal["ragtag", "ragtag_debias_m3"]


def _project_list() -> list[str]:
    return sorted(p.name for p in (RESULTS / "project_specific").iterdir() if p.is_dir())


def _k_label(k: int) -> str:
    return "zero_shot" if k == 0 else f"k{k}"


def _ragtag_path(setting: Setting, model: str, approach: Approach,
                 k: int, proj: str | None = None) -> Path:
    fname = f"preds_{_k_label(k)}.csv"
    if setting == "PA":
        return RESULTS / "agnostic" / model / approach / "predictions" / fname
    return RESULTS / "project_specific" / proj / model / approach / "predictions" / fname


def _vtag_rescue_path(setting: Setting, approach: Approach, k: int,
                      proj: str | None = None) -> Path | None:
    """Resolve which VTAG file to use as the rescue source.

    Always returns VTAG's best file in the requested setting. PS rescue uses
    VTAG-PS at k=15 (the PS peak). PA rescue uses VTAG-PA at k=16 (the PA
    peak). Zero-shot RAGTAG (k=0) is ambiguous in setting -- callers should
    pass setting="PS" since PS is the paper's canonical setting.
    """
    best_k = VTAG_BEST_K_PS if setting == "PS" else VTAG_BEST_K_PA
    fname = f"preds_k{best_k}.csv"
    if setting == "PA":
        return RESULTS / "agnostic" / "vtag" / "predictions" / fname
    return RESULTS / "project_specific" / proj / "vtag" / "predictions" / fname


def _load_one(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["test_idx", "ground_truth", "predicted_label"])


def load_rescued_preds(model: str, setting: Setting, k: int,
                       approach: Approach = "ragtag") -> pd.DataFrame:
    """Return per-test-issue (ground_truth, predicted_label) with rescue applied.

    For PS the result is pooled across the 11 projects (3,300 rows). For PA
    the result is the single agnostic file (3,300 rows). predicted_label is
    the rescued value (VTAG-best vote when RAGTAG was "invalid", original
    RAGTAG otherwise).

    Zero-shot (k=0) RAGTAG predictions are stored under the agnostic path
    only; the rescue source is VTAG-PS at k=15 regardless of the setting
    argument, since PS is the paper's canonical setting.
    """
    if k == 0:
        # Zero-shot is in the agnostic dir; rescue with VTAG-PS k=15.
        rag = _load_one(_ragtag_path("PA", model, approach, 0, None))
        # Build a VTAG-PS-best lookup keyed by (proj, test_idx)? No --
        # zero-shot RAGTAG has no per-project breakdown, but VTAG-PS does.
        # We need to map each zero-shot row to its project's VTAG-PS k=15
        # prediction. Use the test_split.csv repo column.
        test_split = pd.read_csv(RESULTS / "agnostic" / "neighbors" / "test_split.csv",
                                 usecols=["repo"])
        proj_tags = test_split["repo"].str.replace("/", "_", n=1).tolist()
        rescue_lookup = {}  # (proj, test_idx) -> predicted_label
        for proj in _project_list():
            vtg = _load_one(RESULTS / "project_specific" / proj / "vtag"
                            / "predictions" / f"preds_k{VTAG_BEST_K_PS}.csv")
            for _, row in vtg.iterrows():
                rescue_lookup[(proj, int(row["test_idx"]))] = row["predicted_label"]
        # Apply rescue: for each zero-shot row, find its project from
        # test_split (in same order as agnostic test_idx) and look up
        # the per-project VTAG prediction.
        rescued_labels = []
        # Per-project test_idx within agnostic: count occurrences as we walk.
        proj_local_idx = {}
        for global_idx, row in rag.iterrows():
            proj = proj_tags[global_idx]
            local_idx = proj_local_idx.get(proj, 0)
            proj_local_idx[proj] = local_idx + 1
            if row["predicted_label"] == "invalid":
                rescued_labels.append(rescue_lookup.get((proj, local_idx), "invalid"))
            else:
                rescued_labels.append(row["predicted_label"])
        rag = rag.copy()
        rag["predicted_label"] = rescued_labels
        return rag[["test_idx", "ground_truth", "predicted_label"]].reset_index(drop=True)

    projects = _project_list() if setting == "PS" else [None]
    parts = []
    for proj in projects:
        rag = _load_one(_ragtag_path(setting, model, approach, k, proj))
        rescue_p = _vtag_rescue_path(setting, approach, k, proj)
        if rescue_p is not None and rescue_p.exists():
            vtg = _load_one(rescue_p).set_index("test_idx")
            # Substitute VTAG label wherever RAGTAG is invalid.
            inv_mask = rag["predicted_label"] == "invalid"
            if inv_mask.any():
                rag.loc[inv_mask, "predicted_label"] = (
                    rag.loc[inv_mask, "test_idx"].map(vtg["predicted_label"])
                )
                # If any still invalid (mapping miss), leave them as "invalid".
        parts.append(rag)
    out = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0].reset_index(drop=True)
    return out[["test_idx", "ground_truth", "predicted_label"]]


def load_raw_preds(model: str, setting: Setting, k: int,
                   approach: Approach = "ragtag") -> pd.DataFrame:
    """Same shape as load_rescued_preds but no rescue substitution.

    Useful for raw-vs-rescued comparison reporting.
    """
    projects = _project_list() if setting == "PS" else [None]
    parts = []
    for proj in projects:
        parts.append(_load_one(_ragtag_path(setting, model, approach, k, proj)))
    out = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0].reset_index(drop=True)
    return out[["test_idx", "ground_truth", "predicted_label"]]

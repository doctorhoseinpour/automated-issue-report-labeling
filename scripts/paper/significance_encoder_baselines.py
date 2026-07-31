"""Significance tests: encoder baselines vs. the paper's LLM-based methods.

For each encoder baseline (SetFit-mpnet, SetFit-issues, RoBERTa-base) at its
stronger data scope, against each Qwen size's best configuration of
RAGTAG-PS, BRAGTAG-PS, and Fine-Tune-PA — all on raw predictions (invalid
LLM outputs count as incorrect, matching the paper's headline reporting).

Per comparison:
  1. Paired bootstrap 95% CI on (encoder - LLM method) macro F1 difference
     (1,000 resamples, seed 42)
  2. McNemar's test (continuity-corrected) on accuracy disagreements —
     the exact test used by Colavito et al. (IST 2025) for this comparison
  3. TOST equivalence at delta = 0.01 / 0.02 / 0.05

Pairing: every frame is brought into the agnostic global test ordering
(3,300 rows). PA frames are already global; pooled-PS frames are reordered
via the test_split.csv repo walk (ps_pooled_to_global, same methodology as
significance_method_comparison._aligned_pair). Ground-truth alignment is
asserted per pair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rescue import RESULTS, load_raw_preds  # noqa: E402
from _encoders import ENCODERS, load_encoder_preds, ps_pooled_to_global  # noqa: E402

LABELS = ["bug", "feature", "question"]
KS_RAG = [1, 3, 6, 9, 12, 15]
N_BOOTSTRAP = 1000
RNG_SEED = 42

QWEN_MODELS = [
    ("unsloth_Qwen2_5_3B_Instruct_bnb_4bit",  "Qwen-3B"),
    ("unsloth_Qwen2_5_7B_Instruct_bnb_4bit",  "Qwen-7B"),
    ("unsloth_Qwen2_5_14B_Instruct_bnb_4bit", "Qwen-14B"),
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "Qwen-32B"),
]


def _macro(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)


def _global_frame_encoder(tag: str, approach: str, fname: str, setting: str) -> pd.DataFrame:
    df = load_encoder_preds(tag, approach, fname, setting)
    return ps_pooled_to_global(df) if setting == "PS" else df.sort_values("test_idx").reset_index(drop=True)


def _global_frame_llm(model: str, method: str) -> tuple[pd.DataFrame, str]:
    """Best-config raw predictions in global order. Returns (frame, config label)."""
    if method == "finetune":
        df = pd.read_csv(
            RESULTS / "agnostic" / model / "finetune_fixed" / "preds_finetune_fixed.csv",
            usecols=["test_idx", "ground_truth", "predicted_label"],
        ).sort_values("test_idx").reset_index(drop=True)
        return df, "PA"
    def _k_macro(k: int) -> float:
        df = load_raw_preds(model, "PS", k, method)
        return _macro(df["ground_truth"].values, df["predicted_label"].values)

    best_k = max(KS_RAG, key=_k_macro)
    df = ps_pooled_to_global(load_raw_preds(model, "PS", best_k, method))
    return df, f"PS k={best_k}"


def _bootstrap_diff_ci(y_true, yp_a, yp_b, n=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n_obs = len(y_true)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        diffs[i] = _macro(y_true[idx], yp_a[idx]) - _macro(y_true[idx], yp_b[idx])
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def _mcnemar_p(y_true, yp_a, yp_b) -> float:
    a_ok = yp_a == y_true
    b_ok = yp_b == y_true
    table = [
        [int((a_ok & b_ok).sum()), int((a_ok & ~b_ok).sum())],
        [int((~a_ok & b_ok).sum()), int((~a_ok & ~b_ok).sum())],
    ]
    return float(mcnemar(table, exact=False, correction=True).pvalue)


def _tost(lo: float, hi: float, delta: float) -> str:
    return "PASS" if (lo > -delta and hi < +delta) else "fail"


def main():
    # Choose each encoder's stronger scope on pooled raw macro F1.
    enc_frames = {}
    for tag, approach, fname, lbl in ENCODERS:
        best = None
        for setting in ("PA", "PS"):
            g = _global_frame_encoder(tag, approach, fname, setting)
            m = _macro(g["ground_truth"].values, g["predicted_label"].values)
            if best is None or m > best[2]:
                best = (setting, g, m)
        enc_frames[lbl] = best
        print(f"{lbl}: using {best[0]} scope (pooled macro F1 = {best[2]:.4f})")

    print()
    print("=" * 132)
    print(f"{'Encoder':<17} {'vs LLM method':<26} {'cfg':<9} "
          f"{'mean diff':>10}  {'95% CI':<22} {'McNemar p':>11} "
          f"{'T.01':>5} {'T.02':>5} {'T.05':>5}")
    print("=" * 132)

    for enc_lbl, (enc_setting, enc_g, _) in enc_frames.items():
        y_enc = enc_g["ground_truth"].values
        yp_enc = enc_g["predicted_label"].values
        for model, mlbl in QWEN_MODELS:
            for method, mname in (("ragtag", "RAGTAG"), ("ragtag_debias_m3", "BRAGTAG"),
                                  ("finetune", "Fine-Tune")):
                llm_g, cfg = _global_frame_llm(model, method)
                if not (llm_g["ground_truth"].values == y_enc).all():
                    raise RuntimeError(f"ground-truth misalignment: {enc_lbl} vs {mname} {mlbl}")
                yp_llm = llm_g["predicted_label"].values
                mean_d, lo, hi = _bootstrap_diff_ci(y_enc, yp_enc, yp_llm)
                pval = _mcnemar_p(y_enc, yp_enc, yp_llm)
                sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                print(f"{enc_lbl:<17} {mname + ' (' + mlbl + ')':<26} {cfg:<9} "
                      f"{mean_d:>+10.4f}  [{lo:+.4f}, {hi:+.4f}]  {pval:>8.2e}{sig:<3} "
                      f"{_tost(lo, hi, 0.01):>5} {_tost(lo, hi, 0.02):>5} {_tost(lo, hi, 0.05):>5}")
        print("-" * 132)

    print()
    print("diff = encoder - LLM method, macro F1, raw predictions (invalid LLM outputs count as incorrect).")
    print("Bootstrap: paired, 1000 resamples, seed 42, percentile 95% CI, on the aligned 3300-row global pairing.")
    print("McNemar:   continuity-corrected on accuracy disagreements (the test used by Colavito et al., IST 2025).")
    print("TOST:      PASS = 95% CI entirely within [-delta, +delta].")


if __name__ == "__main__":
    main()

"""RQ2.9: Bootstrap CIs and McNemar tests on the headline pairs.

For each pair (predictor_a, predictor_b), align predictions on a common
3,300-issue test index, then:
  - bootstrap 1000 resamples of the test set, compute macro F1 deltas
  - run McNemar's exact test on paired correct/incorrect decisions

Headline pairs:
  H1.  VTAG-9 ag vs Qwen-3B zero-shot ag         (smallest LLM beats VTAG?)
  H2.  RAGTAG-best ag vs VTAG (best k) ag        (LLM adds value over retrieval) per Qwen size
  H3.  RAGTAG-best vs FT, per Qwen size          (RAGTAG vs SOTA prior approach)
  H4.  Debias-best ps vs RAGTAG-best ps          (debias actually moves the needle?)
  H5.  Debias-best ps vs FT-ag                   (debias closes/beats FT gap?)
  H6.  Qwen-7B FT-ag vs Qwen-7B Debias-best ps   (the +0.011 anomaly)

Outputs:
  docs/analysis/rq2_significance.csv
  docs/analysis/rq2_significance.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).parent))
from _utils import (  # noqa: E402
    DOCS_ANALYSIS,
    PROJECTS,
    REPO_ROOT,
    RESULTS_DIR,
    VALID_LABELS,
    ensure_dirs,
    load_predictions,
    rel,
)


N_BOOTSTRAP = 1000
RANDOM_SEED = 0


def _project_global_indices() -> dict[str, list[int]]:
    """For each project, list the agnostic global indices in this project's test set."""
    ts = pd.read_csv(RESULTS_DIR / "agnostic" / "neighbors" / "test_split.csv")
    out: dict[str, list[int]] = {}
    for proj in PROJECTS:
        repo = proj.replace("_", "/", 1)
        idx = ts.index[ts["repo"] == repo].tolist()
        out[proj] = idx
    return out


def _load_aligned(spec: dict, preds_index: pd.DataFrame, proj_idx: dict[str, list[int]]) -> pd.DataFrame:
    """Return DataFrame indexed by global_test_idx in [0..3299] with columns 'gt' and 'pred'.

    spec:
      mode='agnostic'  -> single predictions CSV (3300 rows, test_idx 0..3299)
      mode='ps_concat' -> concat 11 projects, mapping local test_idx -> global via proj_idx
    """
    if spec["mode"] == "agnostic":
        rows = preds_index[
            (preds_index["model"] == spec["model"])
            & (preds_index["approach"] == spec["approach"])
            & (preds_index["setting"] == "agnostic")
            & (preds_index["project"] == "_overall")
            & (preds_index["k_label"] == spec["k_label"])
        ]
        if rows.empty:
            raise RuntimeError(f"missing predictions: {spec}")
        df = load_predictions(REPO_ROOT / rows.iloc[0]["predictions_path"])
        df = df.dropna(subset=["test_idx"])
        df["global_idx"] = df["test_idx"].astype(int)
        return df.set_index("global_idx")[["ground_truth", "predicted_label"]].rename(
            columns={"ground_truth": "gt", "predicted_label": "pred"},
        )

    if spec["mode"] == "vtag_agnostic":
        rows = preds_index[
            (preds_index["approach"] == "vtag")
            & (preds_index["setting"] == "agnostic")
            & (preds_index["k_label"] == spec["k_label"])
        ]
        if rows.empty:
            raise RuntimeError(f"missing vtag preds: {spec}")
        df = load_predictions(REPO_ROOT / rows.iloc[0]["predictions_path"])
        df["global_idx"] = df["test_idx"].astype(int)
        return df.set_index("global_idx")[["ground_truth", "predicted_label"]].rename(
            columns={"ground_truth": "gt", "predicted_label": "pred"},
        )

    if spec["mode"] == "ps_concat":
        parts: list[pd.DataFrame] = []
        for proj in PROJECTS:
            rows = preds_index[
                (preds_index["model"] == spec["model"])
                & (preds_index["approach"] == spec["approach"])
                & (preds_index["setting"] == "project_specific")
                & (preds_index["project"] == proj)
                & (preds_index["k_label"] == spec["k_label"])
            ]
            if rows.empty:
                continue
            df = load_predictions(REPO_ROOT / rows.iloc[0]["predictions_path"])
            df = df.dropna(subset=["test_idx"])
            local = df["test_idx"].astype(int).tolist()
            mapping = proj_idx[proj]
            df["global_idx"] = [mapping[i] for i in local]
            parts.append(df[["global_idx", "ground_truth", "predicted_label"]])
        if not parts:
            raise RuntimeError(f"missing ps preds: {spec}")
        full = pd.concat(parts, ignore_index=True).set_index("global_idx")
        return full[["ground_truth", "predicted_label"]].rename(
            columns={"ground_truth": "gt", "predicted_label": "pred"},
        )

    raise ValueError(f"unknown mode: {spec['mode']}")


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = [str(x).lower().strip() for x in y_true]
    yp = [str(x).lower().strip() for x in y_pred]
    _, _, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=VALID_LABELS, average="macro", zero_division=0,
    )
    return float(f1)


def _bootstrap_delta(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = N_BOOTSTRAP) -> dict:
    """Resample shared indices, compute (F1_a, F1_b, delta = a - b) on each resample."""
    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]
    n = len(common)
    rng = np.random.default_rng(RANDOM_SEED)

    yt = a["gt"].astype(str).str.lower().str.strip().to_numpy()
    pa = a["pred"].astype(str).str.lower().str.strip().to_numpy()
    pb = b["pred"].astype(str).str.lower().str.strip().to_numpy()

    f1_a_all = _macro_f1(yt, pa)
    f1_b_all = _macro_f1(yt, pb)

    deltas = np.empty(n_boot)
    f1as = np.empty(n_boot)
    f1bs = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.integers(0, n, n)
        f1a = _macro_f1(yt[sample], pa[sample])
        f1b = _macro_f1(yt[sample], pb[sample])
        f1as[i] = f1a
        f1bs[i] = f1b
        deltas[i] = f1a - f1b

    def ci(arr: np.ndarray) -> tuple[float, float]:
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    a_corr = (yt == pa)
    b_corr = (yt == pb)
    table = np.array([
        [int(((a_corr) & (b_corr)).sum()),  int(((a_corr) & (~b_corr)).sum())],
        [int(((~a_corr) & (b_corr)).sum()), int(((~a_corr) & (~b_corr)).sum())],
    ])
    # Use exact McNemar when the off-diagonal is small.
    use_exact = (table[0, 1] + table[1, 0]) < 25
    res = mcnemar(table, exact=use_exact, correction=not use_exact)

    return {
        "n": n,
        "f1_a_point": f1_a_all,
        "f1_b_point": f1_b_all,
        "delta_point": f1_a_all - f1_b_all,
        "f1_a_ci_lo": ci(f1as)[0], "f1_a_ci_hi": ci(f1as)[1],
        "f1_b_ci_lo": ci(f1bs)[0], "f1_b_ci_hi": ci(f1bs)[1],
        "delta_ci_lo": ci(deltas)[0], "delta_ci_hi": ci(deltas)[1],
        "mcnemar_stat": float(res.statistic) if res.statistic is not None else float("nan"),
        "mcnemar_pvalue": float(res.pvalue),
        "mcnemar_method": "exact" if use_exact else "chi2_continuity",
        "n_a_only_correct": int(table[0, 1]),
        "n_b_only_correct": int(table[1, 0]),
    }


def _build_pairs() -> list[tuple[str, dict, dict]]:
    pairs: list[tuple[str, dict, dict]] = []
    # H1: VTAG-9 ag vs Qwen-3B zero-shot ag
    pairs.append((
        "H1: Qwen-3B zero-shot (ag) vs VTAG k=9 (ag)",
        dict(mode="agnostic", model="Qwen-3B", approach="ragtag", k_label="zero_shot"),
        dict(mode="vtag_agnostic", k_label="k9"),
    ))
    # H2: RAGTAG-best vs VTAG (best k) per Qwen size, agnostic
    rag_best = {"Qwen-3B": "k3", "Qwen-7B": "k6", "Qwen-14B": "k9", "Qwen-32B": "k9"}
    vtag_best = "k9"  # within RAGTAG's k grid; full peak is k=16 but we keep this comparable
    for m, k in rag_best.items():
        pairs.append((
            f"H2[{m}]: RAGTAG ag {k} vs VTAG ag k=9",
            dict(mode="agnostic", model=m, approach="ragtag", k_label=k),
            dict(mode="vtag_agnostic", k_label=vtag_best),
        ))
    # H3: RAGTAG-best ag vs FT-ag per Qwen size
    for m, k in rag_best.items():
        pairs.append((
            f"H3[{m}]: RAGTAG ag {k} vs FT ag",
            dict(mode="agnostic", model=m, approach="ragtag", k_label=k),
            dict(mode="agnostic", model=m, approach="ft", k_label="finetune_fixed"),
        ))
    # H4: Debias-best ps vs RAGTAG-best ps per Qwen size
    deb_best = {"Qwen-3B": "k6", "Qwen-7B": "k6", "Qwen-14B": "k9", "Qwen-32B": "k9"}
    rag_ps_best = {"Qwen-3B": "k3", "Qwen-7B": "k6", "Qwen-14B": "k9", "Qwen-32B": "k9"}
    for m in rag_best.keys():
        pairs.append((
            f"H4[{m}]: Debias ps {deb_best[m]} vs RAGTAG ps {rag_ps_best[m]}",
            dict(mode="ps_concat", model=m, approach="ragtag_debias", k_label=deb_best[m]),
            dict(mode="ps_concat", model=m, approach="ragtag", k_label=rag_ps_best[m]),
        ))
    # H5: Debias-best ps vs FT-ag per Qwen size (the closing-the-gap claim)
    for m in rag_best.keys():
        pairs.append((
            f"H5[{m}]: Debias ps {deb_best[m]} vs FT ag",
            dict(mode="ps_concat", model=m, approach="ragtag_debias", k_label=deb_best[m]),
            dict(mode="agnostic", model=m, approach="ft", k_label="finetune_fixed"),
        ))
    # H6: Qwen-7B FT-ag vs Qwen-7B Debias-best ps (the +0.011 anomaly direction reversed)
    pairs.append((
        "H6: Qwen-7B FT ag vs Qwen-7B Debias ps k6 (the 7B anomaly)",
        dict(mode="agnostic", model="Qwen-7B", approach="ft", k_label="finetune_fixed"),
        dict(mode="ps_concat", model="Qwen-7B", approach="ragtag_debias", k_label="k6"),
    ))
    return pairs


def main() -> None:
    ensure_dirs()
    preds_index = pd.read_csv(DOCS_ANALYSIS / "preds_index.csv")
    proj_idx = _project_global_indices()
    pairs = _build_pairs()

    rows: list[dict] = []
    for label, spec_a, spec_b in pairs:
        try:
            a = _load_aligned(spec_a, preds_index, proj_idx)
            b = _load_aligned(spec_b, preds_index, proj_idx)
        except Exception as e:
            print(f"SKIP {label}: {e}")
            continue
        res = _bootstrap_delta(a, b)
        res["pair"] = label
        rows.append(res)
        sig = "***" if res["mcnemar_pvalue"] < 0.001 else (
            "**" if res["mcnemar_pvalue"] < 0.01 else (
                "*" if res["mcnemar_pvalue"] < 0.05 else "ns"
            )
        )
        print(
            f"{label:60s}  "
            f"ΔF1={res['delta_point']:+.4f}  "
            f"95%CI=[{res['delta_ci_lo']:+.4f}, {res['delta_ci_hi']:+.4f}]  "
            f"McNemar p={res['mcnemar_pvalue']:.4g} [{sig}]",
        )

    df = pd.DataFrame(rows)
    df = df[
        ["pair", "n", "f1_a_point", "f1_a_ci_lo", "f1_a_ci_hi",
         "f1_b_point", "f1_b_ci_lo", "f1_b_ci_hi",
         "delta_point", "delta_ci_lo", "delta_ci_hi",
         "mcnemar_stat", "mcnemar_pvalue", "mcnemar_method",
         "n_a_only_correct", "n_b_only_correct"]
    ]
    df.to_csv(DOCS_ANALYSIS / "rq2_significance.csv", index=False)
    print(f"\nwrote {rel(DOCS_ANALYSIS / 'rq2_significance.csv')}")

    # Compact markdown summary.
    md = ["# RQ2.9 — Bootstrap CIs + McNemar on headline pairs", ""]
    md.append("| Pair | n | F1 A | F1 B | ΔF1 (A−B) | 95% CI | McNemar p | Sig |")
    md.append("|---|---:|---:|---:|---:|---|---:|:---:|")
    for _, r in df.iterrows():
        sig = "***" if r["mcnemar_pvalue"] < 0.001 else (
            "**" if r["mcnemar_pvalue"] < 0.01 else (
                "*" if r["mcnemar_pvalue"] < 0.05 else "ns"
            )
        )
        md.append(
            f"| {r['pair']} | {int(r['n'])} | "
            f"{r['f1_a_point']:.4f} | {r['f1_b_point']:.4f} | "
            f"{r['delta_point']:+.4f} | "
            f"[{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}] | "
            f"{r['mcnemar_pvalue']:.3g} | {sig} |",
        )
    md.append("")
    md.append("Significance markers: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` not significant.")
    (DOCS_ANALYSIS / "rq2_significance.md").write_text("\n".join(md) + "\n")
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq2_significance.md')}")


if __name__ == "__main__":
    main()

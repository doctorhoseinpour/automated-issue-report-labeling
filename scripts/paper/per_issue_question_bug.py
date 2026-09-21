"""Per-issue analysis of the question->bug confusion (RQ2 -> RQ3 motivation).

Runs on the lab machine only (reads results/). CPU, ~1 min.

Questions answered, per Qwen size at RAGTAG-PS's best k (raw macro F1) and
pooled over the four sizes, always restricted to TRUE-QUESTION test issues
unless stated otherwise:

  A. Correction table. Among questions that ZERO-SHOT labeled bug, how often
     does RAGTAG correct them to question, split by (i) VOTAG-PS's vote on the
     same issue and (ii) the label mix of the exact top-k examples RAGTAG saw
     (bug majority vs not). Tests "the LLM follows the neighbors on this
     boundary": correction should be rarer when the examples favor bug.
  B. Overlap. Among questions RAGTAG still labels bug, the share VOTAG also
     labels bug, against the base rate P(VOTAG=bug | question). Tests whether
     the two methods err on the SAME questions, not just at the same rate.
  C. Same-issue, same-k BRAGTAG flips. Among questions RAGTAG labels bug,
     split by whether the debias trigger fired (bug_count - question_count
     <= 3 on the top-k), what BRAGTAG (same k) predicts. Non-fired prompts
     are identical, so those predictions must match (sanity check). Also the
     cost side: true bugs RAGTAG got right that BRAGTAG flips away.
  D. Count confound. Within fired questions, RAGTAG->BRAGTAG correction rate
     binned by how many examples were removed / how many remained. If the
     rate does not fall with fewer remaining examples, the effect is from
     WHICH examples were removed, not how many.

Convention: pooled, raw predictions (invalid counts as its own outcome).
Alignment of the agnostic zero-shot file to per-project files follows
significance_method_comparison.py (walk test_split.csv's repo column).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rescue import RESULTS, VTAG_BEST_K_PS, _project_list  # noqa: E402

LABELS = ["bug", "feature", "question"]
OUTCOMES = LABELS + ["invalid"]
KS = [1, 3, 6, 9, 12, 15]
MARGIN = 3
MODELS = [
    ("unsloth_Qwen2_5_3B_Instruct_bnb_4bit", "Qwen-3B"),
    ("unsloth_Qwen2_5_7B_Instruct_bnb_4bit", "Qwen-7B"),
    ("unsloth_Qwen2_5_14B_Instruct_bnb_4bit", "Qwen-14B"),
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "Qwen-32B"),
]
BUG_SYN = {"bug", "bugfix", "defect", "issue", "fix"}
Q_SYN = {"question", "support", "howto", "help"}
OUT_CSV = Path(__file__).resolve().parents[2] / "paper" / "tables" / "per_issue_question_bug.csv"


def _norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().str.strip()
    return s.where(s.isin(LABELS), "invalid")


def _preds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["test_idx", "ground_truth", "predicted_label"])
    df["ground_truth"] = df["ground_truth"].astype(str).str.lower().str.strip()
    df["predicted_label"] = _norm(df["predicted_label"])
    return df.set_index("test_idx")


def _best_k(model: str) -> int:
    best_k, best = None, -1.0
    for k in KS:
        parts = [pd.read_csv(RESULTS / "project_specific" / p / model / "ragtag" / "predictions"
                             / f"preds_k{k}.csv", usecols=["ground_truth", "predicted_label"])
                 for p in _project_list()]
        df = pd.concat(parts, ignore_index=True)
        m = f1_score(df["ground_truth"], _norm(df["predicted_label"]), labels=LABELS,
                     average="macro", zero_division=0)
        if m > best:
            best, best_k = m, k
    return best_k


def _neighbor_mix(proj: str, k: int) -> pd.DataFrame:
    """Per test_idx: bug/question/feature counts among the top-k neighbors,
    and whether the margin-3 debias trigger fires (mirrors llm_labeler)."""
    nb = pd.read_csv(RESULTS / "project_specific" / proj / "neighbors" / "neighbors_k30.csv",
                     usecols=["test_idx", "neighbor_rank", "neighbor_label"])
    nb = nb[nb["neighbor_rank"] <= k]
    lab = nb["neighbor_label"].astype(str).str.lower().str.strip()
    nb = nb.assign(is_bug=lab.isin(BUG_SYN), is_q=lab.isin(Q_SYN))
    g = nb.groupby("test_idx").agg(n_bug=("is_bug", "sum"), n_q=("is_q", "sum"), n=("is_bug", "size"))
    g["n_feat"] = g["n"] - g["n_bug"] - g["n_q"]
    g["fired"] = (g["n_bug"] > 0) & (g["n_bug"] - g["n_q"] <= MARGIN)
    g["bug_majority"] = g["n_bug"] > g["n_q"]
    return g


def build(model: str, k: int) -> pd.DataFrame:
    """One row per test issue (3,300): truth, zero-shot, ragtag@k, bragtag@k,
    votag-ps@15, neighbor mix @k."""
    zs = _preds(RESULTS / "agnostic" / model / "ragtag" / "predictions" / "preds_zero_shot.csv")
    repo = pd.read_csv(RESULTS / "agnostic" / "neighbors" / "test_split.csv", usecols=["repo"])["repo"]
    proj_tags = repo.str.replace("/", "_", n=1).tolist()
    per_proj = {}
    for p in _project_list():
        base = RESULTS / "project_specific" / p
        df = _preds(base / model / "ragtag" / "predictions" / f"preds_k{k}.csv").rename(
            columns={"predicted_label": "ragtag"})
        df["bragtag"] = _preds(base / model / "ragtag_debias_m3" / "predictions" / f"preds_k{k}.csv")["predicted_label"]
        df["votag"] = _preds(base / "vtag" / "predictions" / f"preds_k{VTAG_BEST_K_PS}.csv")["predicted_label"]
        df = df.join(_neighbor_mix(p, k))
        per_proj[p] = df
    counters = {p: 0 for p in per_proj}
    rows = []
    for g_idx, p in enumerate(proj_tags):
        li = counters[p]
        counters[p] += 1
        r = per_proj[p].loc[li]
        if r["ground_truth"] != zs.loc[g_idx, "ground_truth"]:
            raise RuntimeError(f"ground-truth misalignment at global {g_idx} ({p}, local {li})")
        rows.append({"project": p, "truth": r["ground_truth"], "zs": zs.loc[g_idx, "predicted_label"],
                     "ragtag": r["ragtag"], "bragtag": r["bragtag"], "votag": r["votag"],
                     "n_bug": int(r["n_bug"]), "n_q": int(r["n_q"]), "n_feat": int(r["n_feat"]),
                     "fired": bool(r["fired"]), "bug_majority": bool(r["bug_majority"])})
    return pd.DataFrame(rows)


def _dist(sub: pd.DataFrame, col: str) -> str:
    n = len(sub)
    if n == 0:
        return "n=0"
    return f"n={n:<4} " + " ".join(f"{o[:4]} {100*(sub[col]==o).mean():4.1f}%" for o in OUTCOMES)


def report(lbl: str, k: int, d: pd.DataFrame, rec: list) -> None:
    q = d[d.truth == "question"]
    zs_wrong = q[q.zs == "bug"]
    print(f"\n{'='*100}\n{lbl}  (RAGTAG-PS k={k}; {len(q)} true questions; zero-shot labeled {len(zs_wrong)} of them bug)")

    print("A. RAGTAG outcome on questions zero-shot got wrong, by VOTAG's vote on the same issue:")
    for v in LABELS:
        print(f"   VOTAG={v:<9} {_dist(zs_wrong[zs_wrong.votag == v], 'ragtag')}")
    print("   ...by label mix of the examples RAGTAG saw:")
    for name, mask in (("bug majority", zs_wrong.bug_majority), ("not bug majority", ~zs_wrong.bug_majority),
                       ("no bug examples", zs_wrong.n_bug == 0)):
        print(f"   {name:<17} {_dist(zs_wrong[mask], 'ragtag')}")
    corr_bm = (zs_wrong[zs_wrong.bug_majority].ragtag == "question").mean() if zs_wrong.bug_majority.any() else float("nan")
    corr_nb = (zs_wrong[~zs_wrong.bug_majority].ragtag == "question").mean() if (~zs_wrong.bug_majority).any() else float("nan")

    still = q[q.ragtag == "bug"]
    base = (q.votag == "bug").mean()
    ov = (still.votag == "bug").mean()
    ov_bm = still.bug_majority.mean()
    print(f"B. Questions RAGTAG still labels bug: n={len(still)} ({100*len(still)/len(q):.1f}% of questions). "
          f"VOTAG also bug: {100*ov:.1f}% (base rate over all questions {100*base:.1f}%); "
          f"examples bug-majority: {100*ov_bm:.1f}% (base {100*q.bug_majority.mean():.1f}%)")

    print("C. BRAGTAG (same k) on questions RAGTAG labels bug, by debias trigger:")
    for name, mask in (("fired", still.fired), ("not fired", ~still.fired)):
        print(f"   {name:<10} {_dist(still[mask], 'bragtag')}")
    nf = d[~d.fired]
    mism = (nf.ragtag != nf.bragtag).mean() if len(nf) else float("nan")
    print(f"   sanity: non-fired prompts identical -> RAGTAG/BRAGTAG disagree on {100*mism:.2f}% of {len(nf)} issues")
    fired_q = q[q.fired]
    print(f"   trigger fired for {100*q.fired.mean():.1f}% of questions, "
          f"{100*d[d.truth=='bug'].fired.mean():.1f}% of bugs, {100*d[d.truth=='feature'].fired.mean():.1f}% of features")
    b_ok = d[(d.truth == "bug") & (d.ragtag == "bug") & d.fired]
    print(f"   cost: true bugs RAGTAG got right and trigger fired: {_dist(b_ok, 'bragtag')}")
    fixed = still[still.fired]
    corr_brag = (fixed.bragtag == "question").mean() if len(fixed) else float("nan")

    print("D. Within fired questions RAGTAG labels bug: BRAGTAG correction by examples removed / remaining:")
    fx = fixed.assign(removed=fixed.n_bug, remaining=fixed.n_q + fixed.n_feat)
    for name, ser in (("removed", fx.removed), ("remaining", fx.remaining)):
        bins = sorted(ser.unique())
        print(f"   by {name:<9} " + "  ".join(
            f"{b}:{100*(fx[ser==b].bragtag=='question').mean():.0f}%(n={int((ser==b).sum())})" for b in bins))

    rec.append({"model": lbl, "k": k, "n_questions": len(q), "zs_bug": len(zs_wrong),
                "corr_rate_bug_majority": corr_bm, "corr_rate_not_bug_majority": corr_nb,
                "ragtag_still_bug": len(still), "votag_also_bug": ov, "votag_bug_base": base,
                "still_bug_examples_bug_majority": ov_bm, "fired_share_questions": q.fired.mean(),
                "bragtag_corrects_fired": corr_brag})


def main() -> None:
    rec, pooled = [], []
    for tag, lbl in MODELS:
        try:
            k = _best_k(tag)
            d = build(tag, k)
        except FileNotFoundError as e:
            print(f"[skip] {lbl}: {e}", file=sys.stderr)
            continue
        report(lbl, k, d, rec)
        pooled.append(d.assign(model=lbl))
    if len(pooled) > 1:
        report("POOLED (4 sizes)", -1, pd.concat(pooled, ignore_index=True), rec)
    pd.DataFrame(rec).to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()

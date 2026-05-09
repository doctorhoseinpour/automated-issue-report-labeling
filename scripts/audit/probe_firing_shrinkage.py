"""Probe alternative interpretations of the firing rate and shrinkage claims."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "paper"))
from _rescue import _project_list  # noqa

RESULTS = REPO / "results" / "issues11k"


def load_neighbors_ps(k: int) -> pd.DataFrame:
    parts = []
    for proj in _project_list():
        df = pd.read_csv(RESULTS / "project_specific" / proj / "neighbors" / f"neighbors_k{k}.csv")
        df["__proj"] = proj
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


df30 = load_neighbors_ps(30)

print("\n=== FIRE RATE per-k by true label (rule: bug_count>0 AND bug-q<=3) ===")
print("k    fire-Q   fire-B   fire-F")
fire_q_per_k, fire_b_per_k = [], []
for k in range(1, 31):
    sub = df30[df30["neighbor_rank"] < k]
    gp = sub.groupby(["__proj", "test_idx"]).agg(
        true_label=("test_label", "first"),
        n_bug=("neighbor_label", lambda s: (s == "bug").sum()),
        n_q=("neighbor_label", lambda s: (s == "question").sum()),
    )
    fired = (gp["n_bug"] > 0) & ((gp["n_bug"] - gp["n_q"]) <= 3)
    fq = fired[gp["true_label"] == "question"].mean()
    fb = fired[gp["true_label"] == "bug"].mean()
    ff = fired[gp["true_label"] == "feature"].mean()
    fire_q_per_k.append(fq); fire_b_per_k.append(fb)
    print(f"  {k:3d}  {100*fq:5.1f}%  {100*fb:5.1f}%  {100*ff:5.1f}%")

print(f"\nMean fire-Q over k=[1,30]:  {100*np.mean(fire_q_per_k):.1f}%")
print(f"Mean fire-Q over k=[3,30]:  {100*np.mean(fire_q_per_k[2:]):.1f}%")
print(f"Mean fire-Q over k=[1,15]:  {100*np.mean(fire_q_per_k[:15]):.1f}%")
print(f"Mean fire-B over k=[1,30]:  {100*np.mean(fire_b_per_k):.1f}%")
print(f"Mean fire-B over k=[3,30]:  {100*np.mean(fire_b_per_k[2:]):.1f}%")

# Try the rule WITHOUT bug>0 precondition (matches my original audit script)
print("\n=== ALT RULE: bug-q<=margin only (no bug>0 precondition) ===")
print("k    fire-Q   fire-B")
fq_alt, fb_alt = [], []
for k in range(1, 31):
    sub = df30[df30["neighbor_rank"] < k]
    gp = sub.groupby(["__proj", "test_idx"]).agg(
        true_label=("test_label", "first"),
        n_bug=("neighbor_label", lambda s: (s == "bug").sum()),
        n_q=("neighbor_label", lambda s: (s == "question").sum()),
    )
    fired = (gp["n_bug"] - gp["n_q"]) <= 3
    fq = fired[gp["true_label"] == "question"].mean()
    fb = fired[gp["true_label"] == "bug"].mean()
    fq_alt.append(fq); fb_alt.append(fb)
print(f"Mean fire-Q over k=[1,30]: {100*np.mean(fq_alt):.1f}%")
print(f"Mean fire-Q over k=[3,30]: {100*np.mean(fq_alt[2:]):.1f}%")
print(f"Mean fire-B over k=[1,30]: {100*np.mean(fb_alt):.1f}%")
print(f"Mean fire-B over k=[3,30]: {100*np.mean(fb_alt[2:]):.1f}%")

# Shrinkage: a pile of interpretations
print("\n=== SHRINKAGE INTERPRETATIONS at k=1 and k=3 (correct rule with bug>0) ===")
for k in [1, 3, 6, 9, 12, 15]:
    sub = df30[df30["neighbor_rank"] < k]
    gp = sub.groupby(["__proj", "test_idx"]).agg(
        true_label=("test_label", "first"),
        n_bug=("neighbor_label", lambda s: (s == "bug").sum()),
        n_q=("neighbor_label", lambda s: (s == "question").sum()),
        n_total=("neighbor_label", "size"),
    )
    fired = (gp["n_bug"] > 0) & ((gp["n_bug"] - gp["n_q"]) <= 3)
    remaining = np.where(fired, gp["n_total"] - gp["n_bug"], gp["n_total"])

    A = (remaining <= 1).mean()
    B = (remaining == 0).mean()
    C = ((remaining <= 1) & fired).mean()
    D = (remaining == 0).sum() / fired.sum() if fired.sum() else float("nan")
    # E: only over question queries
    qmask = gp["true_label"] == "question"
    E = (remaining[qmask] <= 1).mean()
    F = (remaining[qmask] == 0).mean()
    G = (remaining[qmask & fired] <= 1).mean() if (qmask & fired).any() else float("nan")
    print(f"  k={k}: A(<=1 all)={100*A:.1f}%  B(=0 all)={100*B:.1f}%  "
          f"C(<=1 & fired)={100*C:.1f}%  D(=0 | fired)={100*D:.1f}%  "
          f"E(<=1 | Q)={100*E:.1f}%  F(=0 | Q)={100*F:.1f}%  G(<=1 | Q & fired)={100*G:.1f}%")

# H: maybe "ends with ≤ 1 example" means after debias the LLM sees <=1 example,
# computed over ALL queries (since the scope of the claim is unclear)
# Try yet another: fraction where (after firing) remaining < original k.
print("\n=== another reading: trigger fires AT ALL ===")
for k in [1, 3, 6, 9]:
    sub = df30[df30["neighbor_rank"] < k]
    gp = sub.groupby(["__proj", "test_idx"]).agg(
        n_bug=("neighbor_label", lambda s: (s == "bug").sum()),
        n_q=("neighbor_label", lambda s: (s == "question").sum()),
    )
    fired = (gp["n_bug"] > 0) & ((gp["n_bug"] - gp["n_q"]) <= 3)
    print(f"  k={k}: trigger fires for {100*fired.mean():.1f}% of all queries")

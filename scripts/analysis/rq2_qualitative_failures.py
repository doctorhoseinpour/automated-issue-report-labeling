"""RQ2.8: Qualitative failure samples.

Sample test issues that ALL four Qwen sizes' best plain RAGTAG (agnostic)
mislabel as bug despite ground truth being feature or question. Output a
markdown table with the issue text plus auto-categorization heuristics
(error-trace presence, question phrasing, "feature request" cues).

The categorization is a starting point for hand review, not a final label.

Outputs:
  docs/analysis/rq2_qualitative_failures.csv     # full data
  docs/analysis/rq2_qualitative_failures.md      # 30-issue review-ready table
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _utils import (  # noqa: E402
    DOCS_ANALYSIS,
    MODEL_ORDER,
    REPO_ROOT,
    ensure_dirs,
    load_predictions,
    rel,
)


# Best plain RAGTAG agnostic k per model (from rq2_leaderboard).
BEST_RAGTAG_AG_K = {
    "Qwen-3B": "k3",
    "Qwen-7B": "k6",
    "Qwen-14B": "k9",
    "Qwen-32B": "k9",
}


ERROR_PATTERNS = [
    r"\btraceback\b", r"\berror:\s", r"\bexception\b",
    r"\b[A-Z][a-zA-Z]+Error\b",
    r"\bsegfault\b", r"\bsegmentation fault\b",
    r"\bstack trace\b", r"\bcrash\b",
    r"\bfailed to\b", r"\bcannot find\b",
]
QUESTION_PATTERNS = [
    r"^how (do|can|should)\b", r"^what (is|does|are)\b",
    r"^why (does|is)\b", r"^is it possible\b",
    r"^can (i|we|you)\b", r"\?\s*$",
]
FEATURE_PATTERNS = [
    r"\bfeature request\b", r"\bproposal\b", r"\bsupport for\b",
    r"\badd support\b", r"\bplease add\b", r"\bit would be (nice|great|helpful)\b",
    r"\benhancement\b", r"\bRFC\b",
]


def _categorize(title: str, body: str) -> str:
    text = f"{title}\n{body}".lower()[:1500]
    cats: list[str] = []
    if any(re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in ERROR_PATTERNS):
        cats.append("error-trace")
    if any(re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in QUESTION_PATTERNS):
        cats.append("question-phrased")
    if any(re.search(p, text, re.IGNORECASE) for p in FEATURE_PATTERNS):
        cats.append("feature-cue")
    return ",".join(cats) if cats else "ambiguous"


def main() -> None:
    ensure_dirs()
    preds_index = pd.read_csv(DOCS_ANALYSIS / "preds_index.csv")

    # Load best agnostic plain RAGTAG predictions per model.
    pred_per_model: dict[str, pd.DataFrame] = {}
    for model in MODEL_ORDER:
        k_label = BEST_RAGTAG_AG_K[model]
        rows = preds_index[
            (preds_index["model"] == model)
            & (preds_index["approach"] == "ragtag")
            & (preds_index["setting"] == "agnostic")
            & (preds_index["project"] == "_overall")
            & (preds_index["k_label"] == k_label)
        ]
        if rows.empty:
            print(f"WARN: no predictions for {model} ag {k_label}")
            continue
        df = load_predictions(REPO_ROOT / rows.iloc[0]["predictions_path"])
        df = df[["test_idx", "ground_truth", "predicted_label", "title", "body"]].rename(
            columns={"predicted_label": f"pred_{model}"},
        )
        pred_per_model[model] = df

    # Inner-join on test_idx to get aligned predictions.
    base = None
    for model, df in pred_per_model.items():
        if base is None:
            base = df
        else:
            base = base.merge(df[["test_idx", f"pred_{model}"]], on="test_idx", how="inner")
    assert base is not None

    # Identify consensus-bug failures: GT in {feature, question}, all models predicted bug.
    pred_cols = [f"pred_{m}" for m in MODEL_ORDER]
    base["all_bug"] = base[pred_cols].apply(
        lambda r: all(str(x).lower().strip() == "bug" for x in r),
        axis=1,
    )
    failures = base[
        (base["ground_truth"].str.lower().isin(["feature", "question"]))
        & (base["all_bug"])
    ].copy()
    failures["body_excerpt"] = failures["body"].fillna("").astype(str).str[:300]
    failures["category"] = failures.apply(
        lambda r: _categorize(r["title"], r["body_excerpt"]),
        axis=1,
    )
    failures.to_csv(DOCS_ANALYSIS / "rq2_qualitative_failures.csv", index=False)
    print(f"wrote {rel(DOCS_ANALYSIS / 'rq2_qualitative_failures.csv')} ({len(failures)} failures)")

    # Category counts.
    print("\n--- Auto-categorization counts (consensus-bug-failures) ---")
    cats = failures["category"].value_counts()
    print(cats.to_string())
    print()
    by_gt = failures.groupby(["ground_truth", "category"]).size().unstack(fill_value=0)
    print("By ground truth:")
    print(by_gt.to_string())

    # Sample 30 for the markdown table — 15 features + 15 questions.
    parts: list[pd.DataFrame] = []
    for gt in ("feature", "question"):
        sub = failures[failures["ground_truth"].str.lower() == gt]
        if not sub.empty:
            parts.append(sub.sample(min(15, len(sub)), random_state=0))
    sample = pd.concat(parts, ignore_index=True) if parts else failures.head(0)

    md_lines: list[str] = []
    md_lines.append("# RQ2.8 — Qualitative failure samples")
    md_lines.append("")
    md_lines.append(
        "Issues where all four Qwen sizes' best plain RAGTAG (agnostic) "
        "predicted **bug** despite ground truth being **feature** or **question**.",
    )
    md_lines.append("")
    md_lines.append(f"Total consensus-bug failures: **{len(failures)}** "
                    f"(of {len(base)} test issues = {len(failures)/len(base):.1%}).")
    md_lines.append("")
    md_lines.append("| # | GT | Auto-cat | Title | Body excerpt |")
    md_lines.append("|---:|---|---|---|---|")
    for i, (_, r) in enumerate(sample.iterrows(), 1):
        title = str(r["title"]).replace("|", "\\|")[:80]
        excerpt = (
            str(r["body_excerpt"])
            .replace("\n", " ")
            .replace("\r", " ")
            .replace("|", "\\|")
        )[:150]
        md_lines.append(
            f"| {i} | {r['ground_truth']} | {r['category']} | {title} | {excerpt}... |",
        )
    md_lines.append("")
    md_lines.append("## Hand-review codebook")
    md_lines.append("")
    md_lines.append(
        "- **error-trace**: contains traceback, exception, or *Error* class names — looks bug-like even if it's a question or a feature.",
    )
    md_lines.append(
        "- **question-phrased**: starts with how/what/why/is-it-possible/can-i, or ends with `?` — looks like a question (even if labeled as feature).",
    )
    md_lines.append(
        "- **feature-cue**: phrases like 'feature request', 'proposal', 'add support', 'enhancement' — should have been classified feature.",
    )
    md_lines.append(
        "- **ambiguous**: none of the above — likely labeling noise or genuinely ambiguous.",
    )

    out_md = DOCS_ANALYSIS / "rq2_qualitative_failures.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"\nwrote {rel(out_md)}")


if __name__ == "__main__":
    main()

"""Shared helpers for the 11k analysis scripts.

All scripts read from results/ (read-only) and write to docs/analysis/.
Run from repo root.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "issues11k"
DOCS_ANALYSIS = REPO_ROOT / "docs" / "analysis"
FIGURES = DOCS_ANALYSIS / "figures"

MODEL_TAG_TO_NAME = {
    "unsloth_Qwen2_5_3B_Instruct_bnb_4bit": "Qwen-3B",
    "unsloth_Qwen2_5_7B_Instruct_bnb_4bit": "Qwen-7B",
    "unsloth_Qwen2_5_14B_Instruct_bnb_4bit": "Qwen-14B",
    "unsloth_Qwen2_5_32B_Instruct_bnb_4bit": "Qwen-32B",
}
MODEL_ORDER = ["Qwen-3B", "Qwen-7B", "Qwen-14B", "Qwen-32B"]
ACTIVE_MODEL_TAGS = list(MODEL_TAG_TO_NAME.keys())

PROJECTS = [
    "ansible_ansible", "bitcoin_bitcoin", "dart-lang_sdk", "dotnet_roslyn",
    "facebook_react", "flutter_flutter", "kubernetes_kubernetes",
    "microsoft_TypeScript", "microsoft_vscode", "opencv_opencv",
    "tensorflow_tensorflow",
]

# v1/v2 resolution: which (model_tag, setting, approach_dir_basename) pairs
# have a *_v2 directory we should prefer.
V2_PREFERENCE = {
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "agnostic", "ragtag"),
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "project_specific", "ragtag"),
    ("unsloth_Qwen2_5_32B_Instruct_bnb_4bit", "project_specific", "ragtag_debias_m3"),
    ("unsloth_Qwen2_5_14B_Instruct_bnb_4bit", "project_specific", "ragtag_debias_m3"),
}


def parse_k_from_eval_filename(name: str) -> tuple[str, int | None]:
    """Return (k_label, k_int) from an eval CSV filename.

    eval_zero_shot.csv -> ('zero_shot', 0)
    eval_k9.csv        -> ('k9', 9)
    eval_finetune_fixed.csv -> ('finetune_fixed', None)
    """
    base = os.path.basename(name).removeprefix("eval_").removesuffix(".csv")
    if base == "zero_shot":
        return "zero_shot", 0
    m = re.match(r"^k(\d+)$", base)
    if m:
        return base, int(m.group(1))
    return base, None


def model_name(tag: str) -> str:
    return MODEL_TAG_TO_NAME.get(tag, tag)


def ensure_dirs() -> None:
    DOCS_ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


VALID_LABELS = ["bug", "feature", "question"]


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Mirror evaluate.py: lowercase+strip, invalid counts as wrong, macro+weighted."""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
    )

    yt = [str(x).lower().strip() for x in y_true]
    yp = [str(x).lower().strip() for x in y_pred]
    total = len(yt)
    n_invalid = sum(1 for p in yp if p == "invalid")

    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        yt, yp, labels=VALID_LABELS, zero_division=0,
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        yt, yp, labels=VALID_LABELS, average="macro", zero_division=0,
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        yt, yp, labels=VALID_LABELS, average="weighted", zero_division=0,
    )
    acc = accuracy_score(yt, yp) if total > 0 else 0.0

    out = {
        "total_issues": total,
        "invalid_count": n_invalid,
        "invalid_rate": round(n_invalid / total if total else 0.0, 4),
        "accuracy": round(acc, 4),
        "precision_macro": round(p_macro, 4),
        "recall_macro": round(r_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(p_weighted, 4),
        "recall_weighted": round(r_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
    }
    for i, lab in enumerate(VALID_LABELS):
        out[f"precision_{lab}"] = round(p_per[i], 4)
        out[f"recall_{lab}"] = round(r_per[i], 4)
        out[f"f1_{lab}"] = round(f1_per[i], 4)
        out[f"support_{lab}"] = int(sup_per[i])
    return out


def project_for_test_idx() -> dict[int, str]:
    """Map test_idx -> project_tag (e.g., 'ansible_ansible')."""
    import pandas as pd

    test_split = pd.read_csv(RESULTS_DIR / "agnostic" / "neighbors" / "test_split.csv")
    # repo column is like 'ansible/ansible'; project tag uses underscore.
    repos = test_split["repo"].astype(str).tolist()
    return {i: repo.replace("/", "_") for i, repo in enumerate(repos)}


def load_predictions(pred_path: Path) -> "pd.DataFrame":
    """Read a predictions CSV; predictions may have multi-line body fields."""
    import pandas as pd

    return pd.read_csv(pred_path, dtype={"test_idx": "Int64"}, engine="python",
                       on_bad_lines="skip")

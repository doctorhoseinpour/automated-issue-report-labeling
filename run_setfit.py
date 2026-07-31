#!/usr/bin/env python3
"""
run_setfit.py
=============
SetFit baseline for issue classification on the 11k benchmark.

Replicates the literature configuration used by Colavito et al. (IST 2025,
JSS 2026) and the NLBSE'24 competition baseline: sentence-transformer body,
contrastive fine-tune with num_iterations=20 pair sampling, 1 body epoch,
batch size 16, LogisticRegression head, seed 42.

Runs inside venv-setfit/ (setfit pins transformers<5; the main venv has 5.x).

Outputs match run_transformer_ft.py / the LLM-FT pipeline so evaluate.py and
downstream analyses work uniformly:
  predictions/preds_setfit.csv   — same schema family
  evaluations/eval_setfit.csv    — produced by evaluate.py
  cost_metrics.csv               — same fields as Unsloth FT

Usage:
  venv-setfit/bin/python run_setfit.py \\
      --body_model sentence-transformers/all-mpnet-base-v2 \\
      --train_csv results/issues11k/agnostic/neighbors/train_split.csv \\
      --test_csv  results/issues11k/agnostic/neighbors/test_split.csv \\
      --output_dir results/issues11k/agnostic/sentence-transformers_all-mpnet-base-v2/setfit
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LABELS = ["bug", "feature", "question"]


def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    """Splits use 'labels' (plural) per neighbors/train_split.csv."""
    if "labels" in df.columns:
        df = df.copy()
        df["label"] = df["labels"].astype(str).str.lower()
    elif "label" in df.columns:
        df = df.copy()
        df["label"] = df["label"].astype(str).str.lower()
    else:
        raise ValueError(f"No label column found in DataFrame: {list(df.columns)}")
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    return df


def issue_text(row) -> str:
    return f"Title: {row['title']}\nBody: {row['body']}"


def main():
    parser = argparse.ArgumentParser(description="SetFit baseline for IRC")
    parser.add_argument("--body_model", default="sentence-transformers/all-mpnet-base-v2",
                        help="Sentence-transformer body checkpoint")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True,
                        help="Will create predictions/ and evaluations/ subdirs underneath")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=20,
                        help="Contrastive pair-generation iterations (Colavito et al. config)")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_for_eval", default=None)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    pred_dir = out_root / "predictions"
    eval_dir = out_root / "evaluations"
    pred_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    preds_csv = pred_dir / "preds_setfit.csv"
    eval_csv = eval_dir / "eval_setfit.csv"
    cost_csv = out_root / "cost_metrics.csv"

    if preds_csv.exists():
        print(f"  SKIP: {preds_csv} already exists")
        return

    print("=" * 60)
    print("  SetFit baseline for IRC")
    print("=" * 60)
    print(f"  Body model:     {args.body_model}")
    print(f"  Train CSV:      {args.train_csv}")
    print(f"  Test CSV:       {args.test_csv}")
    print(f"  Output:         {pred_dir}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Body epochs:    {args.num_epochs}")
    print(f"  Num iterations: {args.num_iterations}")
    print(f"  Seed:           {args.seed}")
    print("=" * 60)

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------
    train_df = normalize_label_column(pd.read_csv(args.train_csv))
    test_df = normalize_label_column(pd.read_csv(args.test_csv))
    print(f"\nTrain: {len(train_df)} rows, Test: {len(test_df)} rows")

    # ------------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------------
    print(f"\nLoading body model {args.body_model}...")
    t0 = time.time()
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    from transformers import set_seed

    set_seed(args.seed)
    model = SetFitModel.from_pretrained(args.body_model, labels=LABELS)
    model_load_time = time.time() - t0
    max_seq_length = int(model.model_body.max_seq_length)
    print(f"  loaded in {model_load_time:.1f}s  (body max_seq_length={max_seq_length})")

    # ------------------------------------------------------------------------
    # Token stats (one-time pass over test set for cost reporting)
    # ------------------------------------------------------------------------
    print("\nComputing token stats on test set...")
    tokenizer = model.model_body.tokenizer
    test_token_lens = []
    for _, row in test_df.iterrows():
        ids = tokenizer.encode(issue_text(row), truncation=True, max_length=max_seq_length)
        test_token_lens.append(len(ids))

    # ------------------------------------------------------------------------
    # Train (contrastive body fine-tune + LogisticRegression head)
    # ------------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_ds = Dataset.from_dict({
        "text": [issue_text(r) for _, r in train_df.iterrows()],
        "label": train_df["label"].tolist(),
    })

    training_args = TrainingArguments(
        output_dir=str(out_root / "_trainer_output"),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_iterations=args.num_iterations,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        column_mapping={"text": "text", "label": "label"},
    )

    print(f"\nTraining (num_iterations={args.num_iterations}, epochs={args.num_epochs})...")
    train_t0 = time.time()
    trainer.train()
    training_time = time.time() - train_t0
    train_peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0
    print(f"  training done in {training_time:.1f}s  (peak GPU train: {train_peak_mb:.0f} MB)")

    # ------------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------------
    print(f"\nInference on test set ({len(test_df)} issues)...")
    test_texts = [issue_text(r) for _, r in test_df.iterrows()]

    inf_t0 = time.time()
    probs = model.predict_proba(test_texts, batch_size=args.eval_batch_size)
    probs = np.asarray(probs.cpu() if torch.is_tensor(probs) else probs)
    head_classes = [str(c) for c in np.asarray(model.model_head.classes_).tolist()]
    preds = [head_classes[i] for i in probs.argmax(-1)]
    raw_outputs = [json.dumps(dict(zip(head_classes, p.tolist()))) for p in probs]
    inference_time = time.time() - inf_t0
    final_peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0
    print(f"  inference done in {inference_time:.1f}s  ({len(test_df) / inference_time:.2f} issues/s)")

    # ------------------------------------------------------------------------
    # Write predictions
    # ------------------------------------------------------------------------
    out_df = pd.DataFrame({
        "test_idx": list(range(len(test_df))),
        "title": test_df["title"].tolist(),
        "body": test_df["body"].tolist(),
        "ground_truth": test_df["label"].tolist(),
        "predicted_label": preds,
        "raw_output": raw_outputs,
        "truncated": [tl >= max_seq_length for tl in test_token_lens],
        "neighbors_truncated": False,
        "query_truncated": [tl >= max_seq_length for tl in test_token_lens],
        "tokens_removed": 0,
        "parsed_via": "argmax",
        "prompt_tokens": test_token_lens,
        "generated_tokens": 0,
    })
    out_df.to_csv(preds_csv, index=False)
    print(f"  preds → {preds_csv}")

    # ------------------------------------------------------------------------
    # Cost metrics (matches LLM-FT schema for direct comparison)
    # ------------------------------------------------------------------------
    cost_stats = {
        "model": args.body_model,
        "top_k": "N/A",
        "k_label": "setfit",
        "model_load_time_s": round(model_load_time, 2),
        "training_time_s": round(training_time, 2),
        "wall_time_s": round(inference_time, 2),
        "issues_per_second": round(len(test_df) / inference_time, 3) if inference_time > 0 else 0,
        "total_issues": len(test_df),
        "total_prompt_tokens": int(sum(test_token_lens)),
        "total_generated_tokens": 0,
        "avg_prompt_tokens": round(sum(test_token_lens) / len(test_token_lens), 1),
        "avg_generated_tokens": 0,
        "min_prompt_tokens": int(min(test_token_lens)),
        "max_prompt_tokens": int(max(test_token_lens)),
        "gpu_peak_memory_mb": round(final_peak_mb, 0),
        "gpu_peak_memory_training_mb": round(train_peak_mb, 0),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_total_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 2, 0)
        if torch.cuda.is_available() else 0,
        "max_seq_length": max_seq_length,
        "max_new_tokens": "N/A",
        "load_in_4bit": False,
        # setfit-specific extras
        "body_model": args.body_model,
        "num_epochs": args.num_epochs,
        "num_iterations": args.num_iterations,
        "batch_size": args.batch_size,
        "head": type(model.model_head).__name__,
        "seed": args.seed,
    }
    pd.DataFrame([cost_stats]).to_csv(cost_csv, index=False)
    print(f"  cost  → {cost_csv}")

    # ------------------------------------------------------------------------
    # Auto-evaluate
    # ------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, "evaluate.py")
    if os.path.exists(eval_script):
        model_name = args.model_name_for_eval or args.body_model.replace("/", "_")
        print(f"\nEvaluating via evaluate.py...")
        subprocess.run([
            sys.executable, eval_script,
            "--preds_csv", str(preds_csv),
            "--top_k", "0",
            "--output_csv", str(eval_csv),
            "--model_name", model_name,
        ], check=False)

    print(f"\nTotal wall: load={model_load_time:.1f}s + train={training_time:.1f}s + infer={inference_time:.1f}s")


if __name__ == "__main__":
    main()

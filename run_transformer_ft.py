#!/usr/bin/env python3
"""
run_transformer_ft.py
=====================
Fine-tune a transformer encoder (e.g., DeBERTa-v3-large) for issue
classification on the 11k benchmark, agnostic setting.

Outputs match the existing LLM-FT pipeline so evaluate.py and downstream
analyses work uniformly:
  predictions/preds_finetune_transformer.csv   — same schema family
  evaluations/eval_finetune_transformer.csv    — produced by evaluate.py
  cost_metrics.csv                             — same fields as Unsloth FT

Usage:
  python run_transformer_ft.py \\
      --model microsoft/deberta-v3-large \\
      --train_csv results/issues11k/agnostic/neighbors/train_split.csv \\
      --test_csv  results/issues11k/agnostic/neighbors/test_split.csv \\
      --output_dir results/issues11k/agnostic/microsoft_deberta-v3-large/finetune_transformer
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

LABELS = ["bug", "feature", "question"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


class IssuesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = f"Title: {row['title']}\nBody: {row['body']}"
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        enc["labels"] = LABEL2ID[str(row["label"]).lower()]
        return enc


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


def main():
    parser = argparse.ArgumentParser(description="Fine-tune transformer encoder for IRC")
    parser.add_argument("--model", default="microsoft/deberta-v3-large",
                        help="HuggingFace model id (encoder for sequence classification)")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True,
                        help="Will create predictions/ and evaluations/ subdirs underneath")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_for_eval", default=None)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    pred_dir = out_root / "predictions"
    eval_dir = out_root / "evaluations"
    pred_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    preds_csv = pred_dir / "preds_finetune_transformer.csv"
    eval_csv = eval_dir / "eval_finetune_transformer.csv"
    cost_csv = out_root / "cost_metrics.csv"

    if preds_csv.exists():
        print(f"  SKIP: {preds_csv} already exists")
        return

    print("=" * 60)
    print("  Transformer fine-tune (encoder) for IRC")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Train CSV:   {args.train_csv}")
    print(f"  Test CSV:    {args.test_csv}")
    print(f"  Output:      {pred_dir}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size} (train), {args.eval_batch_size} (eval)")
    print(f"  LR:          {args.lr}")
    print(f"  Max seq len: {args.max_seq_length}")
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
    print(f"\nLoading model {args.model}...")
    t0 = time.time()
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3, label2id=LABEL2ID, id2label=ID2LABEL,
    )
    model_load_time = time.time() - t0
    print(f"  loaded in {model_load_time:.1f}s")

    # ------------------------------------------------------------------------
    # Token stats (one-time pass over test set for cost reporting)
    # ------------------------------------------------------------------------
    print("\nComputing token stats on test set...")
    test_token_lens = []
    for _, row in test_df.iterrows():
        text = f"Title: {row['title']}\nBody: {row['body']}"
        ids = tokenizer.encode(text, truncation=True, max_length=args.max_seq_length)
        test_token_lens.append(len(ids))

    # ------------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_ds = IssuesDataset(train_df, tokenizer, max_length=args.max_seq_length)
    test_ds = IssuesDataset(test_df, tokenizer, max_length=args.max_seq_length)

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(out_root / "_trainer_output"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        bf16=bf16_supported,
        fp16=not bf16_supported and torch.cuda.is_available(),
        remove_unused_columns=False,
        seed=args.seed,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    print(f"\nTraining {args.epochs} epochs...")
    train_t0 = time.time()
    trainer.train()
    training_time = time.time() - train_t0
    train_peak_mb = (torch.cuda.max_memory_allocated() / 1024 ** 2) if torch.cuda.is_available() else 0
    print(f"  training done in {training_time:.1f}s  (peak GPU train: {train_peak_mb:.0f} MB)")

    # ------------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------------
    print(f"\nInference on test set ({len(test_df)} issues)...")
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    inf_t0 = time.time()
    preds: list[str] = []
    raw_outputs: list[str] = []
    bs = args.eval_batch_size
    for batch_start in range(0, len(test_df), bs):
        batch = test_df.iloc[batch_start: batch_start + bs]
        texts = [f"Title: {r['title']}\nBody: {r['body']}" for _, r in batch.iterrows()]
        enc = tokenizer(
            texts, truncation=True, max_length=args.max_seq_length,
            padding=True, return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            logits = model(**enc).logits
        probs = logits.softmax(-1)
        ids = logits.argmax(-1).cpu().tolist()
        for i, pid in enumerate(ids):
            preds.append(ID2LABEL[pid])
            raw_outputs.append(json.dumps(probs[i].cpu().numpy().tolist()))
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
        "truncated": [tl >= args.max_seq_length for tl in test_token_lens],
        "neighbors_truncated": False,
        "query_truncated": [tl >= args.max_seq_length for tl in test_token_lens],
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
        "model": args.model,
        "top_k": "N/A",
        "k_label": "finetune_transformer",
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
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": "N/A",
        "load_in_4bit": False,
        # transformer-specific extras
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "warmup_ratio": args.warmup_ratio,
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
        model_name = args.model_name_for_eval or args.model.replace("/", "_")
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

#!/usr/bin/env python3
"""
fixed_fine-tune.py
==================
Corrected fine-tuning pipeline for GitHub issue classification.

Fixes over the original notebook:
  - Same prompt template for training and inference (no mismatch)
  - No chain-of-thought prefix — direct label output
  - Proper tokenizer.eos_token (not hardcoded)
  - Strict token truncation during training data formatting
  - num_train_epochs=1 (trains on full data, not just 60 steps)
  - paged_adamw_8bit optimizer (lower VRAM via CPU offload)
  - Training data is deduplicated and cross-set deduped (no leakage)
  - All predictions included in output (invalids not skipped)

GOLDEN TEST SET: Derived from the deduplicated dataset using balanced top-N,
identical to build_and_query_index.py (RAGTAG).

Usage:
  python fixed_fine-tune.py \
    --model unsloth/Llama-3.2-3B-Instruct \
    --dataset issues11k.csv \
    --train_csv issues11k_train.csv \
    --test_csv issues11k_test.csv \
    --output_dir results/issues11k/agnostic/<tag>/finetune_fixed
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import logging
import warnings

import pandas as pd
import torch
import psutil
from tqdm import tqdm
from sklearn.metrics import classification_report
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments

# Suppress the max_new_tokens / max_length warning
logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*", category=UserWarning)

# ============================================================================
# PROMPT — SAME FOR TRAINING AND INFERENCE
# ============================================================================

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
You are a GitHub issue classifier. Classify the following GitHub issue based on its title and description.
You must respond with exactly one word: 'bug', 'feature', or 'question'. Do not provide any other text.

### Issue:
{}

### Response:
{}"""


# ============================================================================
# MONITORING
# ============================================================================

class SystemMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.stage_time = time.time()
        self.performance_logs = []

    def log(self, stage_name):
        now = time.time()
        elapsed_total = now - self.start_time
        elapsed_stage = now - self.stage_time
        self.stage_time = now

        ram = psutil.virtual_memory()
        ram_gb_left = ram.available / (1024 ** 3)
        ram_gb_total = ram.total / (1024 ** 3)

        print(f"\n{'-' * 50}")
        print(f"  STAGE: {stage_name}")
        print(f"  Time since last stage: {elapsed_stage:.2f}s | Total elapsed: {elapsed_total:.2f}s")
        print(f"  System RAM Available: {ram_gb_left:.2f} GB / {ram_gb_total:.2f} GB")

        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  GPU VRAM Allocated:   {vram_allocated:.2f} GB / {vram_total:.2f} GB")
            print(f"  GPU VRAM Reserved:    {vram_reserved:.2f} GB")
        print(f"{'-' * 50}\n")

    def record_phase(self, phase_name, start_time):
        duration = time.time() - start_time
        ram = psutil.virtual_memory()
        ram_used_gb = (ram.total - ram.available) / (1024 ** 3)
        vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0
        self.performance_logs.append({
            "Phase": phase_name,
            "Duration_Seconds": round(duration, 2),
            "RAM_Used_GB": round(ram_used_gb, 2),
            "VRAM_Allocated_GB": round(vram_allocated, 2),
            "VRAM_Reserved_GB": round(vram_reserved, 2),
        })


# ============================================================================
# DEDUP + GOLDEN TEST SET (shared logic with flawed baseline & RAGTAG)
# ============================================================================

def _dedup_key(row) -> str:
    t = str(row.get("title", "") or "").strip().lower()
    b = str(row.get("body", "") or "").strip().lower()
    return hashlib.md5(f"{t}||{b}".encode("utf-8")).hexdigest()


def parse_test_size(value: str):
    f = float(value)
    if 0.0 < f < 1.0:
        return f
    return int(f)


def get_golden_test_indices(deduped_df: pd.DataFrame, test_size):
    """
    From a deduplicated dataframe (with ORIGINAL indices preserved),
    select balanced top-N test indices. Returns a sorted list of
    original-index values that form the golden test set.

    This is identical to the split logic in build_and_query_index.py.
    """
    label_col = "labels"
    labels = sorted(deduped_df[label_col].unique())
    n_labels = len(labels)

    if isinstance(test_size, float):
        per_label_counts = {}
        for lab in labels:
            group_size = (deduped_df[label_col] == lab).sum()
            per_label_counts[lab] = int(group_size * test_size)
    else:
        per_label = test_size // n_labels
        per_label_counts = {lab: per_label for lab in labels}

    test_indices = []
    for lab in labels:
        group = deduped_df[deduped_df[label_col] == lab]
        n_test = min(per_label_counts[lab], len(group))
        test_indices.extend(group.index[:n_test].tolist())

    return sorted(test_indices)


# ============================================================================
# DATASET FORMATTING WITH STRICT TRUNCATION
# ============================================================================

def format_train_data(df, tokenizer, max_seq_length):
    """Format training examples with strict token truncation."""
    formatted = []

    empty_prompt = PROMPT_TEMPLATE.format("", "")
    prompt_overhead = len(tokenizer.encode(empty_prompt, add_special_tokens=False))
    max_issue_tokens = max_seq_length - prompt_overhead - 30  # 10 for answer + 20 buffer

    n_truncated = 0
    for _, row in df.iterrows():
        issue_text = f"Title: {row['title']}\nBody: {row['body']}"

        token_ids = tokenizer.encode(issue_text, add_special_tokens=False)
        if len(token_ids) > max_issue_tokens:
            token_ids = token_ids[:max_issue_tokens]
            issue_text = tokenizer.decode(token_ids) + "\n...[TRUNCATED]"
            n_truncated += 1

        text = PROMPT_TEMPLATE.format(issue_text, row['labels']) + tokenizer.eos_token
        formatted.append({"text": text})

    print(f"  Formatted {len(formatted)} training examples "
          f"({n_truncated} truncated at {max_seq_length} tokens)")
    return formatted


# ============================================================================
# LABEL PARSING
# ============================================================================

def parse_label(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "invalid"
    match = re.search(r'\b(bug|feature|question)\b', raw, re.IGNORECASE)
    if match:
        return match.group(0).lower()
    return "invalid"


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fixed Fine-Tune Pipeline for Issue Classification"
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=10,
                        help="Max tokens for inference output (default: 10)")
    parser.add_argument("--inference_batch_size", type=int, default=1,
                        help="Batch size for inference (default: 1)")
    parser.add_argument("--test_size", type=str, default="0.5",
                        help="Float (0,1) = fraction; int >= 1 = absolute count")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Pre-computed train split CSV (skips internal splitting)")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Pre-computed test split CSV (skips internal splitting)")
    parser.add_argument("--output_dir", type=str, default="results/fixed_finetune",
                        help="Directory for all outputs")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, load saved adapters")
    parser.add_argument("--adapter_dir", type=str, default=None,
                        help="Directory to save/load LoRA adapters")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace model cache directory")
    parser.add_argument("--skip_save_adapter", action="store_true",
                        help="Skip saving LoRA adapter weights post-training (used for ephemeral runs where the adapter is never reloaded)")
    args = parser.parse_args()

    test_size = parse_test_size(args.test_size)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.adapter_dir is None:
        safe_model = args.model.replace("/", "_")
        args.adapter_dir = os.path.join(args.output_dir, f"adapters_{safe_model}")

    monitor = SystemMonitor()
    monitor.log("Initialization")

    # ==================================================================
    # 1. DERIVE GOLDEN TEST SET, THEN BUILD CLEAN TRAIN SET
    # ==================================================================
    if args.train_csv and args.test_csv:
        # --- Use pre-computed splits (from RAGTAG pipeline) ---
        print(f"Using pre-computed splits:")
        print(f"  Train: {args.train_csv}")
        print(f"  Test:  {args.test_csv}")
        train_df = pd.read_csv(args.train_csv)
        train_df['body'] = train_df['body'].fillna("")
        train_df['title'] = train_df['title'].fillna("")
        if 'label' in train_df.columns and 'labels' not in train_df.columns:
            train_df = train_df.rename(columns={'label': 'labels'})
        train_df['labels'] = train_df['labels'].astype(str).str.lower().str.strip()

        golden_test_df = pd.read_csv(args.test_csv)
        golden_test_df['body'] = golden_test_df['body'].fillna("")
        golden_test_df['title'] = golden_test_df['title'].fillna("")
        if 'label' in golden_test_df.columns and 'labels' not in golden_test_df.columns:
            golden_test_df = golden_test_df.rename(columns={'label': 'labels'})
        golden_test_df['labels'] = golden_test_df['labels'].astype(str).str.lower().str.strip()

        labels = sorted(set(train_df['labels'].unique()) | set(golden_test_df['labels'].unique()))
        print(f"\n  Pre-computed splits:")
        for lab in labels:
            n_test = (golden_test_df['labels'] == lab).sum()
            n_train = (train_df['labels'] == lab).sum()
            print(f"    {lab}: test={n_test}, train={n_train}")
        print(f"  Total: test={len(golden_test_df)}, train={len(train_df)} (external splits)")
    else:
        # --- Internal splitting (original behavior) ---
        print(f"Loading dataset: {args.dataset}")
        raw_df = pd.read_csv(args.dataset)
        raw_df['body'] = raw_df['body'].fillna("")
        raw_df['title'] = raw_df['title'].fillna("")
        if 'label' in raw_df.columns and 'labels' not in raw_df.columns:
            raw_df = raw_df.rename(columns={'label': 'labels'})
        raw_df['labels'] = raw_df['labels'].astype(str).str.lower().str.strip()

        print(f"  Raw dataset: {len(raw_df)} issues")

        # Step A: Deduplicate (keeping first occurrence, preserving original indices)
        dedup_keys = raw_df.apply(_dedup_key, axis=1)
        dedup_mask = ~dedup_keys.duplicated(keep="first")
        deduped_df = raw_df[dedup_mask]  # DO NOT reset_index — keep original indices
        n_dupes = (~dedup_mask).sum()
        print(f"  Deduplication: {n_dupes} duplicates found, {len(deduped_df)} unique issues")

        # Step B: Get golden test indices from deduplicated data
        golden_test_indices = get_golden_test_indices(deduped_df, test_size)
        golden_test_df = raw_df.loc[golden_test_indices].reset_index(drop=True)

        # Step C: CLEAN train set = deduped data minus golden test rows
        train_df = deduped_df.drop(index=golden_test_indices).reset_index(drop=True)

        # Step D: Cross-set dedup — remove any train rows content-identical to test
        golden_test_keys = set(golden_test_df.apply(_dedup_key, axis=1))
        train_keys = train_df.apply(_dedup_key, axis=1)
        cross_mask = ~train_keys.isin(golden_test_keys)
        n_cross_dupes = (~cross_mask).sum()
        if n_cross_dupes:
            print(f"  Removed {n_cross_dupes} train issues that duplicate test content.")
        train_df = train_df[cross_mask].reset_index(drop=True)

        labels = sorted(raw_df['labels'].unique())
        print(f"\n  Golden test set (from deduplicated data):")
        for lab in labels:
            n_test = (golden_test_df['labels'] == lab).sum()
            n_train = (train_df['labels'] == lab).sum()
            print(f"    {lab}: test={n_test}, train={n_train}")
        print(f"  Total: test={len(golden_test_df)}, train={len(train_df)} (clean, deduplicated)")

    monitor.log(f"Data Split (Train: {len(train_df)}, Test: {len(golden_test_df)})")

    # ==================================================================
    # 2. LOAD MODEL
    # ==================================================================
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.cache_dir, "hub")

    print(f"Loading model: {args.model}")
    model_load_t0 = time.time()

    if not args.skip_training:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    else:
        print(f"  SKIPPING TRAINING. Loading adapters from: {args.adapter_dir}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter_dir,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

    model_load_time = time.time() - model_load_t0
    monitor.log(f"Model Loaded ({model_load_time:.1f}s)")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reset GPU peak tracking ONCE — before training — so we capture the
    # absolute peak across training + inference.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ==================================================================
    # 3. FORMAT TRAINING DATA (with strict truncation)
    # ==================================================================
    formatted_data = format_train_data(train_df, tokenizer, args.max_seq_length)
    train_dataset = Dataset.from_list(formatted_data)

    # ==================================================================
    # 4. TRAINING
    # ==================================================================
    training_time_s = 0.0

    if not args.skip_training:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            dataset_num_proc=2,
            args=TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=16,
                warmup_steps=5,
                num_train_epochs=3,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="paged_adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=os.path.join(args.output_dir, "checkpoints"),
                save_strategy="no",
                report_to="none",
            ),
        )

        print("Starting training...")
        train_start = time.time()
        trainer.train()
        training_time_s = time.time() - train_start
        monitor.record_phase("Training", train_start)
        monitor.log("Training Complete")

        if args.skip_save_adapter:
            print("Skipping adapter save (--skip_save_adapter)")
        else:
            print(f"Saving LoRA adapters to: {args.adapter_dir}")
            os.makedirs(args.adapter_dir, exist_ok=True)
            model.save_pretrained(args.adapter_dir)
            tokenizer.save_pretrained(args.adapter_dir)

    # Capture peak GPU after training (includes model load + training)
    gpu_peak_training_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    # ==================================================================
    # 5. INFERENCE
    # ==================================================================
    FastLanguageModel.for_inference(model)
    monitor.log("Switched to Inference Mode")

    # Compute prompt overhead for truncation during inference
    empty_prompt = PROMPT_TEMPLATE.format("", "")
    prompt_overhead = len(tokenizer.encode(empty_prompt, add_special_tokens=False))
    max_issue_tokens = args.max_seq_length - prompt_overhead - 30
    max_new_tokens = args.max_new_tokens
    inference_batch_size = args.inference_batch_size

    results = []
    all_prompt_tokens = []
    all_generated_tokens = []

    print(f"\nStarting inference on {len(golden_test_df)} test issues (batch_size={inference_batch_size})...")
    eval_start = time.time()

    # NOTE: Do NOT reset peak memory here — we want the absolute peak across
    # training + inference for fair resource comparison with RAGTAG.

    # --- Pre-build all prompts with truncation ---
    prepared = []
    for test_idx in range(len(golden_test_df)):
        row = golden_test_df.iloc[test_idx]
        issue_text = f"Title: {row['title']}\nBody: {row['body']}"

        truncated = False
        token_ids = tokenizer.encode(issue_text, add_special_tokens=False)
        if len(token_ids) > max_issue_tokens:
            token_ids = token_ids[:max_issue_tokens]
            issue_text = tokenizer.decode(token_ids) + "\n...[TRUNCATED]"
            truncated = True

        prompt = PROMPT_TEMPLATE.format(issue_text, "").rstrip()
        prepared.append({
            "test_idx": test_idx,
            "row": row,
            "prompt": prompt,
            "truncated": truncated,
        })

    # --- Process in batches ---
    total_batches = (len(prepared) + inference_batch_size - 1) // inference_batch_size

    for batch_start in tqdm(range(0, len(prepared), inference_batch_size),
                            desc="  Evaluating", unit="batch", total=total_batches):
        batch = prepared[batch_start : batch_start + inference_batch_size]
        prompts = [item["prompt"] for item in batch]

        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        tokenizer.padding_side = orig_side

        per_item_prompt_tokens = [
            (inputs.attention_mask[i] != 0).sum().item()
            for i in range(len(batch))
        ]

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    temperature=0.1,
                    top_p=0.9,
                    top_k=1,
                )

            for i, item in enumerate(batch):
                generated_ids = outputs[i][inputs.input_ids.shape[1]:]
                generated_ids = generated_ids[generated_ids != tokenizer.pad_token_id]
                generated_token_count = len(generated_ids)
                raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                all_prompt_tokens.append(per_item_prompt_tokens[i])
                all_generated_tokens.append(generated_token_count)

                predicted_label = parse_label(raw_output)
                row = item["row"]
                results.append({
                    "test_idx": item["test_idx"],
                    "title": row['title'],
                    "body": row['body'],
                    "ground_truth": row['labels'],
                    "predicted_label": predicted_label,
                    "raw_output": raw_output[:300],
                    "truncated": item["truncated"],
                    "neighbors_truncated": False,
                    "query_truncated": False,
                    "tokens_removed": 0,
                    "parsed_via": "regex" if predicted_label != "invalid" else "failed",
                    "prompt_tokens": per_item_prompt_tokens[i],
                    "generated_tokens": generated_token_count,
                })

        except Exception as e:
            for i, item in enumerate(batch):
                row = item["row"]
                all_prompt_tokens.append(per_item_prompt_tokens[i])
                all_generated_tokens.append(0)
                results.append({
                    "test_idx": item["test_idx"],
                    "title": row['title'],
                    "body": row['body'],
                    "ground_truth": row['labels'],
                    "predicted_label": "invalid",
                    "raw_output": f"ERROR: {e}"[:300],
                    "truncated": item["truncated"],
                    "neighbors_truncated": False,
                    "query_truncated": False,
                    "tokens_removed": 0,
                    "parsed_via": "failed",
                    "prompt_tokens": per_item_prompt_tokens[i],
                    "generated_tokens": 0,
                })

    elapsed = time.time() - eval_start
    monitor.record_phase("Evaluation", eval_start)
    monitor.log("Evaluation Complete")

    # ==================================================================
    # 6. OUTPUT — COMPATIBLE WITH evaluate.py
    # ==================================================================
    total = len(results)
    n_invalid = sum(1 for r in results if r["predicted_label"] == "invalid")

    print(f"  Total: {total} | Invalid: {n_invalid} ({100*n_invalid/total:.1f}%) | "
          f"Time: {elapsed:.1f}s ({total/elapsed:.1f} issues/s)")

    # Write standard prediction CSV
    preds_csv = os.path.join(args.output_dir, "preds_finetune_fixed.csv")
    cols = ["test_idx", "title", "body", "ground_truth", "predicted_label",
            "raw_output", "truncated", "neighbors_truncated", "query_truncated",
            "tokens_removed", "parsed_via", "prompt_tokens", "generated_tokens"]
    pd.DataFrame(results)[cols].to_csv(preds_csv, index=False)
    print(f"  Predictions: {preds_csv}")

    # Print classification report
    y_true = [r["ground_truth"] for r in results]
    y_pred = [r["predicted_label"] for r in results]

    print(f"\n{'='*60}")
    print(f"  CLASSIFICATION REPORT (all {total} issues)")
    print(f"{'='*60}")
    print(classification_report(
        y_true, y_pred,
        labels=["bug", "feature", "question"],
        target_names=["bug", "feature", "question"],
        zero_division=0,
    ))

    # Write cost metrics — EXACT same fields as llm_labeler.py
    gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    gpu_device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    gpu_total_memory_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 0) if torch.cuda.is_available() else 0

    cost_stats = {
        "model": args.model,
        "top_k": "N/A",
        "k_label": "finetune_fixed",
        "model_load_time_s": round(model_load_time, 2),
        "training_time_s": round(training_time_s, 2),
        "wall_time_s": round(elapsed, 2),
        "issues_per_second": round(total / elapsed, 2) if elapsed > 0 else 0.0,
        "total_issues": total,
        "total_prompt_tokens": sum(all_prompt_tokens),
        "total_generated_tokens": sum(all_generated_tokens),
        "avg_prompt_tokens": round(sum(all_prompt_tokens) / total, 1) if total else 0.0,
        "avg_generated_tokens": round(sum(all_generated_tokens) / total, 1) if total else 0.0,
        "min_prompt_tokens": min(all_prompt_tokens) if all_prompt_tokens else 0,
        "max_prompt_tokens": max(all_prompt_tokens) if all_prompt_tokens else 0,
        "gpu_peak_memory_mb": round(gpu_peak_mb, 1),
        "gpu_peak_memory_training_mb": round(gpu_peak_training_mb, 1),
        "gpu_device": gpu_device_name,
        "gpu_total_memory_mb": gpu_total_memory_mb,
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": max_new_tokens,
        "load_in_4bit": True,
    }
    cost_csv = os.path.join(args.output_dir, "cost_metrics.csv")
    pd.DataFrame([cost_stats]).to_csv(cost_csv, index=False)
    print(f"  Cost metrics: {cost_csv}")

    # Write performance log
    perf_csv = os.path.join(args.output_dir, "performance.csv")
    pd.DataFrame(monitor.performance_logs).to_csv(perf_csv, index=False)
    print(f"  Performance log: {perf_csv}")

    print(f"\nDone. Total wall time: {time.time() - monitor.start_time:.1f}s")


if __name__ == "__main__":
    main()

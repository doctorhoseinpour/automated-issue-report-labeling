#!/usr/bin/env python3
"""
baseline_finetune_flawed.py
===========================
Faithful CLI translation of the original fine-tune.ipynb notebook.

ALL ORIGINAL FLAWS ARE INTENTIONALLY PRESERVED IN TRAINING:
  - Training prompt includes chain-of-thought prefix; inference prompt does NOT
  - Inference prompt template is completely different from training prompt
  - Hardcoded EOS token ("<|endoftext|>") instead of tokenizer.eos_token
  - max_steps=60 (only 60 gradient updates regardless of dataset size)
  - No token truncation during training data formatting
  - Tokenizer truncation at (max_seq_length - 1) during inference
  - batch_decode on full output, split on "### Response:"
  - Invalid predictions are SKIPPED (inflates their self-reported metrics)
  - Retry logic (2 attempts) with silent skip on failure
  - adamw_8bit optimizer (not paged — higher VRAM usage)
  - Training data includes duplicates and potential test-content leakage

GOLDEN TEST SET: Derived from the deduplicated dataset using balanced top-N,
identical to fixed_fine-tune.py and build_and_query_index.py (RAGTAG).

Usage:
  python baseline_finetune_flawed.py \
    --model unsloth/Llama-3.2-3B-Instruct \
    --dataset issues11k.csv \
    --train_csv issues11k_train.csv \
    --test_csv issues11k_test.csv \
    --output_dir results/issues11k/agnostic/<tag>/finetune_flawed
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
# PROMPTS — PRESERVED EXACTLY FROM NOTEBOOK
# ============================================================================

# FLAW: Training prompt includes chain-of-thought instruction and prefix.
TRAIN_PROMPT_STYLE = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a GitHub issue classifier that will classify GitHub issues based on their title and description as 'bug', 'feature', or 'question'.
'bug', 'feature', or 'question' is the label. And the labels are the only possible response.
Please classify the following GitHub issue.

### Question:
{}

### Response:{}{}"""

CHAIN_OF_THOUGHT_PREFIX = "Analyzing the title and body to determine if it is a bug, feature request, or a general question: "

# FLAW: Inference prompt is completely different from training prompt.
INFERENCE_PROMPT_STYLE = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Please classify the following GitHub issue as 'bug', 'feature', or 'question'. Only return the label.

### Question:
{}

### Response:
"""

# FLAW: Hardcoded EOS token instead of tokenizer.eos_token
EOS_TOKEN = "<|endoftext|>"


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
# LABEL EXTRACTION — PRESERVED FROM NOTEBOOK
# ============================================================================

def extract_keyword(text):
    """FLAW PRESERVED: Simple regex — no canonicalization, no fallback."""
    match = re.search(r'\b(bug|feature|question)\b', text, re.IGNORECASE)
    if match:
        return match.group(0).lower()
    return None


# ============================================================================
# DEDUP + GOLDEN TEST SET (shared logic with fixed pipeline & RAGTAG)
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
        # group.index returns original (raw_df) index values
        test_indices.extend(group.index[:n_test].tolist())

    return sorted(test_indices)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Flawed Baseline Fine-Tune (faithful notebook translation)"
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Max tokens for inference output (default: 20)")
    parser.add_argument("--inference_batch_size", type=int, default=1,
                        help="Batch size for inference (default: 1)")
    parser.add_argument("--test_size", type=str, default="0.5",
                        help="Float (0,1) = fraction; int >= 1 = absolute count")
    parser.add_argument("--train_csv", type=str, default=None,
                        help="Pre-computed train split CSV (skips internal splitting)")
    parser.add_argument("--test_csv", type=str, default=None,
                        help="Pre-computed test split CSV (skips internal splitting)")
    parser.add_argument("--output_dir", type=str, default="results/flawed_baseline",
                        help="Directory for all outputs")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, load saved adapters")
    parser.add_argument("--adapter_dir", type=str, default=None,
                        help="Directory to save/load LoRA adapters")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace model cache directory")
    args = parser.parse_args()

    test_size = parse_test_size(args.test_size)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.adapter_dir is None:
        safe_model = args.model.replace("/", "_")
        args.adapter_dir = os.path.join(args.output_dir, f"adapters_{safe_model}")

    monitor = SystemMonitor()
    monitor.log("Initialization")

    # ==================================================================
    # 1. DERIVE GOLDEN TEST SET, THEN BUILD FLAWED TRAIN SET
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

        # Step C: FLAWED train set = raw data minus golden test rows
        # FLAW PRESERVED: No deduplication of train data. Content-identical copies
        # of test issues that weren't selected as test rows remain in training.
        train_df = raw_df.drop(index=golden_test_indices).reset_index(drop=True)

        labels = sorted(raw_df['labels'].unique())
        print(f"\n  Golden test set (from deduplicated data):")
        for lab in labels:
            n_test = (golden_test_df['labels'] == lab).sum()
            n_train = (train_df['labels'] == lab).sum()
            print(f"    {lab}: test={n_test}, train={n_train}")
        print(f"  Total: test={len(golden_test_df)}, train={len(train_df)} (flawed, with duplicates)")

    monitor.log(f"Data Split (Train: {len(train_df)}, Test: {len(golden_test_df)})")

    # ==================================================================
    # 2. FORMAT TRAINING DATA
    # ==================================================================
    # FLAW PRESERVED: No token truncation during training.
    # FLAW PRESERVED: Chain-of-thought prefix in training response.
    # FLAW PRESERVED: Hardcoded EOS_TOKEN.
    print("Formatting training data (no truncation — notebook behavior)...")
    formatted_data = []
    for _, row in train_df.iterrows():
        issue_text = f"Title: {row['title']}\nBody: {row['body']}"
        formatted_text = TRAIN_PROMPT_STYLE.format(
            issue_text, CHAIN_OF_THOUGHT_PREFIX, row['labels']
        ) + EOS_TOKEN
        formatted_data.append({"text": formatted_text})

    train_dataset = Dataset.from_list(formatted_data)
    print(f"  Formatted {len(formatted_data)} training examples")

    # ==================================================================
    # 3. LOAD MODEL
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

        # PRESERVED: Exact LoRA config from notebook
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
            use_rslora=False,
            loftq_config=None,
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
    # 4. TRAINING
    # ==================================================================
    training_time_s = 0.0

    if not args.skip_training:
        # FLAW PRESERVED: max_steps=60, batch_size=2, adamw_8bit
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            dataset_num_proc=2,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=60,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=os.path.join(args.output_dir, "checkpoints"),
                report_to="none",
            ),
        )

        print("Starting training...")
        train_start = time.time()
        trainer.train()
        training_time_s = time.time() - train_start
        monitor.record_phase("Training", train_start)
        monitor.log("Training Complete")

        print(f"Saving LoRA adapters to: {args.adapter_dir}")
        os.makedirs(args.adapter_dir, exist_ok=True)
        model.save_pretrained(args.adapter_dir)
        tokenizer.save_pretrained(args.adapter_dir)

    # Capture peak GPU after training (includes model load + training)
    gpu_peak_training_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    # ==================================================================
    # 5. INFERENCE — PRESERVED FLAWS
    # ==================================================================
    FastLanguageModel.for_inference(model)
    monitor.log("Switched to Inference Mode")

    # FLAW PRESERVED: max_token_length = max_seq_length - 1
    max_token_length = args.max_seq_length - 1
    max_tries = 2  # FLAW PRESERVED: comment says "5 attempts" but code uses 2
    max_new_tokens = args.max_new_tokens
    inference_batch_size = args.inference_batch_size

    results = []
    # These track the FLAWED metrics (skipping invalids)
    y_true_flawed = []
    y_pred_flawed = []

    all_prompt_tokens = []
    all_generated_tokens = []

    print(f"\nStarting inference on {len(golden_test_df)} test issues (batch_size={inference_batch_size})...")
    eval_start = time.time()

    # NOTE: Do NOT reset peak memory here — we want the absolute peak across
    # training + inference for fair resource comparison with RAGTAG.

    # --- Pre-build all prompts ---
    all_prompts = []
    for test_idx in range(len(golden_test_df)):
        row = golden_test_df.iloc[test_idx]
        description = f"{row['title']} \n {row['body']}"
        # FLAW PRESERVED: Wraps text with extra instruction inside the prompt
        prompt_input = (
            f"Classify, IN ONLY 1 WORD, the following GitHub issue as "
            f"'feature', 'bug', or 'question' based on its title and body:\n"
            f"{description}"
        )
        prompt = INFERENCE_PROMPT_STYLE.format(prompt_input)
        all_prompts.append(prompt)

    # --- Process in batches ---
    total_batches = (len(golden_test_df) + inference_batch_size - 1) // inference_batch_size

    for batch_start in tqdm(range(0, len(golden_test_df), inference_batch_size),
                            desc="  Evaluating", unit="batch", total=total_batches):
        batch_end = min(batch_start + inference_batch_size, len(golden_test_df))
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))

        # FLAW PRESERVED: Tokenizer truncation at max_token_length
        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_token_length,
        ).to("cuda")
        tokenizer.padding_side = orig_side

        per_item_prompt_tokens = [
            (inputs.attention_mask[i] != 0).sum().item()
            for i in range(len(batch_indices))
        ]

        # FLAW PRESERVED: Retry logic (2 attempts), batch_decode on full output
        attempt = 0
        batch_outputs = None

        while attempt < max_tries:
            try:
                batch_outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    temperature=0.1,
                    top_p=0,   # FLAW PRESERVED: top_p=0
                    top_k=1,
                )
                break
            except Exception as e:
                print(f"  Error during generation attempt {attempt+1}: {e}")
            attempt += 1

        # Process each item in batch
        for i, test_idx in enumerate(batch_indices):
            row = golden_test_df.iloc[test_idx]
            correct_label = row['labels']
            prompt_token_count = per_item_prompt_tokens[i]

            if batch_outputs is None:
                # All retries failed
                results.append({
                    "test_idx": test_idx, "title": row['title'], "body": row['body'],
                    "ground_truth": correct_label, "predicted_label": "invalid",
                    "raw_output": "NO_RESPONSE_AFTER_RETRIES",
                    "truncated": False, "neighbors_truncated": False, "query_truncated": False,
                    "tokens_removed": 0, "parsed_via": "failed",
                    "prompt_tokens": prompt_token_count, "generated_tokens": 0,
                })
                all_prompt_tokens.append(prompt_token_count)
                all_generated_tokens.append(0)
                continue

            generated_token_count = batch_outputs.shape[1] - inputs.input_ids.shape[1]

            # FLAW PRESERVED: batch_decode on full output (includes prompt)
            response = tokenizer.batch_decode(batch_outputs[i:i+1])
            raw_output = ""

            if (response and len(response) > 0
                    and "### Response:" in response[0]):
                formatted_answer = response[0].split("### Response:")[-1]
                raw_output = formatted_answer.strip()
            else:
                # FLAW PRESERVED: Skip if no valid response
                results.append({
                    "test_idx": test_idx, "title": row['title'], "body": row['body'],
                    "ground_truth": correct_label, "predicted_label": "invalid",
                    "raw_output": "NO_RESPONSE_MARKER",
                    "truncated": False, "neighbors_truncated": False, "query_truncated": False,
                    "tokens_removed": 0, "parsed_via": "failed",
                    "prompt_tokens": prompt_token_count, "generated_tokens": generated_token_count,
                })
                all_prompt_tokens.append(prompt_token_count)
                all_generated_tokens.append(generated_token_count)
                continue

            predicted_label = extract_keyword(raw_output)

            # FLAW PRESERVED: Skip if keyword extraction fails
            if predicted_label is None:
                results.append({
                    "test_idx": test_idx, "title": row['title'], "body": row['body'],
                    "ground_truth": correct_label, "predicted_label": "invalid",
                    "raw_output": raw_output[:300],
                    "truncated": False, "neighbors_truncated": False, "query_truncated": False,
                    "tokens_removed": 0, "parsed_via": "failed",
                    "prompt_tokens": prompt_token_count, "generated_tokens": generated_token_count,
                })
                all_prompt_tokens.append(prompt_token_count)
                all_generated_tokens.append(generated_token_count)
                continue  # FLAW: skipped in their metrics

            # Valid prediction
            y_true_flawed.append(correct_label)
            y_pred_flawed.append(predicted_label)

            results.append({
                "test_idx": test_idx, "title": row['title'], "body": row['body'],
                "ground_truth": correct_label, "predicted_label": predicted_label,
                "raw_output": raw_output[:300],
                "truncated": False, "neighbors_truncated": False, "query_truncated": False,
                "tokens_removed": 0, "parsed_via": "regex",
                "prompt_tokens": prompt_token_count, "generated_tokens": generated_token_count,
            })
            all_prompt_tokens.append(prompt_token_count)
            all_generated_tokens.append(generated_token_count)

    elapsed = time.time() - eval_start
    monitor.record_phase("Evaluation", eval_start)
    monitor.log("Evaluation Complete")

    # ==================================================================
    # 6. OUTPUT — COMPATIBLE WITH evaluate.py
    # ==================================================================
    total = len(results)
    n_invalid = sum(1 for r in results if r["predicted_label"] == "invalid")

    print(f"  Total: {total} | Invalid/Skipped: {n_invalid} | "
          f"Time: {elapsed:.1f}s ({total/elapsed:.1f} issues/s)")

    # Write standard prediction CSV (all issues, including invalids)
    preds_csv = os.path.join(args.output_dir, "preds_finetune_flawed.csv")
    cols = ["test_idx", "title", "body", "ground_truth", "predicted_label",
            "raw_output", "truncated", "neighbors_truncated", "query_truncated",
            "tokens_removed", "parsed_via", "prompt_tokens", "generated_tokens"]
    pd.DataFrame(results)[cols].to_csv(preds_csv, index=False)
    print(f"  Predictions (all issues): {preds_csv}")

    # Print their FLAWED classification report (excludes invalid predictions)
    print(f"\n{'='*60}")
    print(f"  FLAWED METRICS (excludes {n_invalid} invalid predictions)")
    print(f"{'='*60}")
    if y_true_flawed:
        print(classification_report(
            y_true_flawed, y_pred_flawed,
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
        "k_label": "finetune_flawed",
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

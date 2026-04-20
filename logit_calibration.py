#!/usr/bin/env python3
"""
logit_calibration.py
====================
Logit-level calibration for RAGTAG: Batch Calibration (BC) and Contrastive
Decoding (CD).

Instead of model.generate(), uses a single forward pass to extract logits at
the label token position (right after the "<label>" prefill). This gives us
the model's raw probability distribution over {bug, feature, question}, which
we then calibrate to correct the parametric bug bias.

Methods:
  - bc:    Batch Calibration — subtract marginal class distribution from each
           prediction's softmax probabilities (Zhou et al., 2024)
  - cd:    Contrastive Decoding — subtract zero-shot logits from RAG logits
           to cancel the parametric prior
  - bc_cd: CD first, then BC on the result

Usage:
  python logit_calibration.py --model unsloth/Llama-3.2-3B-Instruct \
    --neighbors_dir results/issues3k_debias/neighbors \
    --top_ks "3" --method bc --output_dir results/issues3k_logit_bc/llama3b/ragtag/predictions

  python logit_calibration.py --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
    --neighbors_dir results/issues30k/neighbors \
    --top_ks "9" --method cd --cd_alpha 1.0 \
    --output_dir results/issues30k_logit_cd_a1.0/llama8b/ragtag/predictions
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from llm_labeler import (
    VALID_LABELS,
    build_chat_messages,
    load_test_issues,
    TestIssue,
    _debias_neighbors,
    print_gpu_stats,
)

# ---------------------------------------------------------------------------
# Token ID discovery
# ---------------------------------------------------------------------------

def get_label_token_ids(tokenizer) -> dict:
    """Resolve token IDs for 'bug', 'feature', 'question' after '<label>' prefix.

    Since the prompt ends with '<label>' and the model generates the label word
    immediately (no leading space), we encode in context to handle tokenizer
    boundary merging.

    Returns:
        dict: {"bug": token_id, "feature": token_id, "question": token_id}

    Raises:
        ValueError: if any label tokenizes to more than one token.
    """
    context = "<label>"
    context_ids = tokenizer.encode(context, add_special_tokens=False)

    label_ids = {}
    for label in VALID_LABELS:
        full_ids = tokenizer.encode(context + label, add_special_tokens=False)
        label_part = full_ids[len(context_ids):]

        if len(label_part) == 1:
            label_ids[label] = label_part[0]
        elif len(label_part) == 0:
            # Tokenizer merged the label into the context — try raw encode
            raw_ids = tokenizer.encode(label, add_special_tokens=False)
            if len(raw_ids) == 1:
                label_ids[label] = raw_ids[0]
            else:
                raise ValueError(
                    f"Label '{label}' produces 0 tokens after '<label>' prefix "
                    f"and {len(raw_ids)} tokens raw. Cannot determine token ID."
                )
        else:
            raise ValueError(
                f"Label '{label}' tokenizes to {len(label_part)} tokens "
                f"after '<label>' prefix: {[tokenizer.decode([t]) for t in label_part]}. "
                f"Logit calibration requires single-token labels."
            )

    # Verify round-trip
    for label, tid in label_ids.items():
        decoded = tokenizer.decode([tid]).strip().lower()
        if decoded != label:
            raise ValueError(
                f"Token ID {tid} decodes to '{decoded}', expected '{label}'"
            )

    return label_ids


# ---------------------------------------------------------------------------
# Core logit extraction
# ---------------------------------------------------------------------------

def extract_logits_batch(
    model,
    tokenizer,
    prompts: List[str],
    label_ids_tensor: torch.Tensor,
    max_seq_length: int = 8192,
) -> torch.Tensor:
    """Run a batched forward pass and extract logits for the 3 label tokens.

    Args:
        prompts: list of N prompt strings (already include "<label>" suffix)
        label_ids_tensor: tensor of [bug_id, feature_id, question_id] on cuda
        max_seq_length: maximum sequence length for the model

    Returns:
        logits: shape [N, 3] — raw logits for bug/feature/question per item
    """
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=False
    ).to("cuda")
    tokenizer.padding_side = orig_side

    seq_len = inputs.input_ids.shape[1]
    if seq_len > max_seq_length:
        # This shouldn't happen — build_chat_messages should have truncated
        # the prompt content to fit. Warn rather than crash.
        print(f"    WARNING: batch seq_len {seq_len} > max_seq_length {max_seq_length}, "
              f"truncating from left (losing early prompt content)")
        inputs.input_ids = inputs.input_ids[:, -max_seq_length:]
        inputs.attention_mask = inputs.attention_mask[:, -max_seq_length:]

    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )

    # With left-padding, the last token is always at index -1 for all items
    last_logits = outputs.logits[:, -1, :]       # [batch_size, vocab_size]
    label_logits = last_logits[:, label_ids_tensor]  # [batch_size, 3]

    # Track prompt token counts (real tokens, excluding padding)
    prompt_tokens = [
        (inputs.attention_mask[i] != 0).sum().item()
        for i in range(len(prompts))
    ]

    return label_logits.float(), prompt_tokens


# ---------------------------------------------------------------------------
# Calibration methods
# ---------------------------------------------------------------------------

def apply_batch_calibration(all_logits: torch.Tensor) -> torch.Tensor:
    """Batch Calibration (Zhou et al., 2024).

    p_BC(c|x_i) = p(c|x_i) - p_bar(c) + 1/C

    Calibrated scores always sum to 1.0 per row.
    """
    C = all_logits.shape[1]
    probs = F.softmax(all_logits, dim=-1)
    marginal = probs.mean(dim=0, keepdim=True)
    calibrated = probs - marginal + (1.0 / C)
    return calibrated


def apply_contrastive_decoding(
    rag_logits: torch.Tensor,
    zs_logits: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Contrastive Decoding: subtract zero-shot logits from RAG logits."""
    return rag_logits - alpha * zs_logits


def apply_bc_cd(
    rag_logits: torch.Tensor,
    zs_logits: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """CD first (on raw logits), then BC on the CD output."""
    cd_logits = apply_contrastive_decoding(rag_logits, zs_logits, alpha)
    return apply_batch_calibration(cd_logits)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

@dataclass
class PreparedItem:
    issue: TestIssue
    rag_prompt: str
    zs_prompt: Optional[str]
    trunc_truncated: bool
    trunc_neighbors_truncated: bool
    trunc_query_truncated: bool
    trunc_tokens_removed: int


def build_prompts(
    test_issues: List[TestIssue],
    k: int,
    tokenizer,
    max_prompt_tokens: int,
    method: str,
    debias_retrieval: bool = False,
    debias_margin: int = 3,
) -> List[PreparedItem]:
    """Build RAG prompts (and optionally zero-shot prompts for CD)."""
    prepared = []
    n_debiased = 0

    for issue in test_issues:
        neighbors = issue.neighbors[:k]

        if debias_retrieval and neighbors:
            original_len = len(neighbors)
            neighbors = _debias_neighbors(neighbors, debias_margin)
            if len(neighbors) < original_len:
                n_debiased += 1

        # RAG prompt
        messages_rag, trunc = build_chat_messages(
            test_title=issue.title,
            test_body=issue.body,
            neighbors=neighbors,
            k=k,
            is_thinking_model=False,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
        )
        rag_prompt = tokenizer.apply_chat_template(
            messages_rag, tokenize=False, add_generation_prompt=True,
        ) + "<label>"

        # Zero-shot prompt (only for CD methods)
        zs_prompt = None
        if method in ("cd", "bc_cd"):
            messages_zs, _ = build_chat_messages(
                test_title=issue.title,
                test_body=issue.body,
                neighbors=[],
                k=0,
                is_thinking_model=False,
                max_prompt_tokens=max_prompt_tokens,
                tokenizer=tokenizer,
            )
            zs_prompt = tokenizer.apply_chat_template(
                messages_zs, tokenize=False, add_generation_prompt=True,
            ) + "<label>"

        prepared.append(PreparedItem(
            issue=issue,
            rag_prompt=rag_prompt,
            zs_prompt=zs_prompt,
            trunc_truncated=trunc.truncated,
            trunc_neighbors_truncated=trunc.neighbors_truncated,
            trunc_query_truncated=trunc.query_truncated,
            trunc_tokens_removed=trunc.tokens_removed,
        ))

    if debias_retrieval:
        print(f"    Debiased: {n_debiased} ({100*n_debiased/len(test_issues):.1f}%) — bug neighbors removed")

    return prepared


# ---------------------------------------------------------------------------
# Main inference loop for one k
# ---------------------------------------------------------------------------

def run_calibration(
    test_issues: List[TestIssue],
    k: int,
    model,
    tokenizer,
    method: str,
    cd_alpha: float,
    label_ids_tensor: torch.Tensor,
    max_prompt_tokens: int,
    max_seq_length: int,
    output_csv: str,
    inference_batch_size: int = 4,
    debias_retrieval: bool = False,
    debias_margin: int = 3,
    save_logits: bool = False,
) -> float:
    """Run logit-calibrated inference for one k value."""

    t0 = time.time()
    print(f"\n  [k={k}, {method}] Building prompts for {len(test_issues)} issues...")

    prepared = build_prompts(
        test_issues, k, tokenizer, max_prompt_tokens,
        method, debias_retrieval, debias_margin,
    )

    # --- Collect RAG logits ---
    all_rag_logits = []
    all_rag_prompt_tokens = []

    total_batches = (len(prepared) + inference_batch_size - 1) // inference_batch_size
    for batch_start in tqdm(range(0, len(prepared), inference_batch_size),
                            desc=f"  k={k} RAG fwd", unit="batch", total=total_batches):
        batch = prepared[batch_start:batch_start + inference_batch_size]
        prompts = [item.rag_prompt for item in batch]
        logits, ptoks = extract_logits_batch(model, tokenizer, prompts, label_ids_tensor, max_seq_length)
        all_rag_logits.append(logits.cpu())
        all_rag_prompt_tokens.extend(ptoks)

    all_rag_logits = torch.cat(all_rag_logits, dim=0)  # [N, 3]

    # --- Collect ZS logits (only for CD methods) ---
    all_zs_logits = None
    all_zs_prompt_tokens = []

    if method in ("cd", "bc_cd"):
        zs_logit_parts = []
        for batch_start in tqdm(range(0, len(prepared), inference_batch_size),
                                desc=f"  k={k} ZS fwd", unit="batch", total=total_batches):
            batch = prepared[batch_start:batch_start + inference_batch_size]
            prompts = [item.zs_prompt for item in batch]
            logits, ptoks = extract_logits_batch(model, tokenizer, prompts, label_ids_tensor, max_seq_length)
            zs_logit_parts.append(logits.cpu())
            all_zs_prompt_tokens.extend(ptoks)
        all_zs_logits = torch.cat(zs_logit_parts, dim=0)  # [N, 3]

    # --- Apply calibration ---
    if method == "bc":
        scores = apply_batch_calibration(all_rag_logits)
    elif method == "cd":
        scores = apply_contrastive_decoding(all_rag_logits, all_zs_logits, cd_alpha)
    elif method == "bc_cd":
        scores = apply_bc_cd(all_rag_logits, all_zs_logits, cd_alpha)
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- Predictions ---
    pred_indices = scores.argmax(dim=-1)
    pred_labels = [VALID_LABELS[idx] for idx in pred_indices.tolist()]

    uncal_pred_indices = all_rag_logits.argmax(dim=-1)
    uncal_labels = [VALID_LABELS[idx] for idx in uncal_pred_indices.tolist()]

    # --- Build result rows ---
    rag_probs = F.softmax(all_rag_logits, dim=-1)
    results = []

    for i, item in enumerate(prepared):
        row = {
            "test_idx": item.issue.idx,
            "title": item.issue.title,
            "body": item.issue.body,
            "ground_truth": item.issue.label,
            "predicted_label": pred_labels[i],
            "uncalibrated_label": uncal_labels[i],
            "method": method,
            "truncated": item.trunc_truncated,
            "neighbors_truncated": item.trunc_neighbors_truncated,
            "query_truncated": item.trunc_query_truncated,
            "tokens_removed": item.trunc_tokens_removed,
            "prompt_tokens": all_rag_prompt_tokens[i],
        }

        for j, label in enumerate(VALID_LABELS):
            row[f"raw_prob_{label}"] = round(rag_probs[i, j].item(), 6)
            row[f"score_{label}"] = round(scores[i, j].item(), 6)

        if save_logits:
            for j, label in enumerate(VALID_LABELS):
                row[f"raw_logit_{label}"] = round(all_rag_logits[i, j].item(), 4)
                if all_zs_logits is not None:
                    row[f"zs_logit_{label}"] = round(all_zs_logits[i, j].item(), 4)

        results.append(row)

    elapsed = time.time() - t0

    # --- Write CSV ---
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    # --- Print diagnostics ---
    _print_calibration_summary(df, method, k, elapsed, cd_alpha,
                               all_rag_logits, all_zs_logits, scores)

    return elapsed


def _print_calibration_summary(df, method, k, elapsed, cd_alpha,
                               rag_logits, zs_logits, scores):
    """Print diagnostic info about the calibration run."""
    N = len(df)
    gt_dist = df["ground_truth"].str.lower().str.strip().value_counts().to_dict()
    uncal_dist = df["uncalibrated_label"].value_counts().to_dict()
    cal_dist = df["predicted_label"].value_counts().to_dict()

    print(f"\n  [k={k}, {method}] {N} issues in {elapsed:.1f}s ({N/elapsed:.1f}/s)")
    print(f"    Ground truth:  { {l: gt_dist.get(l, 0) for l in VALID_LABELS} }")
    print(f"    Uncalibrated:  { {l: uncal_dist.get(l, 0) for l in VALID_LABELS} }")
    print(f"    Calibrated:    { {l: cal_dist.get(l, 0) for l in VALID_LABELS} }")

    if method in ("bc", "bc_cd"):
        rag_probs = F.softmax(rag_logits, dim=-1)
        marginal = rag_probs.mean(dim=0)
        print(f"    BC marginal:   " +
              ", ".join(f"{l}={marginal[i]:.4f}" for i, l in enumerate(VALID_LABELS)))

    if method in ("cd", "bc_cd") and zs_logits is not None:
        diff = rag_logits.mean(dim=0) - zs_logits.mean(dim=0)
        print(f"    CD avg delta (RAG-ZS): " +
              ", ".join(f"{l}={diff[i]:.3f}" for i, l in enumerate(VALID_LABELS)))
        print(f"    CD alpha:      {cd_alpha}")

    flipped = (df["predicted_label"] != df["uncalibrated_label"]).sum()
    print(f"    Predictions changed: {flipped} ({100*flipped/N:.1f}%)")

    # Quick accuracy check
    correct_uncal = (df["uncalibrated_label"] == df["ground_truth"].str.lower().str.strip()).sum()
    correct_cal = (df["predicted_label"] == df["ground_truth"].str.lower().str.strip()).sum()
    print(f"    Accuracy: uncalibrated={correct_uncal}/{N} ({100*correct_uncal/N:.1f}%) "
          f"→ calibrated={correct_cal}/{N} ({100*correct_cal/N:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Logit-level calibration: Batch Calibration (BC), "
                    "Contrastive Decoding (CD), or both (BC+CD)"
    )
    # Model
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--cache_dir", default=None,
                        help="HuggingFace model cache directory")

    # Data
    parser.add_argument("--neighbors_dir", required=True,
                        help="Directory containing neighbors_k{K}.csv files")
    parser.add_argument("--top_ks", required=True,
                        help="Comma-separated K values (k=0 not allowed)")

    # Method
    parser.add_argument("--method", required=True, choices=["bc", "cd", "bc_cd"],
                        help="Calibration method")
    parser.add_argument("--cd_alpha", type=float, default=1.0,
                        help="Contrastive decoding weight (default: 1.0)")

    # Phase 1 combination
    parser.add_argument("--debias_retrieval", action="store_true",
                        help="Combine with debiased retrieval (Phase 1)")
    parser.add_argument("--debias_margin", type=int, default=3,
                        help="Margin for debias trigger (default: 3)")

    # Output
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for prediction CSVs")
    parser.add_argument("--eval_dir", default=None,
                        help="Directory for evaluation CSVs")
    parser.add_argument("--model_name_for_eval", default=None,
                        help="Model name label for evaluation output")
    parser.add_argument("--save_logits", action="store_true",
                        help="Save raw and calibrated logits to CSV")

    # Batching
    parser.add_argument("--inference_batch_size", type=int, default=1,
                        help="Forward pass batch size (default: 1)")

    args = parser.parse_args()

    if args.no_4bit:
        args.load_in_4bit = False

    ks = [int(x) for x in args.top_ks.split(",")]
    if 0 in ks:
        sys.exit("ERROR: k=0 (zero-shot) is not meaningful for logit calibration. "
                 "BC needs RAG neighbors; CD subtracts zero-shot from RAG.")

    print(f"{'='*60}")
    print(f"  Logit Calibration ({args.method.upper()})")
    print(f"{'='*60}")
    print(f"  Model:           {args.model}")
    print(f"  K values:        {ks}")
    print(f"  Method:          {args.method}")
    if args.method in ("cd", "bc_cd"):
        print(f"  CD alpha:        {args.cd_alpha}")
    print(f"  max_seq_length:  {args.max_seq_length}")
    print(f"  batch_size:      {args.inference_batch_size}")
    print(f"  load_in_4bit:    {args.load_in_4bit}")
    if args.debias_retrieval:
        print(f"  debias_retrieval: margin={args.debias_margin}")
    print(f"{'='*60}")

    # --- Load model ---
    print(f"\nLoading model: {args.model}")
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.cache_dir, "hub")
        os.makedirs(args.cache_dir, exist_ok=True)

    model_load_t0 = time.time()
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    model_load_time = time.time() - model_load_t0

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print_gpu_stats("model loaded")
    print(f"  Model load time: {model_load_time:.1f}s")

    # --- Verify token IDs ---
    label_token_ids = get_label_token_ids(tokenizer)
    print(f"  Label token IDs: {label_token_ids}")
    label_ids_tensor = torch.tensor(
        [label_token_ids[l] for l in VALID_LABELS], device="cuda"
    )

    # --- Load test issues ---
    max_k = max(ks)
    max_k_file = os.path.join(args.neighbors_dir, f"neighbors_k{max_k}.csv")
    if not os.path.exists(max_k_file):
        # Fall back to largest available
        import glob
        nb_files = glob.glob(os.path.join(args.neighbors_dir, "neighbors_k*.csv"))
        if not nb_files:
            sys.exit(f"ERROR: No neighbor files found in {args.neighbors_dir}")

        def extract_k(p):
            try:
                return int(os.path.basename(p).replace("neighbors_k", "").replace(".csv", ""))
            except ValueError:
                return 0
        nb_files.sort(key=extract_k, reverse=True)
        max_k_file = nb_files[0]
        max_k = extract_k(max_k_file)
        print(f"  Using largest neighbor file: {max_k_file} (k={max_k})")

    print(f"\nLoading test issues from {max_k_file} (max_k={max_k})...")
    test_issues = load_test_issues(max_k_file, max_k)
    print(f"  Loaded {len(test_issues)} test issues with up to {max_k} neighbors each")

    # Conservative budget: reserve 100 tokens for chat template overhead
    # (role markers, BOS/EOS, formatting). build_chat_messages does smart
    # truncation (proportional neighbor body compression) within this budget,
    # so every prompt will fit within max_seq_length after tokenization.
    max_prompt_tokens = args.max_seq_length - 100
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Run each k ---
    total_time = 0
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, "evaluate.py")

    for k in ks:
        k_label = f"k{k}"
        method_tag = args.method
        if args.method in ("cd", "bc_cd"):
            method_tag += f"_a{args.cd_alpha}"

        output_csv = os.path.join(args.output_dir, f"preds_{k_label}_{method_tag}.csv")

        if os.path.exists(output_csv):
            print(f"\n  [{k_label}] Already exists, SKIPPING: {output_csv}")
            continue

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        elapsed = run_calibration(
            test_issues=test_issues,
            k=k,
            model=model,
            tokenizer=tokenizer,
            method=args.method,
            cd_alpha=args.cd_alpha,
            label_ids_tensor=label_ids_tensor,
            max_prompt_tokens=max_prompt_tokens,
            max_seq_length=args.max_seq_length,
            output_csv=output_csv,
            inference_batch_size=args.inference_batch_size,
            debias_retrieval=args.debias_retrieval,
            debias_margin=args.debias_margin,
            save_logits=args.save_logits,
        )
        total_time += elapsed

        # Evaluate
        if args.eval_dir and os.path.exists(eval_script) and os.path.exists(output_csv):
            eval_csv = os.path.join(args.eval_dir, f"eval_{k_label}_{method_tag}.csv")
            eval_model_name = args.model_name_for_eval or args.model
            print(f"  [{k_label}] Evaluating...")
            subprocess.run([
                sys.executable, eval_script,
                "--preds_csv", output_csv,
                "--top_k", str(k),
                "--output_csv", eval_csv,
                "--model_name", eval_model_name,
            ], check=False)

    print_gpu_stats("all done")
    print(f"\nAll K values complete. Total inference time: {total_time:.1f}s")
    print(f"  Model load time: {model_load_time:.1f}s")
    if torch.cuda.is_available():
        print(f"  GPU peak memory: {torch.cuda.max_memory_allocated()/(1024**3):.2f} GB")


if __name__ == "__main__":
    main()

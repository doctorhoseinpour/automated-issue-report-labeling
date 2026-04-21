#!/usr/bin/env python3
"""
activation_steering.py
======================
Apply activation steering during RAGTAG inference to correct the model's bug bias.

Supports two steering methods:
  1. caa (Contrastive Activation Addition): add steering vector to residual stream
  2. ablation (Directional Ablation / NTW): project out bias direction from residual stream

Usage:
  # Single layer CAA steering
  python activation_steering.py \
    --model unsloth/Llama-3.2-3B-Instruct \
    --neighbors_csv results/issues3k_debias/neighbors/neighbors_k3.csv \
    --steering_vectors results/steering_vectors/llama3b_3k/steering_vectors.pt \
    --output_dir results/issues3k_steering/ \
    --method caa --layer 9 --multiplier -1.0 --top_k 3

  # Layer sweep
  python activation_steering.py \
    --neighbors_csv results/issues3k_debias/neighbors/neighbors_k3.csv \
    --steering_vectors results/steering_vectors/llama3b_3k/steering_vectors.pt \
    --output_dir results/issues3k_steering/layer_sweep/ \
    --method caa --layer sweep --multiplier -1.0 --top_k 3

  # Directional ablation (NTW)
  python activation_steering.py \
    --neighbors_csv results/issues3k_debias/neighbors/neighbors_k3.csv \
    --steering_vectors results/steering_vectors/llama3b_3k/steering_vectors.pt \
    --output_dir results/issues3k_steering/ \
    --method ablation --layer 9 --top_k 3

  # Full ablation (all layers)
  python activation_steering.py \
    --steering_vectors results/steering_vectors/llama3b_3k/steering_vectors.pt \
    --output_dir results/issues3k_steering/ \
    --method ablation --layer all --top_k 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

# Import shared infrastructure from llm_labeler
from llm_labeler import (
    SYSTEM_PROMPT,
    VALID_LABELS,
    build_chat_messages,
    load_test_issues,
    parse_label,
)

import logging
import warnings

logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*", category=UserWarning)


# ---------------------------------------------------------------------------
# Steering hook factories
# ---------------------------------------------------------------------------

def make_caa_hook(steering_vector: torch.Tensor, multiplier: float):
    """Create a forward hook that adds a scaled steering vector to hidden states.

    CAA (Contrastive Activation Addition): translates activations in the
    residual stream to shift the model's output distribution.

    Args:
        steering_vector: shape (hidden_dim,)
        multiplier: scaling factor (negative = subtract bug direction)
    """
    vec = steering_vector.clone()

    def hook_fn(module, input, output):
        hidden_states = output[0]  # (batch, seq, hidden_dim)
        device = hidden_states.device
        dtype = hidden_states.dtype
        # Add steering vector to all token positions
        hidden_states = hidden_states + multiplier * vec.to(device=device, dtype=dtype)
        return (hidden_states,) + output[1:]

    return hook_fn


def make_ablation_hook(steering_vector: torch.Tensor):
    """Create a forward hook that performs directional ablation.

    NTW (No Training Wheels): projects out the bias direction from the
    residual stream, preventing the model from representing this direction.

    x' = x - (x . r_hat) * r_hat

    Args:
        steering_vector: shape (hidden_dim,) — will be unit-normalized
    """
    direction = steering_vector.clone().float()
    direction = direction / direction.norm()

    def hook_fn(module, input, output):
        hidden_states = output[0]  # (batch, seq, hidden_dim)
        device = hidden_states.device
        dtype = hidden_states.dtype
        d = direction.to(device=device, dtype=dtype)
        # Project out the bias direction from all positions
        # (batch, seq, hidden) @ (hidden,) -> (batch, seq)
        proj_coeff = (hidden_states * d).sum(dim=-1, keepdim=True)
        hidden_states = hidden_states - proj_coeff * d.unsqueeze(0).unsqueeze(0)
        return (hidden_states,) + output[1:]

    return hook_fn


# ---------------------------------------------------------------------------
# Hook management
# ---------------------------------------------------------------------------

def register_steering_hooks(
    model,
    steering_vectors: Dict[int, torch.Tensor],
    method: str,
    layer: int | str,
    multiplier: float = -1.0,
) -> list:
    """Register forward hooks on model layers.

    Args:
        model: The loaded LLM
        steering_vectors: dict mapping layer_idx -> vector (hidden_dim,)
        method: "caa" or "ablation"
        layer: integer layer index, "all" for all layers, or handled externally for "sweep"
        multiplier: scaling factor for CAA method

    Returns:
        List of hook handles (call .remove() to deregister)
    """
    handles = []
    num_layers = model.config.num_hidden_layers

    if isinstance(layer, str) and layer == "all":
        target_layers = list(range(num_layers))
    elif isinstance(layer, int):
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Layer {layer} out of range [0, {num_layers})")
        target_layers = [layer]
    else:
        raise ValueError(f"Invalid layer specification: {layer}")

    for l in target_layers:
        vec = steering_vectors[l]
        if method == "caa":
            hook = make_caa_hook(vec, multiplier)
        elif method == "ablation":
            hook = make_ablation_hook(vec)
        else:
            raise ValueError(f"Unknown method: {method}")

        handle = model.model.layers[l].register_forward_hook(hook)
        handles.append(handle)

    return handles


def remove_hooks(handles: list):
    """Remove all registered hooks."""
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Steered inference
# ---------------------------------------------------------------------------

def run_steered_inference(
    test_issues,
    k: int,
    model,
    tokenizer,
    max_new_tokens: int,
    max_prompt_tokens: int,
    output_csv: str,
    inference_batch_size: int = 1,
) -> Tuple[float, dict]:
    """Run inference on test issues with steering hooks already registered.

    This reuses the prompt construction and label parsing from llm_labeler.py
    but is self-contained for the inference loop.

    Returns:
        (elapsed_time, cost_stats)
    """
    is_zero_shot = (k == 0)
    mode_label = "zero-shot" if is_zero_shot else f"k={k}"

    results = []
    t0 = time.time()
    n_truncated = n_xml_parsed = n_regex_parsed = n_invalid = 0
    all_prompt_tokens = []
    all_generated_tokens = []

    print(f"\n  [{mode_label}] Starting steered inference: {len(test_issues)} issues (batch_size={inference_batch_size})")

    # Pre-build all prompts
    prepared = []
    for issue in test_issues:
        neighbors_for_prompt = issue.neighbors[:k] if not is_zero_shot else []

        messages, trunc = build_chat_messages(
            test_title=issue.title,
            test_body=issue.body,
            neighbors=neighbors_for_prompt,
            k=k,
            is_thinking_model=False,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
        )

        if trunc.truncated:
            n_truncated += 1

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt = prompt + "<label>"

        prepared.append({
            "issue": issue,
            "prompt": prompt,
            "trunc": trunc,
        })

    # Process in batches
    total_batches = (len(prepared) + inference_batch_size - 1) // inference_batch_size

    for batch_start in tqdm(range(0, len(prepared), inference_batch_size),
                            desc=f"  {mode_label}", unit="batch", total=total_batches):
        batch = prepared[batch_start:batch_start + inference_batch_size]
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
                gen_kwargs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.eos_token_id,
                    "use_cache": True,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 50,
                }
                outputs = model.generate(**gen_kwargs)

            for i, item in enumerate(batch):
                generated_ids = outputs[i][inputs.input_ids.shape[1]:]
                generated_ids = generated_ids[generated_ids != tokenizer.pad_token_id]
                generated_token_count = len(generated_ids)
                raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                raw_output = "<label>" + raw_output

                all_prompt_tokens.append(per_item_prompt_tokens[i])
                all_generated_tokens.append(generated_token_count)

                pred = parse_label(raw_output)
                has_xml = bool(re.search(r"<label>.*?</label>", raw_output, re.IGNORECASE))
                if pred == "invalid":
                    n_invalid += 1
                elif has_xml:
                    n_xml_parsed += 1
                else:
                    n_regex_parsed += 1

                issue = item["issue"]
                trunc = item["trunc"]
                results.append({
                    "test_idx": issue.idx,
                    "title": issue.title,
                    "body": issue.body,
                    "ground_truth": issue.label,
                    "predicted_label": pred,
                    "raw_output": raw_output[:300],
                    "truncated": trunc.truncated,
                    "prompt_tokens": per_item_prompt_tokens[i],
                    "generated_tokens": generated_token_count,
                })

        except Exception as e:
            for i, item in enumerate(batch):
                issue = item["issue"]
                all_prompt_tokens.append(per_item_prompt_tokens[i])
                all_generated_tokens.append(0)
                n_invalid += 1
                results.append({
                    "test_idx": issue.idx,
                    "title": issue.title,
                    "body": issue.body,
                    "ground_truth": issue.label,
                    "predicted_label": "invalid",
                    "raw_output": f"ERROR: {e}"[:300],
                    "truncated": item["trunc"].truncated,
                    "prompt_tokens": per_item_prompt_tokens[i],
                    "generated_tokens": 0,
                })

    elapsed = time.time() - t0

    # Write output
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["test_idx", "title", "body", "ground_truth", "predicted_label",
            "raw_output", "truncated", "prompt_tokens", "generated_tokens"]
    df = pd.DataFrame(results)
    df[cols].to_csv(out_path, index=False)

    total = len(df)
    print(f"  [{mode_label}] Done: {total} predictions -> {out_path}")
    print(f"    XML parsed: {n_xml_parsed} ({100 * n_xml_parsed / total:.1f}%)  "
          f"Regex: {n_regex_parsed} ({100 * n_regex_parsed / total:.1f}%)  "
          f"Invalid: {n_invalid} ({100 * n_invalid / total:.1f}%)")
    print(f"    Truncated: {n_truncated} ({100 * n_truncated / total:.1f}%)")
    print(f"    Time: {elapsed:.1f}s ({total / elapsed:.1f} issues/s)")

    gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    cost_stats = {
        "wall_time_s": round(elapsed, 2),
        "issues_per_second": round(total / elapsed, 2) if elapsed > 0 else 0.0,
        "total_issues": total,
        "total_prompt_tokens": sum(all_prompt_tokens),
        "total_generated_tokens": sum(all_generated_tokens),
        "avg_prompt_tokens": round(sum(all_prompt_tokens) / total, 1) if total > 0 else 0.0,
        "gpu_peak_memory_mb": round(gpu_peak_mb, 1),
        "invalid_count": n_invalid,
        "invalid_rate": round(100 * n_invalid / total, 2) if total > 0 else 0.0,
    }

    return elapsed, cost_stats


# ---------------------------------------------------------------------------
# Layer sweep
# ---------------------------------------------------------------------------

def run_layer_sweep(
    model,
    tokenizer,
    test_issues,
    steering_vectors: Dict[int, torch.Tensor],
    method: str,
    multiplier: float,
    k: int,
    max_new_tokens: int,
    max_prompt_tokens: int,
    output_dir: str,
    eval_dir: str | None,
    inference_batch_size: int,
):
    """Run steered inference for each layer independently, collecting per-layer metrics."""
    num_layers = model.config.num_hidden_layers
    sweep_results = []

    for layer_idx in range(num_layers):
        print(f"\n{'=' * 40}")
        print(f"  Layer sweep: layer {layer_idx}/{num_layers - 1}")
        print(f"{'=' * 40}")

        # Register hook for this layer
        handles = register_steering_hooks(
            model, steering_vectors, method, layer_idx, multiplier,
        )

        # Build output filename
        if method == "caa":
            pred_name = f"preds_{method}_layer{layer_idx}_m{multiplier}.csv"
        else:
            pred_name = f"preds_{method}_layer{layer_idx}.csv"

        pred_path = os.path.join(output_dir, "predictions", pred_name)

        # Skip if already exists
        if os.path.exists(pred_path):
            print(f"  Already exists, skipping: {pred_path}")
            remove_hooks(handles)
            # Try to load existing results for sweep summary
            try:
                existing_df = pd.read_csv(pred_path)
                sweep_results.append(_evaluate_predictions(existing_df, layer_idx, eval_dir, pred_path, method, multiplier, k))
            except Exception:
                pass
            continue

        # Reset peak memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        elapsed, cost_stats = run_steered_inference(
            test_issues=test_issues,
            k=k,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
            output_csv=pred_path,
            inference_batch_size=inference_batch_size,
        )

        # Remove hooks after this layer's run
        remove_hooks(handles)

        # Evaluate
        pred_df = pd.read_csv(pred_path)
        result = _evaluate_predictions(pred_df, layer_idx, eval_dir, pred_path, method, multiplier, k)
        sweep_results.append(result)

    # Write sweep summary
    if sweep_results:
        sweep_df = pd.DataFrame(sweep_results)
        sweep_path = os.path.join(output_dir, "layer_sweep_results.csv")
        sweep_df.to_csv(sweep_path, index=False)
        print(f"\n  Layer sweep results saved to {sweep_path}")

        # Print summary table
        print(f"\n  Layer Sweep Summary (method={method}, multiplier={multiplier}):")
        print(f"  {'Layer':>5} {'F1_macro':>8} {'F1_bug':>7} {'F1_feat':>7} {'F1_ques':>7} {'R_bug':>6} {'R_ques':>6} {'Inv%':>5}")
        print(f"  {'-' * 55}")
        best_idx = sweep_df["f1_macro"].idxmax()
        for _, row in sweep_df.iterrows():
            marker = " <-- BEST" if row.name == best_idx else ""
            print(f"  {int(row['layer']):5d} {row['f1_macro']:8.4f} {row['f1_bug']:7.3f} "
                  f"{row['f1_feature']:7.3f} {row['f1_question']:7.3f} "
                  f"{row['r_bug']:6.3f} {row['r_question']:6.3f} {row['invalid_rate']:5.1f}{marker}")

    return sweep_results


def _evaluate_predictions(
    pred_df: pd.DataFrame,
    layer_idx: int,
    eval_dir: str | None,
    pred_path: str,
    method: str,
    multiplier: float,
    k: int,
) -> dict:
    """Evaluate a predictions DataFrame and return a summary dict."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    y_true = pred_df["ground_truth"].astype(str).str.lower().str.strip().tolist()
    y_pred = pred_df["predicted_label"].astype(str).str.lower().str.strip().tolist()

    # Filter out invalid predictions for metrics
    valid_mask = [p in VALID_LABELS for p in y_pred]
    y_true_valid = [t for t, m in zip(y_true, valid_mask) if m]
    y_pred_valid = [p for p, m in zip(y_pred, valid_mask) if m]

    total = len(y_true)
    n_invalid = total - len(y_true_valid)

    if y_true_valid:
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true_valid, y_pred_valid, labels=VALID_LABELS, average=None, zero_division=0,
        )
        macro_f1 = f1.mean()
        acc = accuracy_score(y_true_valid, y_pred_valid)
    else:
        prec = rec = f1 = [0, 0, 0]
        macro_f1 = 0
        acc = 0

    result = {
        "layer": layer_idx,
        "method": method,
        "multiplier": multiplier,
        "top_k": k,
        "f1_macro": round(float(macro_f1), 4),
        "accuracy": round(float(acc), 4),
        "f1_bug": round(float(f1[0]), 4),
        "f1_feature": round(float(f1[1]), 4),
        "f1_question": round(float(f1[2]), 4),
        "p_bug": round(float(prec[0]), 4),
        "p_feature": round(float(prec[1]), 4),
        "p_question": round(float(prec[2]), 4),
        "r_bug": round(float(rec[0]), 4),
        "r_feature": round(float(rec[1]), 4),
        "r_question": round(float(rec[2]), 4),
        "invalid_count": n_invalid,
        "invalid_rate": round(100 * n_invalid / total, 2) if total > 0 else 0,
        "total": total,
    }

    # Also run evaluate.py if eval_dir provided
    if eval_dir and os.path.exists(pred_path):
        eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py")
        if os.path.exists(eval_script):
            import subprocess
            if method == "caa":
                eval_name = f"eval_{method}_layer{layer_idx}_m{multiplier}.csv"
            else:
                eval_name = f"eval_{method}_layer{layer_idx}.csv"
            eval_csv = os.path.join(eval_dir, eval_name)
            os.makedirs(eval_dir, exist_ok=True)
            subprocess.run([
                sys.executable, eval_script,
                "--preds_csv", pred_path,
                "--top_k", str(k),
                "--output_csv", eval_csv,
            ], check=False)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply activation steering during RAGTAG inference"
    )
    parser.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--neighbors_csv", required=True,
                        help="Path to neighbors CSV (e.g. neighbors_k3.csv)")
    parser.add_argument("--steering_vectors", required=True,
                        help="Path to steering_vectors.pt file")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for predictions and evaluations")
    parser.add_argument("--method", default="caa", choices=["caa", "ablation"],
                        help="Steering method (default: caa)")
    parser.add_argument("--layer", default="9",
                        help="Layer index, 'sweep' for all-layer sweep, 'all' for full ablation")
    parser.add_argument("--multiplier", type=float, default=-1.0,
                        help="Steering multiplier for CAA (default: -1.0, negative = subtract bug direction)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of neighbors to use (default: 3)")
    parser.add_argument("--max_seq_length", type=int, default=8192,
                        help="Max sequence length (default: 8192)")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Max generated tokens (default: 20)")
    parser.add_argument("--inference_batch_size", type=int, default=1,
                        help="Batch size for inference (default: 1)")
    parser.add_argument("--eval_dir", default=None,
                        help="Directory for evaluation CSVs (auto-runs evaluate.py)")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--cache_dir", default=None,
                        help="HuggingFace model cache directory")
    args = parser.parse_args()

    if args.no_4bit:
        args.load_in_4bit = False

    # Parse layer argument
    is_sweep = args.layer.lower() == "sweep"
    is_all = args.layer.lower() == "all"

    print(f"{'=' * 60}")
    print(f"  Activation Steering — RAGTAG Inference")
    print(f"{'=' * 60}")
    print(f"  Model:            {args.model}")
    print(f"  Neighbors CSV:    {args.neighbors_csv}")
    print(f"  Steering vectors: {args.steering_vectors}")
    print(f"  Method:           {args.method}")
    print(f"  Layer:            {args.layer}")
    if args.method == "caa":
        print(f"  Multiplier:       {args.multiplier}")
    print(f"  Top K:            {args.top_k}")
    print(f"  Max seq length:   {args.max_seq_length}")
    print(f"  Batch size:       {args.inference_batch_size}")
    print(f"{'=' * 60}")

    # --- Load steering vectors ---
    print(f"\nLoading steering vectors from {args.steering_vectors}")
    steering_vectors = torch.load(args.steering_vectors, map_location="cpu", weights_only=True)
    num_sv_layers = len(steering_vectors)
    sv_dim = steering_vectors[0].shape[0]
    print(f"  Loaded {num_sv_layers} layer vectors, dim={sv_dim}")

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
    print(f"  Model loaded in {model_load_time:.1f}s")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_model_layers = model.config.num_hidden_layers
    print(f"  Architecture: {num_model_layers} layers, hidden_size={model.config.hidden_size}")

    # Verify compatibility
    if num_sv_layers != num_model_layers:
        print(f"  WARNING: Steering vectors have {num_sv_layers} layers but model has {num_model_layers}")
    if sv_dim != model.config.hidden_size:
        raise ValueError(f"Steering vector dim {sv_dim} != model hidden_size {model.config.hidden_size}")

    # --- Load test issues ---
    print(f"\nLoading test issues from {args.neighbors_csv}")
    test_issues = load_test_issues(args.neighbors_csv, args.top_k)
    print(f"  Loaded {len(test_issues)} test issues with up to {args.top_k} neighbors each")

    max_prompt_tokens = args.max_seq_length - args.max_new_tokens - 20
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Run inference ---
    if is_sweep:
        # Layer sweep mode
        run_layer_sweep(
            model=model,
            tokenizer=tokenizer,
            test_issues=test_issues,
            steering_vectors=steering_vectors,
            method=args.method,
            multiplier=args.multiplier,
            k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=max_prompt_tokens,
            output_dir=args.output_dir,
            eval_dir=args.eval_dir,
            inference_batch_size=args.inference_batch_size,
        )
    else:
        # Single configuration
        layer = "all" if is_all else int(args.layer)

        # Register hooks
        handles = register_steering_hooks(
            model, steering_vectors, args.method, layer, args.multiplier,
        )

        layer_label = "all" if is_all else f"layer{layer}"
        if args.method == "caa":
            pred_name = f"preds_{args.method}_{layer_label}_m{args.multiplier}.csv"
        else:
            pred_name = f"preds_{args.method}_{layer_label}.csv"
        pred_path = os.path.join(args.output_dir, "predictions", pred_name)

        if os.path.exists(pred_path):
            print(f"\n  Predictions already exist, skipping: {pred_path}")
        else:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            elapsed, cost_stats = run_steered_inference(
                test_issues=test_issues,
                k=args.top_k,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                max_prompt_tokens=max_prompt_tokens,
                output_csv=pred_path,
                inference_batch_size=args.inference_batch_size,
            )

            # Add steering config to cost stats
            cost_stats["method"] = args.method
            cost_stats["layer"] = str(layer)
            cost_stats["multiplier"] = args.multiplier
            cost_stats["model"] = args.model

            # Save cost stats
            cost_path = os.path.join(args.output_dir, "steering_cost_metrics.json")
            with open(cost_path, "w") as f:
                json.dump(cost_stats, f, indent=2)

        # Remove hooks
        remove_hooks(handles)

        # Evaluate
        if args.eval_dir and os.path.exists(pred_path):
            pred_df = pd.read_csv(pred_path)
            result = _evaluate_predictions(
                pred_df, layer if isinstance(layer, int) else -1,
                args.eval_dir, pred_path, args.method, args.multiplier, args.top_k,
            )
            print(f"\n  Evaluation results:")
            print(f"    F1_macro:   {result['f1_macro']:.4f}")
            print(f"    F1_bug:     {result['f1_bug']:.4f}")
            print(f"    F1_feature: {result['f1_feature']:.4f}")
            print(f"    F1_question:{result['f1_question']:.4f}")
            print(f"    R_bug:      {result['r_bug']:.4f}")
            print(f"    R_question: {result['r_question']:.4f}")
            print(f"    Invalid:    {result['invalid_count']} ({result['invalid_rate']:.1f}%)")

    if torch.cuda.is_available():
        print(f"\n  GPU peak memory: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"\nDone.")


if __name__ == "__main__":
    main()

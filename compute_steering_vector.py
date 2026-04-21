#!/usr/bin/env python3
"""
compute_steering_vector.py
==========================
Compute per-layer steering vectors for activation steering during RAGTAG inference.

Implements three contrastive pair strategies:
  1. answer_conditioned (CAA-faithful): same prompt, different label continuations
  2. faiss_matched (topic-controlled): FAISS-matched bug/question pairs
  3. class_means (NTW-style): mean activation difference between classes

Usage:
  python compute_steering_vector.py \
    --model unsloth/Llama-3.2-3B-Instruct \
    --train_csv results/ablation_random_3k/neighbors_faiss/train_split.csv \
    --output_dir results/steering_vectors/llama3b_3k/ \
    --pair_strategy answer_conditioned \
    --max_pairs 300 \
    --max_seq_length 4096
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Import shared infrastructure from llm_labeler
from llm_labeler import SYSTEM_PROMPT, build_chat_messages


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------

def build_zero_shot_prompt(title: str, body: str, tokenizer, max_prompt_tokens: int,
                           continuation: str | None = None) -> str:
    """Build a zero-shot RAGTAG prompt for a single issue.

    Args:
        continuation: If provided (e.g. "bug" or "question"), append after <label> prefill.
                      Used for answer_conditioned strategy.
    """
    messages, _ = build_chat_messages(
        test_title=title,
        test_body=body,
        neighbors=[],
        k=0,
        is_thinking_model=False,
        max_prompt_tokens=max_prompt_tokens,
        tokenizer=tokenizer,
    )

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt = prompt + "<label>"

    if continuation:
        prompt = prompt + continuation

    return prompt


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states_batched(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 4,
    desc: str = "Extracting",
) -> torch.Tensor:
    """Run forward passes and return hidden states at the last non-padding token per layer.

    Returns:
        Tensor of shape (N, num_layers, hidden_dim)
    """
    all_states = []

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for batch_start in tqdm(range(0, len(prompts), batch_size), desc=desc, unit="batch"):
        batch = prompts[batch_start:batch_start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")

        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
            )

        # outputs.hidden_states: tuple of (num_layers+1) tensors, each (batch, seq, hidden_dim)
        # Index 0 = embedding layer output, 1..L = transformer layer outputs
        num_layers = len(outputs.hidden_states) - 1

        for i in range(len(batch)):
            seq_len = inputs.attention_mask[i].sum().item()
            per_layer = []
            for layer_idx in range(1, num_layers + 1):
                # Extract hidden state at last non-padding token
                h = outputs.hidden_states[layer_idx][i, seq_len - 1, :].float().cpu()
                per_layer.append(h)
            all_states.append(torch.stack(per_layer))  # (num_layers, hidden_dim)

    tokenizer.padding_side = orig_side

    return torch.stack(all_states)  # (N, num_layers, hidden_dim)


# ---------------------------------------------------------------------------
# Pair selection strategies
# ---------------------------------------------------------------------------

def strategy_answer_conditioned(
    train_df: pd.DataFrame,
    model,
    tokenizer,
    max_pairs: int,
    max_prompt_tokens: int,
    batch_size: int,
) -> torch.Tensor:
    """CAA-faithful: same prompt, different label continuations.

    For each training issue, create two prompts:
      - prompt + "<label>bug"
      - prompt + "<label>question"
    Extract hidden states at the label token, compute difference.

    Returns:
        steering_vectors: Tensor of shape (num_layers, hidden_dim)
    """
    # Balanced sample across all labels
    labels = sorted(train_df["labels"].unique())
    per_label = max_pairs // len(labels)
    sampled = []
    for lab in labels:
        group = train_df[train_df["labels"] == lab]
        n = min(per_label, len(group))
        sampled.append(group.sample(n=n, random_state=42))
    sample_df = pd.concat(sampled).reset_index(drop=True)

    print(f"  Strategy: answer_conditioned")
    print(f"  Sampled {len(sample_df)} issues ({per_label} per label)")

    # Build prompt pairs
    bug_prompts = []
    question_prompts = []
    for _, row in sample_df.iterrows():
        title = str(row.get("title", ""))
        body = str(row.get("body", ""))
        bug_prompts.append(build_zero_shot_prompt(title, body, tokenizer, max_prompt_tokens, "bug"))
        question_prompts.append(build_zero_shot_prompt(title, body, tokenizer, max_prompt_tokens, "question"))

    # Extract hidden states
    print(f"  Running forward passes for bug continuations...")
    bug_states = extract_hidden_states_batched(model, tokenizer, bug_prompts, batch_size, "bug")
    print(f"  Running forward passes for question continuations...")
    question_states = extract_hidden_states_batched(model, tokenizer, question_prompts, batch_size, "question")

    # Compute per-issue difference and average
    # bug_states, question_states: (N, num_layers, hidden_dim)
    diffs = bug_states - question_states  # (N, num_layers, hidden_dim)
    steering_vectors = diffs.mean(dim=0)  # (num_layers, hidden_dim)

    return steering_vectors


def strategy_faiss_matched(
    train_df: pd.DataFrame,
    model,
    tokenizer,
    max_pairs: int,
    max_prompt_tokens: int,
    batch_size: int,
) -> torch.Tensor:
    """Topic-controlled: FAISS-matched bug/question pairs.

    For each bug issue, find its nearest question issue by embedding similarity.
    Extract hidden states for both, compute difference.

    Returns:
        steering_vectors: Tensor of shape (num_layers, hidden_dim)
    """
    import faiss
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print(f"  Strategy: faiss_matched")

    bug_df = train_df[train_df["labels"] == "bug"].reset_index(drop=True)
    question_df = train_df[train_df["labels"] == "question"].reset_index(drop=True)

    print(f"  Bug issues: {len(bug_df)}, Question issues: {len(question_df)}")

    # Build FAISS index on question issues
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    question_texts = [
        f"{row['title']} {row['body']}" for _, row in question_df.iterrows()
    ]
    print(f"  Embedding question issues...")
    q_vectors = np.array(embedder.embed_documents(question_texts), dtype="float32")
    faiss.normalize_L2(q_vectors)
    dim = q_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(q_vectors)

    # For each bug issue, find nearest question issue
    bug_texts = [
        f"{row['title']} {row['body']}" for _, row in bug_df.iterrows()
    ]
    print(f"  Embedding bug issues and querying...")
    b_vectors = np.array(embedder.embed_documents(bug_texts), dtype="float32")
    faiss.normalize_L2(b_vectors)

    n_pairs = min(max_pairs, len(bug_df))
    # Use a random subset of bug issues if we have too many
    if len(bug_df) > max_pairs:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(bug_df), size=max_pairs, replace=False)
    else:
        indices = np.arange(len(bug_df))

    # Query for each selected bug issue
    selected_b_vectors = b_vectors[indices]
    distances, nn_indices = index.search(selected_b_vectors, 1)

    # Build prompts for matched pairs
    bug_prompts = []
    question_prompts = []
    for i, bug_idx in enumerate(indices):
        q_idx = int(nn_indices[i, 0])
        bug_row = bug_df.iloc[bug_idx]
        q_row = question_df.iloc[q_idx]

        bug_prompts.append(build_zero_shot_prompt(
            str(bug_row["title"]), str(bug_row["body"]),
            tokenizer, max_prompt_tokens,
        ))
        question_prompts.append(build_zero_shot_prompt(
            str(q_row["title"]), str(q_row["body"]),
            tokenizer, max_prompt_tokens,
        ))

    print(f"  Matched {len(bug_prompts)} bug/question pairs")
    print(f"  Mean similarity of matches: {distances.mean():.4f}")

    # Extract hidden states
    print(f"  Running forward passes for bug issues...")
    bug_states = extract_hidden_states_batched(model, tokenizer, bug_prompts, batch_size, "bug")
    print(f"  Running forward passes for question issues...")
    question_states = extract_hidden_states_batched(model, tokenizer, question_prompts, batch_size, "question")

    # Compute per-pair difference and average
    diffs = bug_states - question_states
    steering_vectors = diffs.mean(dim=0)

    return steering_vectors


def strategy_class_means(
    train_df: pd.DataFrame,
    model,
    tokenizer,
    max_prompt_tokens: int,
    batch_size: int,
) -> torch.Tensor:
    """NTW-style: mean activation difference between bug and question classes.

    Returns:
        steering_vectors: Tensor of shape (num_layers, hidden_dim)
    """
    print(f"  Strategy: class_means")

    bug_df = train_df[train_df["labels"] == "bug"].reset_index(drop=True)
    question_df = train_df[train_df["labels"] == "question"].reset_index(drop=True)

    print(f"  Bug issues: {len(bug_df)}, Question issues: {len(question_df)}")

    # Build prompts for all bug and question issues
    bug_prompts = [
        build_zero_shot_prompt(str(row["title"]), str(row["body"]), tokenizer, max_prompt_tokens)
        for _, row in bug_df.iterrows()
    ]
    question_prompts = [
        build_zero_shot_prompt(str(row["title"]), str(row["body"]), tokenizer, max_prompt_tokens)
        for _, row in question_df.iterrows()
    ]

    # Extract hidden states
    print(f"  Running forward passes for {len(bug_prompts)} bug issues...")
    bug_states = extract_hidden_states_batched(model, tokenizer, bug_prompts, batch_size, "bug")
    print(f"  Running forward passes for {len(question_prompts)} question issues...")
    question_states = extract_hidden_states_batched(model, tokenizer, question_prompts, batch_size, "question")

    # Compute class means and difference
    bug_mean = bug_states.mean(dim=0)       # (num_layers, hidden_dim)
    question_mean = question_states.mean(dim=0)

    steering_vectors = bug_mean - question_mean  # (num_layers, hidden_dim)

    return steering_vectors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-layer steering vectors from training data"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--train_csv", required=True, help="Path to train_split.csv")
    parser.add_argument("--output_dir", required=True, help="Directory to save steering vectors")
    parser.add_argument("--pair_strategy", default="answer_conditioned",
                        choices=["answer_conditioned", "faiss_matched", "class_means"],
                        help="Pair selection strategy (default: answer_conditioned)")
    parser.add_argument("--max_pairs", type=int, default=300,
                        help="Max number of pairs/issues for vector computation (default: 300)")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                        help="Max sequence length for model (default: 4096)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for forward passes (default: 4)")
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--cache_dir", default=None,
                        help="HuggingFace model cache directory")
    args = parser.parse_args()

    if args.no_4bit:
        args.load_in_4bit = False

    print(f"{'=' * 60}")
    print(f"  Steering Vector Computation")
    print(f"{'=' * 60}")
    print(f"  Model:          {args.model}")
    print(f"  Train CSV:      {args.train_csv}")
    print(f"  Strategy:       {args.pair_strategy}")
    print(f"  Max pairs:      {args.max_pairs}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Load in 4bit:   {args.load_in_4bit}")
    print(f"{'=' * 60}")

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

    # Report model architecture
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"  Architecture: {num_layers} layers, hidden_size={hidden_size}")

    # --- Load training data ---
    print(f"\nLoading training data from {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)

    # Normalize label column
    if "label" in train_df.columns and "labels" not in train_df.columns:
        train_df = train_df.rename(columns={"label": "labels"})

    label_counts = train_df["labels"].value_counts()
    print(f"  Loaded {len(train_df)} training issues")
    for lab, count in label_counts.items():
        print(f"    {lab}: {count}")

    max_prompt_tokens = args.max_seq_length - 50  # small buffer

    # --- Compute steering vectors ---
    print(f"\nComputing steering vectors...")
    t0 = time.time()

    if args.pair_strategy == "answer_conditioned":
        steering_vectors = strategy_answer_conditioned(
            train_df, model, tokenizer, args.max_pairs, max_prompt_tokens, args.batch_size,
        )
    elif args.pair_strategy == "faiss_matched":
        steering_vectors = strategy_faiss_matched(
            train_df, model, tokenizer, args.max_pairs, max_prompt_tokens, args.batch_size,
        )
    elif args.pair_strategy == "class_means":
        steering_vectors = strategy_class_means(
            train_df, model, tokenizer, max_prompt_tokens, args.batch_size,
        )

    compute_time = time.time() - t0
    print(f"\n  Steering vectors computed in {compute_time:.1f}s")
    print(f"  Shape: {steering_vectors.shape}")  # (num_layers, hidden_dim)

    # --- Save outputs ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Save vectors
    vectors_path = os.path.join(args.output_dir, "steering_vectors.pt")
    # Store as a dict mapping layer index to vector
    vectors_dict = {i: steering_vectors[i] for i in range(steering_vectors.shape[0])}
    torch.save(vectors_dict, vectors_path)
    print(f"  Saved steering vectors to {vectors_path}")

    # Save per-layer norms
    norms = []
    for i in range(steering_vectors.shape[0]):
        norm = steering_vectors[i].norm().item()
        norms.append({"layer": i, "l2_norm": round(norm, 4)})
    norms_df = pd.DataFrame(norms)
    norms_path = os.path.join(args.output_dir, "per_layer_norms.csv")
    norms_df.to_csv(norms_path, index=False)
    print(f"  Saved per-layer norms to {norms_path}")

    # Print norms summary
    print(f"\n  Per-layer L2 norms:")
    for i, row in norms_df.iterrows():
        bar = "#" * int(row["l2_norm"] / norms_df["l2_norm"].max() * 30)
        print(f"    Layer {int(row['layer']):2d}: {row['l2_norm']:.4f}  {bar}")

    # Save metadata
    metadata = {
        "model": args.model,
        "train_csv": args.train_csv,
        "pair_strategy": args.pair_strategy,
        "max_pairs": args.max_pairs,
        "max_seq_length": args.max_seq_length,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "compute_time_s": round(compute_time, 2),
        "model_load_time_s": round(model_load_time, 2),
        "num_train_issues": len(train_df),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")

    if torch.cuda.is_available():
        print(f"\n  GPU peak memory: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")

    print(f"\nDone.")


if __name__ == "__main__":
    main()

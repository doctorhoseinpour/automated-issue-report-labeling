#!/usr/bin/env python3
"""
tri_classifier.py
=================
Tri-binary RAGTAG ensemble for issue report classification.

For each test issue, run THREE binary RAGTAG inferences:
  - bug-vs-feature
  - bug-vs-question
  - feature-vs-question

Each binary call retrieves only neighbors whose label is in the allowed pair
(top-K of those, by similarity). The three predictions are majority-voted.
On 1-1-1 ties or 2+ invalid binaries, fall back to similarity-weighted VOTAG
over all retrieved neighbors as a tiebreaker.

This script imports utilities from llm_labeler.py and vtag.py without
modifying them.

Usage:
  python tri_classifier.py \\
      --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \\
      --neighbors_csv results/issues11k/project_specific/<proj>/neighbors/neighbors_k30.csv \\
      --output_dir results/issues11k/project_specific/<proj>/<model_tag>/triclassifier \\
      --top_k 6 \\
      --max_seq_length 8192 \\
      --inference_batch_size 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

# Imports from existing pipeline (no modifications to those files).
from llm_labeler import (
    VALID_LABELS,
    VALID_LABELS_SET,
    parse_label,
    _count_tokens,
    _truncate_text_by_tokens,
    TruncationInfo,
)
from vtag import vote, canonicalize_label


PAIRS: List[Tuple[str, str]] = [
    ("bug", "feature"),
    ("bug", "question"),
    ("feature", "question"),
]
PAIR_KEYS: List[str] = ["bug_feature", "bug_question", "feature_question"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class TestIssueWithSim:
    idx: int
    title: str
    body: str
    label: str
    neighbors: List[Dict]  # each dict: title, body, label, rank, similarity


def load_test_issues_with_similarity(csv_path: str, pool_k: int) -> List[TestIssueWithSim]:
    """Load a neighbors CSV (as produced by build_11k_index.py), grouping by
    test_idx. Returns list of issues with neighbors keeping `similarity`,
    `rank`, `title`, `body`, `label` (label canonicalized)."""
    df = pd.read_csv(csv_path)
    required = {"test_idx", "neighbor_rank", "neighbor_label", "neighbor_similarity",
                "test_title", "test_body", "test_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"neighbors CSV missing required columns: {missing}. "
            f"Re-run build_11k_index.py to regenerate."
        )

    df = df.sort_values(["test_idx", "neighbor_rank"]).reset_index(drop=True)
    issues: Dict[int, TestIssueWithSim] = {}
    for _, row in df.iterrows():
        ti = int(row["test_idx"])
        rank = int(row["neighbor_rank"])
        if rank >= pool_k:
            continue
        if ti not in issues:
            issues[ti] = TestIssueWithSim(
                idx=ti,
                title=str(row.get("test_title", "")),
                body=str(row.get("test_body", "")),
                label=str(row.get("test_label", "")),
                neighbors=[],
            )
        issues[ti].neighbors.append({
            "rank": rank,
            "similarity": float(row["neighbor_similarity"]),
            "title": str(row.get("neighbor_title", "")),
            "body": str(row.get("neighbor_body", "")),
            "label": canonicalize_label(row["neighbor_label"]),
        })
    # Sort each issue's neighbors by rank ascending (= similarity descending).
    for issue in issues.values():
        issue.neighbors.sort(key=lambda n: n["rank"])
    return [issues[i] for i in sorted(issues.keys())]


# ---------------------------------------------------------------------------
# Per-pair retrieval filtering
# ---------------------------------------------------------------------------

def filter_neighbors_for_pair(
    neighbors: List[Dict],
    pair: Tuple[str, str],
    top_k: int,
) -> List[Dict]:
    """Keep neighbors whose label is in `pair`, take top-K by rank ascending.
    Filter-then-slice (so the third class doesn't push allowed-label neighbors
    out of the budget). May return fewer than top_k if the rare class has too
    few neighbors in the pool."""
    allowed = set(pair)
    filtered = [n for n in neighbors if n["label"] in allowed]
    return filtered[:top_k]


# ---------------------------------------------------------------------------
# Binary chat-message builder (truncation logic mirrors llm_labeler.build_chat_messages)
# ---------------------------------------------------------------------------

def build_binary_chat_messages(
    test_title: str,
    test_body: str,
    neighbors: List[Dict],
    pair: Tuple[str, str],
    max_prompt_tokens: int,
    tokenizer,
):
    """Mirror of llm_labeler.build_chat_messages but with a binary system
    prompt restricted to the two allowed labels."""
    trunc = TruncationInfo()
    a, b = pair
    system = (
        "Classify the GitHub issue into exactly one category.\n"
        "\n"
        "Rules:\n"
        "1. Read the issue title and body.\n"
        f"2. Choose one label: {a} or {b}.\n"
        "3. Respond with ONLY the label wrapped in XML tags.\n"
        "4. Do NOT write anything else. No explanation. No reasoning. No extra text.\n"
        "\n"
        "Correct response format examples:\n"
        f"<label>{a}</label>\n"
        f"<label>{b}</label>"
    )

    def format_issue(title, body):
        return f"Title: {title}\nBody: {body}"

    def format_label(label):
        return f"<label>{label}</label>"

    sys_tokens = _count_tokens(system, tokenizer)
    total_overhead = sys_tokens + 50

    neighbor_data = []
    for nb in neighbors:
        t = str(nb.get("title", ""))
        body_text = str(nb.get("body", ""))
        lab = nb["label"]  # already canonicalized
        # Defensive: if a non-pair label slipped through, skip the neighbor.
        if lab not in (a, b):
            continue
        issue_text = format_issue(t, body_text)
        neighbor_data.append({
            "title": t, "body": body_text, "label": lab,
            "issue_text": issue_text,
            "issue_tokens": _count_tokens(issue_text, tokenizer),
            "body_tokens": _count_tokens(body_text, tokenizer),
            "title_tokens": _count_tokens(t, tokenizer),
        })

    test_issue_text = format_issue(test_title, test_body)
    test_tokens = _count_tokens(test_issue_text, tokenizer)

    total_content_tokens = test_tokens + sum(nd["issue_tokens"] for nd in neighbor_data)
    trunc.original_tokens = total_content_tokens
    budget = max(100, max_prompt_tokens - total_overhead)

    if total_content_tokens > budget:
        trunc.truncated = True
        query_reserve = int(budget * 0.3)
        neighbor_budget = budget - query_reserve
        total_nb_title = sum(nd["title_tokens"] for nd in neighbor_data)
        nb_body_budget = neighbor_budget - total_nb_title
        total_nb_body = sum(nd["body_tokens"] for nd in neighbor_data)

        if nb_body_budget > 0 and total_nb_body > nb_body_budget:
            trunc.neighbors_truncated = True
            for nd in neighbor_data:
                ratio = nd["body_tokens"] / total_nb_body if total_nb_body > 0 else 1.0 / max(1, len(neighbor_data))
                max_b = max(5, int(nb_body_budget * ratio))
                if nd["body_tokens"] > max_b:
                    nd["body"], _ = _truncate_text_by_tokens(nd["body"], max_b, tokenizer)
                    nd["body_tokens"] = max_b
                nd["issue_text"] = format_issue(nd["title"], nd["body"])
                nd["issue_tokens"] = nd["title_tokens"] + nd["body_tokens"]
        elif nb_body_budget <= 0:
            trunc.neighbors_truncated = True
            for nd in neighbor_data:
                nd["body"], _ = _truncate_text_by_tokens(nd["body"], 5, tokenizer)
                nd["body_tokens"] = min(5, nd["body_tokens"])
                nd["issue_text"] = format_issue(nd["title"], nd["body"])
                nd["issue_tokens"] = nd["title_tokens"] + nd["body_tokens"]

        used = sum(nd["issue_tokens"] for nd in neighbor_data)
        q_budget = budget - used
        if test_tokens > q_budget and q_budget > 0:
            trunc.query_truncated = True
            tt_tokens = _count_tokens(test_title, tokenizer)
            bb = q_budget - tt_tokens
            if bb > 10:
                test_body, _ = _truncate_text_by_tokens(test_body, bb, tokenizer)
            else:
                tb = max(5, int(q_budget * 0.4))
                bb = max(5, q_budget - tb)
                test_title, _ = _truncate_text_by_tokens(test_title, tb, tokenizer)
                test_body, _ = _truncate_text_by_tokens(test_body, bb, tokenizer)
            test_issue_text = format_issue(test_title, test_body)
            test_tokens = _count_tokens(test_issue_text, tokenizer)

    trunc.final_tokens = test_tokens + sum(nd["issue_tokens"] for nd in neighbor_data)
    trunc.tokens_removed = trunc.original_tokens - trunc.final_tokens

    messages = [{"role": "system", "content": system}]
    user_content = ""
    if neighbor_data:
        user_content += "Here are some examples of correctly classified issues:\n\n"
        for i, nd in enumerate(neighbor_data, 1):
            user_content += f"--- Example {i} ---\n{nd['issue_text']}\nAnswer: {format_label(nd['label'])}\n\n"
        user_content += "Now, classify the following target issue:\n\n"
    user_content += test_issue_text
    messages.append({"role": "user", "content": user_content})
    return messages, trunc


# ---------------------------------------------------------------------------
# Vote aggregation
# ---------------------------------------------------------------------------

def parse_binary_label(raw: str, pair: Tuple[str, str]) -> str:
    """Parse with llm_labeler.parse_label, then constrain to the pair.
    Anything outside the pair (including 'invalid' from the parser) -> 'invalid'."""
    pred = parse_label(raw)
    if pred in pair:
        return pred
    return "invalid"


def aggregate_votes(
    binary_preds: Dict[str, str],
    all_neighbors: List[Dict],
) -> Tuple[str, bool, str]:
    """Majority-vote across the three binary predictions. On 1-1-1 ties,
    1-1-with-invalid, or 2+invalids: fall back to similarity-weighted VOTAG
    over all retrieved neighbors (all 3 classes). Returns:
      (final_label, tie_flag, tiebreaker_reason)
    """
    counts: Dict[str, int] = defaultdict(int)
    invalid_count = 0
    for pred in binary_preds.values():
        if pred == "invalid":
            invalid_count += 1
        else:
            counts[pred] += 1

    if counts:
        max_v = max(counts.values())
        winners = [c for c, v in counts.items() if v == max_v]
        if len(winners) == 1 and max_v >= 2:
            return winners[0], False, "none"

    # Tiebreaker required.
    if invalid_count >= 2:
        reason = ">=2_invalid"
    elif invalid_count == 1:
        reason = "1-1_with_invalid"
    else:
        reason = "1-1-1"

    # Need at least one neighbor for VOTAG. If somehow none, default to bug
    # (extremely unlikely with k=30 pool).
    if not all_neighbors:
        return "bug", True, reason + "_no_neighbors"

    final = vote(all_neighbors, k=len(all_neighbors), voting="similarity")
    return final, True, reason


# ---------------------------------------------------------------------------
# Subprocess eval
# ---------------------------------------------------------------------------

def run_eval(preds_csv: str, eval_csv: str, model_name: str, top_k: int):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, "evaluate.py")
    if not os.path.exists(eval_script):
        print(f"  WARNING: evaluate.py not found at {eval_script}, skipping eval")
        return
    print(f"\n  Evaluating {preds_csv} -> {eval_csv}")
    Path(eval_csv).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, eval_script,
        "--preds_csv", preds_csv,
        "--top_k", str(top_k),
        "--output_csv", eval_csv,
        "--model_name", model_name,
    ], check=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tri-binary RAGTAG ensemble")
    parser.add_argument("--model", required=True)
    parser.add_argument("--neighbors_csv", required=True,
                        help="Path to neighbors_k{N}.csv with N >= vote_pool_k")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=6,
                        help="Per-pair top-K of allowed-label neighbors (default: 6)")
    parser.add_argument("--vote_pool_k", type=int, default=30,
                        help="Pool size loaded for filtering and VOTAG tiebreaker (default: 30)")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--inference_batch_size", type=int, default=8)
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--no_eval", action="store_true",
                        help="Disable auto-evaluation after inference")
    parser.add_argument("--model_name_for_eval", default=None)
    args = parser.parse_args()

    if args.no_4bit:
        args.load_in_4bit = False
    do_eval = not args.no_eval

    K = args.top_k
    POOL_K = args.vote_pool_k

    pred_dir = os.path.join(args.output_dir, "predictions")
    eval_dir = os.path.join(args.output_dir, "evaluations")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    output_csv = os.path.join(pred_dir, f"preds_k{K}.csv")
    eval_csv = os.path.join(eval_dir, f"eval_k{K}.csv")

    if os.path.exists(output_csv):
        print(f"  SKIP: predictions already exist at {output_csv}")
        if do_eval and not os.path.exists(eval_csv):
            run_eval(output_csv, eval_csv, args.model_name_for_eval or args.model, K)
        return

    print("=" * 60)
    print("  Tri-binary classifier — pairwise majority vote")
    print("=" * 60)
    print(f"  Model:         {args.model}")
    print(f"  K (per pair):  {K}")
    print(f"  Pool K:        {POOL_K}")
    print(f"  Output:        {output_csv}")
    print(f"  Batch size:    {args.inference_batch_size}")
    print(f"  Max seq len:   {args.max_seq_length}")
    print(f"  Decoding:      greedy (do_sample=False)")
    print("=" * 60)

    # Load model
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.cache_dir, "hub")
        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"  Cache dir:     {args.cache_dir}")

    print("\nLoading model via Unsloth...")
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
    print(f"  Model loaded in {model_load_time:.1f}s")

    print(f"\nLoading test issues from {args.neighbors_csv} (pool_k={POOL_K})...")
    test_issues = load_test_issues_with_similarity(args.neighbors_csv, POOL_K)
    print(f"  Loaded {len(test_issues)} test issues")

    max_prompt_tokens = args.max_seq_length - args.max_new_tokens - 20

    # Pre-build all 3 prompts per issue: 3*N total prompts.
    print("\nPre-building binary prompts...")
    prepared = []
    issue_lookup: Dict[int, TestIssueWithSim] = {}
    n_short_filter = defaultdict(int)
    for issue in test_issues:
        issue_lookup[issue.idx] = issue
        for pair, pair_key in zip(PAIRS, PAIR_KEYS):
            filtered = filter_neighbors_for_pair(issue.neighbors, pair, K)
            if len(filtered) < K:
                n_short_filter[pair_key] += 1
            messages, trunc = build_binary_chat_messages(
                test_title=issue.title,
                test_body=issue.body,
                neighbors=filtered,
                pair=pair,
                max_prompt_tokens=max_prompt_tokens,
                tokenizer=tokenizer,
            )
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            prompt = prompt + "<label>"
            prepared.append({
                "test_idx": issue.idx,
                "pair": pair,
                "pair_key": pair_key,
                "prompt": prompt,
                "trunc": trunc,
                "filtered_count": len(filtered),
            })
    print(f"  {len(prepared)} prompts ({len(test_issues)} issues × 3 pairs)")
    for k, n in n_short_filter.items():
        print(f"  Filtered <{K} neighbors [{k}]: {n}/{len(test_issues)}")

    # Batched inference
    raw_outputs: Dict[Tuple[int, str], str] = {}
    prompt_tokens: Dict[Tuple[int, str], int] = {}
    generated_tokens: Dict[Tuple[int, str], int] = {}

    print("\nRunning inference...")
    t_inf0 = time.time()
    bs = args.inference_batch_size
    total_batches = (len(prepared) + bs - 1) // bs

    for batch_start in tqdm(range(0, len(prepared), bs),
                            desc="  inference", unit="batch", total=total_batches):
        batch = prepared[batch_start: batch_start + bs]
        prompts = [item["prompt"] for item in batch]

        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        tokenizer.padding_side = orig_side

        per_item_pt = [
            (inputs.attention_mask[i] != 0).sum().item()
            for i in range(len(batch))
        ]

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,  # greedy decoding
                    use_cache=True,
                )
            for i, item in enumerate(batch):
                gen_ids = outputs[i][inputs.input_ids.shape[1]:]
                gen_ids = gen_ids[gen_ids != tokenizer.pad_token_id]
                gen_count = len(gen_ids)
                raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                raw = "<label>" + raw  # restore prefill in stored output
                key = (item["test_idx"], item["pair_key"])
                raw_outputs[key] = raw
                prompt_tokens[key] = per_item_pt[i]
                generated_tokens[key] = gen_count
        except Exception as e:
            print(f"  Batch {batch_start} failed: {e}")
            for i, item in enumerate(batch):
                key = (item["test_idx"], item["pair_key"])
                raw_outputs[key] = ""
                prompt_tokens[key] = per_item_pt[i]
                generated_tokens[key] = 0

    inference_time = time.time() - t_inf0
    print(f"  Inference done in {inference_time:.1f}s "
          f"({len(test_issues)/inference_time:.2f} issues/s)")

    # Aggregate per issue
    print("\nAggregating votes...")
    results = []
    n_tied = 0
    n_invalid_per_pair: Dict[str, int] = defaultdict(int)
    tiebreaker_reasons: Dict[str, int] = defaultdict(int)

    for issue in test_issues:
        binary_preds: Dict[str, str] = {}
        binary_raws: Dict[str, str] = {}
        for pair, pair_key in zip(PAIRS, PAIR_KEYS):
            key = (issue.idx, pair_key)
            raw = raw_outputs.get(key, "")
            pred = parse_binary_label(raw, pair)
            if pred == "invalid":
                n_invalid_per_pair[pair_key] += 1
            binary_preds[pair_key] = pred
            binary_raws[pair_key] = raw[:200]

        final, tie_flag, tiebreaker_reason = aggregate_votes(binary_preds, issue.neighbors)
        if tie_flag:
            n_tied += 1
        tiebreaker_reasons[tiebreaker_reason] += 1

        nc_bf = sum(1 for n in issue.neighbors if n["label"] in ("bug", "feature"))
        nc_bq = sum(1 for n in issue.neighbors if n["label"] in ("bug", "question"))
        nc_fq = sum(1 for n in issue.neighbors if n["label"] in ("feature", "question"))

        results.append({
            "test_idx": issue.idx,
            "title": issue.title,
            "body": issue.body,
            "ground_truth": canonicalize_label(issue.label),
            "pred_bug_feature": binary_preds["bug_feature"],
            "pred_bug_question": binary_preds["bug_question"],
            "pred_feature_question": binary_preds["feature_question"],
            "predicted_label": final,
            "tie_flag": tie_flag,
            "tiebreaker_reason": tiebreaker_reason,
            "raw_outputs": json.dumps(binary_raws),
            "neighbor_count_bf": min(nc_bf, K),
            "neighbor_count_bq": min(nc_bq, K),
            "neighbor_count_fq": min(nc_fq, K),
            "prompt_tokens_bf": prompt_tokens.get((issue.idx, "bug_feature"), 0),
            "prompt_tokens_bq": prompt_tokens.get((issue.idx, "bug_question"), 0),
            "prompt_tokens_fq": prompt_tokens.get((issue.idx, "feature_question"), 0),
            "generated_tokens_bf": generated_tokens.get((issue.idx, "bug_feature"), 0),
            "generated_tokens_bq": generated_tokens.get((issue.idx, "bug_question"), 0),
            "generated_tokens_fq": generated_tokens.get((issue.idx, "feature_question"), 0),
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    n = len(df)
    print(f"  Wrote {n} rows to {output_csv}")
    print(f"  Tied (VOTAG fallback): {n_tied}/{n} = {n_tied/n*100:.1f}%")
    for reason, count in sorted(tiebreaker_reasons.items(), key=lambda x: -x[1]):
        if reason != "none":
            print(f"    {reason}: {count}")
    for k_pair, n_inv in n_invalid_per_pair.items():
        print(f"  Invalid binary [{k_pair}]: {n_inv}/{n} = {n_inv/n*100:.1f}%")

    # Auto-evaluate
    if do_eval:
        run_eval(output_csv, eval_csv, args.model_name_for_eval or args.model, K)

    # Cost metrics
    cost_csv = os.path.join(args.output_dir, "cost_metrics.csv")
    cost_stats = {
        "model": args.model,
        "top_k": K,
        "pool_k": POOL_K,
        "model_load_time_s": round(model_load_time, 2),
        "inference_time_s": round(inference_time, 2),
        "issues_per_second": round(len(test_issues) / inference_time, 3) if inference_time > 0 else 0,
        "total_prompts_run": len(prepared),
        "total_prompt_tokens": sum(prompt_tokens.values()),
        "avg_prompt_tokens": round(sum(prompt_tokens.values()) / max(len(prompt_tokens), 1), 1),
        "total_generated_tokens": sum(generated_tokens.values()),
        "n_tied": n_tied,
        "n_invalid_bf": n_invalid_per_pair["bug_feature"],
        "n_invalid_bq": n_invalid_per_pair["bug_question"],
        "n_invalid_fq": n_invalid_per_pair["feature_question"],
        "max_seq_length": args.max_seq_length,
        "max_new_tokens": args.max_new_tokens,
        "load_in_4bit": args.load_in_4bit,
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 0)
        if torch.cuda.is_available() else 0,
    }
    pd.DataFrame([cost_stats]).to_csv(cost_csv, index=False)
    print(f"  Cost metrics: {cost_csv}")
    print(f"\nDone. Total wall: {model_load_time + inference_time:.1f}s "
          f"(load {model_load_time:.1f}s + infer {inference_time:.1f}s)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
openai_labeler.py
=================
OpenAI variant of RAGTAG few-shot inference — the API counterpart of
``llm_labeler.py``.  It exists to remove a model-diversity threat to validity:
all RAGTAG experiments so far use only Qwen2.5.  This script reruns RAGTAG with
OpenAI models (``gpt-3.5-turbo`` then ``gpt-4o``) so the results can be compared
against the LoRA fine-tuning numbers of Heo & Lee (2025).

It replicates the existing RAGTAG setup faithfully — same prompt format, same
truncation budget, same neighbor files, same prediction-CSV schema — swapping
only the inference backend from Unsloth/Qwen to the OpenAI Chat Completions API.

ISOLATION: this script does not import ``llm_labeler.py`` (that module imports
torch/unsloth at load).  The pure-logic functions below are COPIED VERBATIM from
``llm_labeler.py`` as of git commit 41dddc0 — keep them in sync if that file
changes:  VALID_LABELS, _CANON_MAP, SYSTEM_PROMPT, _strip_think, _is_label_list,
parse_label, TruncationInfo, _count_tokens, _truncate_text_by_tokens,
build_chat_messages, TestIssue, load_test_issues.

Differences from llm_labeler.py (by design):
  * No Unsloth/torch, no batching — one Chat Completions request per issue.
  * No "<label>" assistant prefill (an Unsloth trick); the full response is
    parsed with the copied parse_label().
  * Token counting/truncation uses tiktoken instead of a HF tokenizer.  GPT-3.5
    uses cl100k_base and GPT-4o uses o200k_base; both differ slightly from
    Qwen's tokenizer, so truncation cut points are approximate (documented
    threat-to-validity footnote).  max_seq_length stays 8192 to keep the
    *content budget* identical to the original 8k Qwen runs even though the
    OpenAI models' real context windows are far larger.

Pricing (USD per 1M tokens, as of 2026-05; override with --input_price /
--output_price):  gpt-3.5-turbo $0.50 in / $1.50 out;  gpt-4o $2.50 in / $10 out.

Usage:
  python openai_labeler.py --model gpt-3.5-turbo \\
    --neighbors_dir results/issues11k/project_specific/ansible_ansible/neighbors \\
    --top_ks "12,15" --output_dir <out>/predictions --eval_dir <out>/evaluations \\
    --model_name_for_eval gpt-3.5-turbo

  # Canary (30 issues, must write OUTSIDE results/):
  python openai_labeler.py --model gpt-3.5-turbo --canary 30 \\
    --neighbors_dir <...>/neighbors --top_ks 12 \\
    --output_dir canary_openai/openai_gpt_3_5_turbo/<proj>/ragtag/predictions
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# ===========================================================================
# === BEGIN verbatim copy from llm_labeler.py (commit 41dddc0) ==============
# ===========================================================================

VALID_LABELS = ["bug", "feature", "question"]
VALID_LABELS_SET = set(VALID_LABELS)

_CANON_MAP = {
    "enhancement": "feature", "feature-request": "feature",
    "feature_request": "feature", "feat": "feature", "request": "feature",
    "bugfix": "bug", "defect": "bug", "issue": "bug", "fix": "bug",
    "support": "question", "howto": "question", "help": "question",
}

SYSTEM_PROMPT = """Classify the GitHub issue into exactly one category.

Rules:
1. Read the issue title and body.
2. Choose one label: bug, feature, or question.
3. Respond with ONLY the label wrapped in XML tags.
4. Do NOT write anything else. No explanation. No reasoning. No extra text.

Correct response format examples:
<label>bug</label>
<label>feature</label>
<label>question</label>"""


def _strip_think(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<think>.*$", "", s, flags=re.DOTALL | re.IGNORECASE)
    return s


def _is_label_list(text: str) -> bool:
    start = text[:80].lower().strip()
    return bool(re.search(r'bug[,\s]+feature[,\s]+(or\s+)?question', start) or
                re.search(r'feature[,\s]+bug[,\s]+(or\s+)?question', start) or
                re.search(r'bug[,\s]+question[,\s]+(or\s+)?feature', start))


def parse_label(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return "invalid"

    # Layer 1: XML tag — take the LAST <label>...</label>
    label_matches = re.findall(r"<label>\s*(.*?)\s*</label>", raw, re.IGNORECASE | re.DOTALL)
    if label_matches:
        candidate = label_matches[-1].strip().lower()
        if candidate in VALID_LABELS_SET:
            return candidate
        if candidate in _CANON_MAP:
            return _CANON_MAP[candidate]
        squash = re.sub(r"[^a-z]", "", candidate)
        for v in VALID_LABELS:
            if v == squash:
                return v

    # Layer 2: Strip think blocks + regex
    s = _strip_think(raw).strip()
    if not s:
        return "invalid"

    if _is_label_list(s):
        after = re.search(r'bug[,\s]+feature[,\s]+(?:or\s+)?question[.\s]*\n*(.*)',
                          s, re.DOTALL | re.IGNORECASE)
        if after:
            remainder = after.group(1).strip()
            tag_match = re.search(r"<label>\s*(.*?)\s*</label>", remainder, re.IGNORECASE)
            if tag_match:
                t = tag_match.group(1).strip().lower()
                if t in VALID_LABELS_SET:
                    return t
            for tok in re.findall(r"[A-Za-z_\-]+", remainder):
                t = tok.lower().strip("-_ ")
                if t in VALID_LABELS_SET:
                    return t
                if t in _CANON_MAP:
                    return _CANON_MAP[t]
        return "invalid"

    # Layer 3: First valid word
    tokens = re.findall(r"[A-Za-z_\-]+", s)
    if not tokens:
        return "invalid"
    tok = tokens[0].lower().strip("-_ ")
    if tok in VALID_LABELS_SET:
        return tok
    if tok in _CANON_MAP:
        return _CANON_MAP[tok]
    squash = re.sub(r"[^a-z]", "", tok)
    for v in VALID_LABELS:
        if v == squash:
            return v
    for k, v in _CANON_MAP.items():
        if re.sub(r"[^a-z]", "", k) == squash:
            return v
    return "invalid"


def _truncate_text_by_tokens(text, max_tokens, tokenizer):
    if not text:
        return "", 0
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text, 0
    removed = len(token_ids) - max_tokens
    truncated_ids = token_ids[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True).rstrip() + "...", removed


@dataclass
class TruncationInfo:
    truncated: bool = False
    neighbors_truncated: bool = False
    query_truncated: bool = False
    original_tokens: int = 0
    final_tokens: int = 0
    tokens_removed: int = 0


def _count_tokens(text, tokenizer):
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_chat_messages(test_title, test_body, neighbors, k, is_thinking_model,
                        max_prompt_tokens, tokenizer):
    trunc = TruncationInfo()
    system = SYSTEM_PROMPT  # is_thinking_model unused in the OpenAI variant

    def format_issue(title, body):
        return f"Title: {title}\nBody: {body}"

    def format_label(label):
        return f"<label>{label}</label>"

    sys_tokens = _count_tokens(system, tokenizer)
    total_overhead = sys_tokens + 50

    neighbor_data = []
    for nb in neighbors:
        t = str(nb.get("title", ""))
        b = str(nb.get("body", ""))
        lab = str(nb.get("label", "")).strip().lower()
        if lab in _CANON_MAP:
            lab = _CANON_MAP[lab]
        if lab not in VALID_LABELS_SET:
            lab = "bug"
        issue_text = format_issue(t, b)
        neighbor_data.append({
            "title": t, "body": b, "label": lab,
            "issue_text": issue_text,
            "issue_tokens": _count_tokens(issue_text, tokenizer),
            "body_tokens": _count_tokens(b, tokenizer),
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


@dataclass
class TestIssue:
    idx: int
    title: str
    body: str
    label: str
    created_at: str
    neighbors: List[Dict[str, str]] = field(default_factory=list)


def load_test_issues(csv_path: str, k: int) -> List[TestIssue]:
    df = pd.read_csv(csv_path)
    issues: Dict[int, TestIssue] = {}
    for _, row in df.iterrows():
        ti = int(row["test_idx"])
        if ti not in issues:
            issues[ti] = TestIssue(
                idx=ti,
                title=str(row.get("test_title", "")),
                body=str(row.get("test_body", "")),
                label=str(row.get("test_label", "")),
                created_at=str(row.get("test_created_at", "")),
            )
        if row.get("neighbor_rank") is not None and int(row["neighbor_rank"]) < k:
            issues[ti].neighbors.append({
                "title": str(row.get("neighbor_title", "")),
                "body": str(row.get("neighbor_body", "")),
                "label": str(row.get("neighbor_label", "")),
            })
    return [issues[k_] for k_ in sorted(issues.keys())]

# ===========================================================================
# === END verbatim copy =====================================================
# ===========================================================================


# Default OpenAI pricing (USD per 1M tokens), as of 2026-05.
PRICE_TABLE = {
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-4o":        {"input": 2.50, "output": 10.00},
}


class TiktokenAdapter:
    """Wraps a tiktoken encoding behind the tokenizer interface that the copied
    build_chat_messages()/_truncate_text_by_tokens() expect."""

    def __init__(self, model: str):
        import tiktoken
        try:
            self.enc = tiktoken.encoding_for_model(model)
        except KeyError:
            # Unknown alias — cl100k_base is a safe, widely-compatible default.
            self.enc = tiktoken.get_encoding("cl100k_base")

    def encode(self, text, add_special_tokens=False):
        return self.enc.encode(text or "")

    def decode(self, ids, skip_special_tokens=True):
        return self.enc.decode(ids)


def call_openai_with_retry(client, model, messages, temperature, top_p,
                           max_tokens, seed, max_retries=6):
    """Single Chat Completions request with exponential backoff on transient
    errors.  Returns (content, usage, resolved_model, system_fingerprint,
    n_retries).  Raises on non-retryable errors."""
    from openai import (RateLimitError, APITimeoutError, APIConnectionError,
                        InternalServerError)
    retryable = (RateLimitError, APITimeoutError, APIConnectionError,
                 InternalServerError)
    delay = 2.0
    n_retries = 0
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
            )
            content = resp.choices[0].message.content or ""
            return (content, resp.usage, resp.model,
                    getattr(resp, "system_fingerprint", None), n_retries)
        except retryable as e:
            if attempt >= max_retries:
                raise
            n_retries += 1
            retry_after = getattr(e, "retry_after", None)
            wait = retry_after if retry_after else delay + random.uniform(0, delay)
            wait = min(wait, 60.0)
            print(f"    [retry {n_retries}/{max_retries}] {type(e).__name__}: "
                  f"waiting {wait:.1f}s")
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


def stratified_sample(test_issues, n, seed):
    """Deterministic label-balanced sample of n issues, drawn round-robin across
    ground-truth labels so the canary exercises all three classes (the test
    split is sorted by label, so a plain head-N draw is degenerate)."""
    buckets: Dict[str, List] = {}
    for iss in test_issues:
        lab = str(iss.label).strip().lower()
        lab = _CANON_MAP.get(lab, lab)
        buckets.setdefault(lab, []).append(iss)
    rng = random.Random(seed)
    for lab in buckets:
        rng.shuffle(buckets[lab])
    picked, labels = [], sorted(buckets)
    cursors = {lab: 0 for lab in labels}
    while len(picked) < n and any(cursors[l] < len(buckets[l]) for l in labels):
        for lab in labels:
            if cursors[lab] < len(buckets[lab]) and len(picked) < n:
                picked.append(buckets[lab][cursors[lab]])
                cursors[lab] += 1
    picked.sort(key=lambda i: i.idx)
    return picked


def run_one_k(test_issues, k, is_zero_shot, client, model, tokenizer,
              temperature, top_p, max_new_tokens, max_prompt_tokens, seed,
              output_csv, log_file, concurrency,
              input_price, output_price):
    """Run inference for a single K value (or zero-shot) via the OpenAI API."""
    mode_label = "zero-shot" if is_zero_shot else f"k={k}"
    print(f"\n  [{mode_label}] Starting inference: {len(test_issues)} issues "
          f"(concurrency={concurrency})")

    # --- Pre-build all prompts + truncation info ---
    prepared = []
    n_truncated = n_nb_truncated = n_q_truncated = 0
    for issue in test_issues:
        neighbors_for_prompt = issue.neighbors[:k] if not is_zero_shot else []
        messages, trunc = build_chat_messages(
            test_title=issue.title, test_body=issue.body,
            neighbors=neighbors_for_prompt, k=k, is_thinking_model=False,
            max_prompt_tokens=max_prompt_tokens, tokenizer=tokenizer,
        )
        if trunc.truncated:
            n_truncated += 1
        if trunc.neighbors_truncated:
            n_nb_truncated += 1
        if trunc.query_truncated:
            n_q_truncated += 1
        prepared.append({"issue": issue, "messages": messages, "trunc": trunc})

    resolved_model = {"name": model}  # captured from first successful response

    def process(item):
        issue = item["issue"]
        trunc = item["trunc"]
        try:
            content, usage, rmodel, fp, n_retries = call_openai_with_retry(
                client, model, item["messages"], temperature, top_p,
                max_new_tokens, seed,
            )
            resolved_model["name"] = rmodel
            pred = parse_label(content)
            has_xml = bool(re.search(r"<label>.*?</label>", content, re.IGNORECASE))
            parsed_via = "xml" if (has_xml and pred != "invalid") else (
                "regex" if pred != "invalid" else "failed")
            return {
                "test_idx": issue.idx, "title": issue.title, "body": issue.body,
                "ground_truth": issue.label, "predicted_label": pred,
                "raw_output": content[:300],
                "truncated": trunc.truncated,
                "neighbors_truncated": trunc.neighbors_truncated,
                "query_truncated": trunc.query_truncated,
                "tokens_removed": trunc.tokens_removed,
                "parsed_via": parsed_via,
                "prompt_tokens": usage.prompt_tokens,
                "generated_tokens": usage.completion_tokens,
                "_n_retries": n_retries, "_failed": False,
            }
        except Exception as e:
            # Non-retryable error or retries exhausted — record as failed,
            # never abort a costly run for one bad issue.
            return {
                "test_idx": issue.idx, "title": issue.title, "body": issue.body,
                "ground_truth": issue.label, "predicted_label": "invalid",
                "raw_output": f"ERROR: {e}"[:300],
                "truncated": trunc.truncated,
                "neighbors_truncated": trunc.neighbors_truncated,
                "query_truncated": trunc.query_truncated,
                "tokens_removed": trunc.tokens_removed,
                "parsed_via": "failed", "prompt_tokens": 0, "generated_tokens": 0,
                "_n_retries": 0, "_failed": True,
            }

    t0 = time.time()
    results: List[dict] = []
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for r in tqdm(ex.map(process, prepared), total=len(prepared),
                          desc=f"  {mode_label}", unit="issue"):
                results.append(r)
    else:
        for item in tqdm(prepared, desc=f"  {mode_label}", unit="issue"):
            results.append(process(item))
    elapsed = time.time() - t0

    results.sort(key=lambda r: r["test_idx"])

    n_invalid = sum(1 for r in results if r["predicted_label"] == "invalid")
    n_failed = sum(1 for r in results if r["_failed"])
    n_xml = sum(1 for r in results if r["parsed_via"] == "xml")
    n_regex = sum(1 for r in results if r["parsed_via"] == "regex")
    n_retries = sum(r["_n_retries"] for r in results)
    total = len(results)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps({
                    "test_idx": r["test_idx"], "raw_output": r["raw_output"],
                    "parsed_label": r["predicted_label"],
                    "ground_truth": r["ground_truth"],
                    "truncated": r["truncated"],
                    "tokens_removed": r["tokens_removed"],
                }) + "\n")

    cols = ["test_idx", "title", "body", "ground_truth", "predicted_label",
            "raw_output", "truncated", "neighbors_truncated", "query_truncated",
            "tokens_removed", "parsed_via", "prompt_tokens", "generated_tokens"]
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results)[cols].to_csv(out_path, index=False)

    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_gen = sum(r["generated_tokens"] for r in results)
    input_cost = total_prompt / 1e6 * input_price
    output_cost = total_gen / 1e6 * output_price

    print(f"  [{mode_label}] Done: {total} predictions -> {out_path}")
    print(f"    XML: {n_xml} ({100*n_xml/total:.1f}%)  "
          f"Regex: {n_regex} ({100*n_regex/total:.1f}%)  "
          f"Invalid: {n_invalid} ({100*n_invalid/total:.1f}%)  "
          f"Failed(API): {n_failed}")
    print(f"    Truncated: {n_truncated} ({100*n_truncated/total:.1f}%)  "
          f"Retries: {n_retries}")
    print(f"    Time: {elapsed:.1f}s ({total/elapsed:.2f} issues/s)  "
          f"Cost: ${input_cost + output_cost:.4f}")

    prompt_toks = [r["prompt_tokens"] for r in results]
    cost_stats = {
        "model": model,
        "model_resolved": resolved_model["name"],
        "top_k": k,
        "k_label": "zero_shot" if is_zero_shot else f"k{k}",
        "total_issues": total,
        "total_prompt_tokens": total_prompt,
        "total_generated_tokens": total_gen,
        "avg_prompt_tokens": round(total_prompt / total, 1) if total else 0.0,
        "avg_generated_tokens": round(total_gen / total, 1) if total else 0.0,
        "min_prompt_tokens": min(prompt_toks) if prompt_toks else 0,
        "max_prompt_tokens": max(prompt_toks) if prompt_toks else 0,
        "input_price_per_1m": input_price,
        "output_price_per_1m": output_price,
        "input_cost_usd": round(input_cost, 4),
        "output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(input_cost + output_cost, 4),
        "wall_time_s": round(elapsed, 2),
        "issues_per_second": round(total / elapsed, 2) if elapsed > 0 else 0.0,
        "n_retries": n_retries,
        "n_failed": n_failed,
        "temperature": temperature,
        "seed": seed,
        "max_seq_length": max_prompt_tokens,  # informational
    }
    return cost_stats


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI RAGTAG few-shot labeler — API counterpart of llm_labeler.py")
    parser.add_argument("--model", required=True, help="OpenAI model id (gpt-3.5-turbo, gpt-4o)")
    parser.add_argument("--neighbors_dir", required=True)
    parser.add_argument("--top_ks", required=True, help="Comma-separated K values (0 = zero-shot)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--eval_dir", default=None,
                        help="If set, runs evaluate.py after each K")
    parser.add_argument("--model_name_for_eval", default=None)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Parallel API requests (forced to 1 in canary mode)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N test issues")
    parser.add_argument("--canary", type=int, default=None,
                        help="Canary mode: implies --limit N and refuses to write under results/")
    parser.add_argument("--input_price", type=float, default=None,
                        help="USD per 1M input tokens (overrides built-in table)")
    parser.add_argument("--output_price", type=float, default=None,
                        help="USD per 1M output tokens (overrides built-in table)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY is not set. Export it before running.")

    # Canary mode: forbid writing under results/, force sequential.
    if args.canary is not None:
        args.concurrency = 1
        results_root = os.path.abspath("results")
        if os.path.abspath(args.output_dir).startswith(results_root):
            sys.exit(f"ERROR: canary mode must NOT write under results/. "
                     f"Got --output_dir {args.output_dir}")

    ks = [int(x) for x in args.top_ks.split(",")]

    # Pricing.
    price = PRICE_TABLE.get(args.model, {"input": 0.0, "output": 0.0})
    input_price = args.input_price if args.input_price is not None else price["input"]
    output_price = args.output_price if args.output_price is not None else price["output"]
    if input_price == 0.0 and output_price == 0.0:
        print(f"  [WARN] No pricing for '{args.model}' — cost will report $0. "
              f"Pass --input_price/--output_price.")

    print("=" * 60)
    print("  OpenAI RAGTAG Labeler")
    print("=" * 60)
    print(f"  Model:           {args.model}")
    print(f"  K values:        {ks}")
    print(f"  max_seq_length:  {args.max_seq_length}")
    print(f"  canary:          {args.canary}  limit: {args.limit}")
    print(f"  concurrency:     {args.concurrency}")
    print(f"  pricing $/1M:    in={input_price}  out={output_price}")
    print("=" * 60)

    from openai import OpenAI
    client = OpenAI()
    tokenizer = TiktokenAdapter(args.model)

    max_prompt_tokens = args.max_seq_length - args.max_new_tokens - 20
    os.makedirs(args.output_dir, exist_ok=True)

    real_ks = [k for k in ks if k > 0]
    max_k = max(real_ks) if real_ks else 1
    max_k_file = os.path.join(args.neighbors_dir, f"neighbors_k{max_k}.csv")
    if not os.path.exists(max_k_file):
        import glob
        nb_files = glob.glob(os.path.join(args.neighbors_dir, "neighbors_k*.csv"))
        def extract_k(p):
            try:
                return int(os.path.basename(p).replace("neighbors_k", "").replace(".csv", ""))
            except ValueError:
                return 0
        nb_files = [f for f in nb_files if extract_k(f) >= max_k]
        if not nb_files:
            sys.exit(f"ERROR: no neighbor file with >= {max_k} neighbors in {args.neighbors_dir}")
        nb_files.sort(key=extract_k)
        max_k_file = nb_files[0]
        print(f"  Using neighbor file: {max_k_file}")

    print(f"\nLoading test issues from {max_k_file} (max_k={max_k})...")
    test_issues = load_test_issues(max_k_file, max_k)
    print(f"  Loaded {len(test_issues)} test issues")
    if args.canary is not None:
        test_issues = stratified_sample(test_issues, args.canary, args.seed)
        from collections import Counter
        dist = Counter(_CANON_MAP.get(str(i.label).strip().lower(),
                                      str(i.label).strip().lower()) for i in test_issues)
        print(f"  Canary: label-balanced sample of {len(test_issues)} issues "
              f"-> {dict(dist)}")
    elif args.limit is not None:
        test_issues = test_issues[:args.limit]
        print(f"  Limited to first {len(test_issues)} issues")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, "evaluate.py")
    all_cost_stats = []

    for k in ks:
        is_zero_shot = (k == 0)
        k_label = "zero_shot" if is_zero_shot else f"k{k}"
        output_csv = os.path.join(args.output_dir, f"preds_{k_label}.csv")
        log_file = os.path.join(args.log_dir, f"{k_label}.jsonl") if args.log_dir else None

        if os.path.exists(output_csv):
            print(f"\n  [{k_label}] Predictions already exist, SKIPPING: {output_csv}")
            continue

        cost_stats = run_one_k(
            test_issues=test_issues, k=k, is_zero_shot=is_zero_shot,
            client=client, model=args.model, tokenizer=tokenizer,
            temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=args.max_new_tokens, max_prompt_tokens=max_prompt_tokens,
            seed=args.seed, output_csv=output_csv, log_file=log_file,
            concurrency=args.concurrency,
            input_price=input_price, output_price=output_price,
        )
        all_cost_stats.append(cost_stats)

        if args.eval_dir and os.path.exists(eval_script):
            import subprocess
            os.makedirs(args.eval_dir, exist_ok=True)
            eval_csv = os.path.join(args.eval_dir, f"eval_{k_label}.csv")
            eval_model_name = args.model_name_for_eval or args.model
            print(f"  [{k_label}] Evaluating...")
            subprocess.run([
                sys.executable, eval_script,
                "--preds_csv", output_csv,
                "--top_k", str(k),
                "--output_csv", eval_csv,
                "--model_name", eval_model_name,
            ], check=False)

    if all_cost_stats:
        cost_csv = os.path.join(args.output_dir, "cost_metrics.csv")
        # Merge with any existing cost rows so re-runs accumulate cleanly.
        if os.path.exists(cost_csv):
            prev = pd.read_csv(cost_csv)
            done_labels = set(prev["k_label"])
            new_rows = [c for c in all_cost_stats if c["k_label"] not in done_labels]
            combined = pd.concat([prev, pd.DataFrame(new_rows)], ignore_index=True)
        else:
            combined = pd.DataFrame(all_cost_stats)
        combined.to_csv(cost_csv, index=False)
        print(f"\n  Cost metrics written to: {cost_csv}")
        print(f"  This run total cost: "
              f"${sum(c['total_cost_usd'] for c in all_cost_stats):.4f}")

    print("\nAll K values complete.")


if __name__ == "__main__":
    main()

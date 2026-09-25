#!/usr/bin/env python3
"""Run a Qwen2.5-Instruct (bnb-4bit, loaded via Unsloth) over RAGTAG prompts and
record, for every issue, the model's *decision state* instead of a generated label:

  logp[i, 3]         log-softmax over the full vocabulary at the answer position,
                     read at the three label tokens (bug, feature, question)
  h_L{l}[i, d]       hidden state at the answer position (last prompt token, i.e.
                     right after the "<label>" prefill) for selected layers l

The prompt is byte-identical to llm_labeler.py (same system prompt, same
"Here are some examples..." user turn, same proportional truncation, same
assistant prefill). k = 0 gives the zero-shot prompt. Demonstrations for query
uid q come from a neighbors file (uid -> ranked list of index uids).

Idempotent: skips if the output .npz exists. Writes <out>.npz + <out>.json.

Usage (lab machine, repo root):
  venv/bin/python scripts/experiments/rag_next/llm_features.py \
      --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --k 0 --roles inner,dev,test \
      --max_seq_length 4096 --out results/issues11k/exploration/rag_next/features/q7_k0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))  # repo root, for llm_labeler
from common import LABELS, load_pool  # noqa: E402

LABEL_TOKEN_STR = LABELS


def build_prompts(pool, uids, k, neighbors, tokenizer, max_prompt_tokens, debias_margin=None,
                  label_override=None):
    from llm_labeler import build_chat_messages, _debias_neighbors
    by_uid = pool.set_index("uid")
    prompts, info = [], []
    for u in uids:
        r = by_uid.loc[u]
        nbs = []
        if k > 0:
            for v in [v for v in neighbors[u] if v >= 0][:k]:
                rv = by_uid.loc[v]
                lab = rv["label"] if label_override is None else label_override.get(v, rv["label"])
                nbs.append({"title": rv["title"], "body": rv["body"], "label": lab})
            if debias_margin is not None:
                nbs = _debias_neighbors(nbs, debias_margin)
        msgs, trunc = build_chat_messages(
            test_title=r["title"], test_body=r["body"], neighbors=nbs, k=k,
            is_thinking_model=False, max_prompt_tokens=max_prompt_tokens, tokenizer=tokenizer)
        p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "<label>"
        prompts.append(p)
        info.append((trunc.truncated, trunc.query_truncated, len(nbs)))
    return prompts, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--neighbors", default=None, help="npz with uids + nb (uid x K) index uids")
    ap.add_argument("--roles", default="inner,dev,test")
    ap.add_argument("--projects", default=None, help="optional comma list of proj tags")
    ap.add_argument("--max_seq_length", type=int, default=8192)
    ap.add_argument("--layers", default="0.5,0.625,0.75,0.875,1.0",
                    help="fractions of depth for hidden states (1.0 = final, post-norm)")
    ap.add_argument("--tok_budget", type=int, default=24000, help="padded tokens per batch")
    ap.add_argument("--max_batch", type=int, default=1,
                    help="1 = exact and batch-independent (as the paper's batch-1 runs); >1 pads")
    ap.add_argument("--debias_margin", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.with_suffix(".npz").exists():
        print(f"SKIP: {out}.npz exists")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    pool = load_pool()
    roles = args.roles.split(",")
    sel = pool[pool.role.isin(roles)]
    if args.projects:
        sel = sel[sel.proj.isin(args.projects.split(","))]
    uids = sel["uid"].to_numpy()

    neighbors = None
    if args.k > 0:
        z = np.load(args.neighbors)
        neighbors = {int(u): [int(v) for v in row] for u, row in zip(z["uids"], z["nb"])}

    from unsloth import FastLanguageModel
    t0 = time.time()
    model, tok = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=args.max_seq_length,
                                                   dtype=None, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    load_s = time.time() - t0
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    label_ids = [tok.encode(s, add_special_tokens=False) for s in LABEL_TOKEN_STR]
    assert all(len(x) == 1 for x in label_ids), label_ids
    label_ids = [x[0] for x in label_ids]

    max_prompt_tokens = args.max_seq_length - 50 - 20  # paper runs: llm_labeler --max_new_tokens 50 (budget = ctx - 50 - 20)
    prompts, info = build_prompts(pool, uids, args.k, neighbors, tok, max_prompt_tokens,
                                  debias_margin=args.debias_margin)
    enc = [tok.encode(p, add_special_tokens=False) for p in prompts]
    lens = np.array([len(e) for e in enc])
    over = int((lens > args.max_seq_length).sum())
    print(f"{len(enc)} prompts; tokens p50={np.median(lens):.0f} p95={np.percentile(lens,95):.0f} "
          f"max={lens.max()} over_ctx={over}")
    # hard cap: keep the last max_seq_length tokens is wrong for prompts (would drop the system turn),
    # so we cap by cutting the middle of the query only if still too long (rare).
    for i, e in enumerate(enc):
        if len(e) > args.max_seq_length:
            cut = len(e) - args.max_seq_length
            tail = e[-60:]
            enc[i] = e[: len(e) - 60 - cut] + tail

    n_layers = model.config.num_hidden_layers
    fr = [float(x) for x in args.layers.split(",")]
    layer_idx = sorted(set(min(n_layers, max(1, int(round(f * n_layers)))) for f in fr))
    print(f"layers: {layer_idx} of {n_layers}")

    # capture only the answer-position vector of the selected layers (memory-safe for 32B @ 8k)
    captured, pooled, ctx = {}, {}, {}
    # tokens shared by every prompt (system turn + user header) are excluded from mean pooling
    a, b = enc[0], enc[-1]
    n_prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    print(f"shared prompt prefix: {n_prefix} tokens (excluded from mean pooling)")

    def make_hook(l):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            captured[l] = t[:, -1, :].detach()
            w = ctx["pool_mask"]
            pooled[l] = ((t.float() * w[..., None]).sum(1) / w.sum(1, keepdim=True).clamp(min=1)).detach()
        return hook

    handles = [model.model.layers[l - 1].register_forward_hook(make_hook(l)) for l in layer_idx]
    if n_layers not in layer_idx:
        handles.append(model.model.layers[n_layers - 1].register_forward_hook(make_hook(n_layers)))

    order = np.argsort(-np.array([len(e) for e in enc]))
    H = {l: np.zeros((len(enc), model.config.hidden_size), dtype=np.float16) for l in layer_idx}
    M = {l: np.zeros((len(enc), model.config.hidden_size), dtype=np.float16) for l in layer_idx if l < n_layers}
    logp = np.zeros((len(enc), 3), dtype=np.float32)
    top1 = np.zeros(len(enc), dtype=np.int64)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    i = 0
    nb = 0
    lm_head = model.lm_head
    with torch.inference_mode():
        while i < len(order):
            L = len(enc[order[i]])
            bs = max(1, min(args.max_batch, args.tok_budget // max(L, 1)))
            idx = order[i:i + bs]
            batch = [enc[j] for j in idx]
            ml = max(len(b) for b in batch)
            ids = torch.full((len(batch), ml), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(batch), ml), dtype=torch.long)
            for r, b in enumerate(batch):
                ids[r, ml - len(b):] = torch.tensor(b)
                att[r, ml - len(b):] = 1
            ids, att = ids.cuda(), att.cuda()
            pos = (att.cumsum(-1) - 1).clamp(min=0)
            ctx["pool_mask"] = ((pos >= n_prefix) & (att > 0)).float()
            captured.clear()
            padded = bool((att == 0).any())
            # Call the causal-LM wrapper, not model.model: Unsloth's wrapper supplies the causal
            # mask (xformers LowerTriangularMask); the bare base model gets none when a batch has
            # no padding and would attend bidirectionally.
            o = model(input_ids=ids, attention_mask=att if padded else None,
                      position_ids=pos if padded else None, use_cache=False, logits_to_keep=1)
            lg = o.logits[:, -1, :].float()
            captured[n_layers] = model.model.norm(captured[n_layers])  # final state, post-norm
            lsm = torch.log_softmax(lg, dim=-1)
            logp[idx] = lsm[:, label_ids].cpu().numpy()
            top1[idx] = lg.argmax(-1).cpu().numpy()
            for l in layer_idx:
                H[l][idx] = captured[l].float().cpu().numpy().astype(np.float16)
                if l in M:
                    M[l][idx] = pooled[l].cpu().numpy().astype(np.float16)
            if nb == 0:  # consistency: lm_head(final state) must reproduce the wrapper's logits
                lg2 = lm_head(captured[n_layers].to(lm_head.weight.dtype)).float()
                print(f"  check: max|lm_head(norm(h_last)) - logits| = {(lg2 - lg).abs().max().item():.4f}")
            del o
            i += bs
            nb += 1
            if nb % 20 == 0:
                el = time.time() - t1
                print(f"  {i}/{len(order)} ({el:.0f}s, {i/el:.1f} it/s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB)",
                      flush=True)
    wall = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2**20
    np.savez(out.with_suffix(".npz"), uids=uids, logp=logp, top1=top1,
             **{f"h_L{l}": H[l] for l in layer_idx}, **{f"m_L{l}": M[l] for l in M},
             truncated=np.array([x[0] for x in info]), query_truncated=np.array([x[1] for x in info]),
             n_demos=np.array([x[2] for x in info]), prompt_tokens=np.array([len(e) for e in enc]))
    meta = dict(model=args.model, k=args.k, neighbors=args.neighbors, roles=roles, n=len(enc),
                max_seq_length=args.max_seq_length, layers=layer_idx, n_layers=n_layers,
                wall_time_s=round(wall, 1), model_load_s=round(load_s, 1), gpu_peak_mb=round(peak, 1),
                gpu=torch.cuda.get_device_name(0), top1_is_label=float(np.isin(top1, label_ids).mean()),
                debias_margin=args.debias_margin, over_ctx=over)
    json.dump(meta, open(out.with_suffix(".json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

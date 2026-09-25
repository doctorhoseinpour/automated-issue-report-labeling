#!/usr/bin/env python3
"""Decode-throughput benchmark for the cost model: greedy generation of N new tokens for six
routed-dev p3-style prompts (≈3.5k prompt tokens each), at batch 1 and at batch 6, with the same
Unsloth bnb-4bit loading as the pilot. Reports prefill+decode wall time and decode tokens/s.

Usage (lab machine): venv/bin/python scripts/experiments/agentic/decode_bench.py --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
from adjudicate import LM, Ctx, sys_cot  # noqa: E402
from common import EXP, load_pool  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--new_tokens", type=int, default=256)
    args = ap.parse_args()
    lm = LM(args.model, 10240)
    ctx = Ctx(load_pool(), lm, 300)
    uids = ctx.uids[:6]
    prompts = [lm.render([{"role": "system", "content": sys_cot(ctx.row(u)["repo"])},
                          {"role": "user", "content": ctx.user_msg(u, True)}]) for u in uids]
    ntok = [ctx.v.ntok(p) for p in prompts]
    out = dict(model=args.model, prompt_tokens=ntok)
    lm.generate(prompts[:1], max_new_tokens=8)  # warm-up
    for bs in [1, 6]:
        lm.torch.cuda.synchronize()
        t0 = time.time()
        # min_new_tokens is not exposed; measure actual generated tokens instead
        texts, ng = lm.generate(prompts, max_new_tokens=args.new_tokens, max_batch=bs, tok_budget=10**9)
        lm.torch.cuda.synchronize()
        wall = time.time() - t0
        t1 = time.time()
        lm.score([p + "<label>" for p in prompts])  # prefill-only reference for the same prompts
        pre = time.time() - t1
        out[f"bs{bs}"] = dict(wall_s=round(wall, 2), gen_tokens=int(sum(ng)), prefill_only_s=round(pre, 2),
                              decode_tok_per_s=round(sum(ng) / max(wall - pre, 1e-6), 1))
        print(json.dumps(out[f"bs{bs}"]), flush=True)
    d = EXP / "probes"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "decode_bench.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()

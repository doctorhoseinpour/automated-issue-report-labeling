#!/usr/bin/env python3
"""Smoke test of one model through run_llm.Runner before any production run.

Hard checks (exit 1 on failure):
  * label words: first in-context tokens after "<label>" are distinct
  * causality: logits at the last prompt position are unchanged when tokens are appended
    (the Unsloth bare-model bug of rag_next showed 1.3-10 nats here)
  * the captured final-layer state reproduces the model's logits through lm_head
    (+ final logit soft-capping), i.e. the hooks read the state the model decides from
  * all decoder layers are captured, states are finite
  * parse rate >= 80% at K=0 and at K=15 (format sanity; the paper saw 3-5% invalids at high k)
Also reports: rendered prompts, kernels in use, per-prompt timing (states, K=0 gen, K=15 gen),
peak memory. Writes raw/<tag>/smoke.json.

  python smoke.py --tag qw35_9b
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
from nm_common import MODELS, NM_RAW, RN_FEATS, VAL_FILE, load_pool  # noqa: E402
from run_llm import Runner, build_prompt, load_neighbors, log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(MODELS))
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    fails, rep = [], {}

    pool = load_pool()
    by_uid = pool.set_index("uid")
    val = pd.read_csv(VAL_FILE)
    uids = val.groupby("proj").head(1)["uid"].astype(int).tolist()[: args.n]  # one per project
    nb = load_neighbors(RN_FEATS / "nb_PS_raw_dev.npz")

    r = Runner(args.tag)
    log(f"loaded {args.tag} in {r.load_s:.0f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    p0, _, _ = build_prompt(r, by_uid, uids[0], 0, nb)
    p2, _, _ = build_prompt(r, by_uid, uids[0], 2, nb)
    print("---- K=0 prompt head/tail:\n", repr(p0[:420]), "\n ... \n", repr(p0[-260:]))
    print("---- K=2 prompt tail:\n", repr(p2[-420:]))
    ids = r.set_label_ids(p0)
    log(f"label first tokens {ids} -> {r.info['label_tokens']}")
    for lab in ("bug", "feature", "question"):
        full = r.tok.encode(p0 + lab + "</label>", add_special_tokens=False)
        base = r.tok.encode(p0, add_special_tokens=False)
        print(f"   continuation '{lab}</label>' -> {[r.tok.decode([t]) for t in full[len(base):]]}")
    rep["label_tokens"] = r.info["label_tokens"]

    # ---- structure: decoder, quantization, BOS, tokenizer parity, kernels
    import bitsandbytes as bnb
    n4 = sum(isinstance(m, bnb.nn.Linear4bit) for m in r.model.modules())
    head = r.model.get_output_embeddings()
    rep["structure"] = {"decoder_is_model.model.language_model": r.dec is getattr(getattr(r.model, "model", None), "language_model", None),
                        "linear4bit_modules": n4, "lm_head_dtype": str(head.weight.dtype), "tokenizer": type(r.tok).__name__}
    log(f"structure: {rep['structure']}")
    if n4 == 0 or head.weight.dtype != torch.bfloat16:
        fails.append("quantization/lm_head")
    e0_ids = r.tok.encode(p0, add_special_tokens=False)
    bos = r.tok.bos_token_id
    nbos = sum(t == bos for t in e0_ids) if bos is not None else 0
    rep["bos"] = {"bos_id": bos, "count": nbos, "first_is_bos": bool(bos is not None and e0_ids[0] == bos)}
    log(f"BOS: {rep['bos']}; prompt tail tokens {[r.tok.decode([t]) for t in e0_ids[-8:]]}")
    if r.tag in ("gm4_12b", "mi3_8b") and not (nbos == 1 and e0_ids[0] == bos):
        fails.append("BOS")
    if r.tag == "qw35_9b":
        kern = r.info.get("qwen_kernels", {})
        log(f"qwen kernels: {kern}")
        if not str(kern.get("torch_chunk_gated_delta_rule", "")).startswith("fla."):
            fails.append("qwen fla kernel not active")
    rep["kernels"] = r.info.get("qwen_kernels")
    log(f"info: {json.dumps(r.info)}")

    # ---- causality: prefix logits must not depend on appended tokens
    e0, _ = r.encode(p0)
    extra = r.tok.encode(" and some appended text that must not change earlier positions.", add_special_tokens=False)
    with torch.inference_mode():
        x = torch.tensor([e0], device="cuda")
        a = r.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1).logits[0, -1]
        y = torch.tensor([e0 + extra], device="cuda")
        b = r.model(input_ids=y, attention_mask=torch.ones_like(y), use_cache=False,
                    logits_to_keep=len(extra) + 1).logits[0, 0]
    la, lb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
    top = torch.topk(la, 10).indices
    d_lab = (la[ids] - lb[ids]).abs().max().item()
    d_top = (la[top] - lb[top]).abs().max().item()
    rep["causal"] = {"max_dlogp_labels": d_lab, "max_dlogp_top10": d_top, "argmax_equal": int(a.argmax()) == int(b.argmax())}
    log(f"causality: {rep['causal']}")
    # bf16 logits move in ~0.06-nat steps; a causal leak (rag_next 4.3) moved them 1.3-10 nats
    if d_lab > 0.5 or d_top > 0.5 or int(a.argmax()) != int(b.argmax()):
        fails.append("causality")

    # ---- states: reconstruction through lm_head
    H, lp, top1, lg = r.states(e0)
    if H.shape != (r.n_layers, r.info["hidden"]) or not np.isfinite(H).all():
        fails.append(f"states shape/finite {H.shape}")
    head = r.model.get_output_embeddings()
    with torch.inference_mode():
        rec = head(torch.tensor(H[-1], device="cuda", dtype=torch.bfloat16)[None])[0]  # bf16, as the model
        if r.softcap:
            rec = torch.tanh(rec / r.softcap) * r.softcap
        rec = rec.float()
    d_rec = (rec[ids] - lg[ids]).abs().max().item()
    rep["reconstruction"] = {"max_abs_label_logit_diff": d_rec, "top1_equal": int(rec.argmax()) == top1,
                             "absmax_state_per_layer": [float(np.abs(h).max()) for h in H]}
    log(f"reconstruction: diff {d_rec:.4f}, top1 equal {int(rec.argmax()) == top1}; "
        f"state |max| first/mid/last {np.abs(H[0]).max():.1f}/{np.abs(H[len(H) // 2]).max():.1f}/{np.abs(H[-1]).max():.1f}")
    if d_rec > 0.1:
        fails.append("reconstruction")
    if np.abs(H).max() > 6e4:
        log("NOTE: some state exceeds the fp16 range -> parts will be stored as fp32")

    # ---- timing and parse sanity
    states_lp = {}
    for name, k in [("states_k0", None), ("gen_k0", 0), ("gen_k15", 15)]:
        rows = []
        for u in uids:
            p, trunc, nd = build_prompt(r, by_uid, u, k or 0, nb)
            e, capped = r.encode(p)
            if r.tok.encode(p, add_special_tokens=False) != r.tok_ref.encode(p, add_special_tokens=False).ids:
                fails.append(f"tokenizer parity uid {u} k {k}")
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if k is None:
                _, lps, t1s, _ = r.states(e); out = {}
                states_lp[u] = (lps, t1s)
            else:
                raw, ng, lp, t1 = r.gen(e)
                from llm_labeler import parse_label
                out = {"raw": raw[:80], "pred": parse_label(raw), "n_gen": ng}
                if k == 0 and u in states_lp:
                    out["dlp_vs_states"] = float(np.abs(lp - states_lp[u][0]).max())
            torch.cuda.synchronize()
            rows.append({"uid": u, "tok": len(e), "trunc": trunc.truncated, "demos": nd, "s": time.perf_counter() - t0, **out})
        df = pd.DataFrame(rows)
        rep[name] = {"mean_s": df.s.mean(), "mean_tok": df.tok.mean(), "tok_per_s": df.tok.sum() / df.s.sum()}
        if k is not None:
            rep[name]["parse_rate"] = float((df.pred != "invalid").mean())
            rep[name]["mean_gen_tokens"] = float(df.n_gen.mean())
            if rep[name]["parse_rate"] < 0.8:
                fails.append(f"parse rate {name}")
            if k == 0:
                rep[name]["max_dlp_vs_states"] = float(df.dlp_vs_states.max())
                if rep[name]["max_dlp_vs_states"] > 0.5:
                    fails.append("gen first-step vs states log-probs")
        print(f"---- {name}\n{df.drop(columns=['uid']).to_string(index=False)}")
        log(f"{name}: {json.dumps(rep[name])}")
    rep["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    rep["info"] = r.info
    rep["fails"] = fails
    out = NM_RAW / args.tag / "smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1, default=str)
    log(f"peak {rep['peak_gb']:.1f} GB; FAILS: {fails or 'none'} -> {out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

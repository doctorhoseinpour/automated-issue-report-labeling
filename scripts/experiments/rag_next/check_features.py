#!/usr/bin/env python3
"""Sanity check for llm_features.py (run on the GPU before any extraction).

For a few dev prompts of different lengths, compares
  (a) reference: model(input_ids).logits[0, -1] (the causal-LM wrapper, full logits)
  (b) extractor path: wrapper with logits_to_keep=1 + layer hooks, and
      lm_head(norm(last-layer hook)) reconstructed from the captured state
  (c) the bare base model model.model(...) without padding (known Unsloth pitfall:
      no causal mask is applied -> should disagree badly)
  (d) a left-padded batch through the wrapper (drift vs (a))
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2]))
from common import load_pool  # noqa: E402
from llm_features import build_prompts  # noqa: E402

model_name = sys.argv[1] if len(sys.argv) > 1 else "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
from unsloth import FastLanguageModel  # noqa: E402

model, tok = FastLanguageModel.from_pretrained(model_name=model_name, max_seq_length=8192, dtype=None, load_in_4bit=True)
FastLanguageModel.for_inference(model)
pool = load_pool()
uids = pool[pool.role == "dev"].uid.to_numpy()[:8]
prompts, _ = build_prompts(pool, uids, 0, None, tok, 8122)
LAB = [2313, 12753, 7841]
enc = [tok.encode(p, add_special_tokens=False) for p in prompts]
n_layers = model.config.num_hidden_layers
cap = {}


def mk(l):
    def h(_m, _i, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        cap[l] = t[:, -1, :].detach()
    return h


hooks = [model.model.layers[l - 1].register_forward_hook(mk(l)) for l in (n_layers // 2, n_layers)]
lsm = lambda x: torch.log_softmax(x.float(), -1)[..., LAB]
ref = []
with torch.inference_mode():
    for e in enc:
        ids = torch.tensor([e]).cuda()
        full = model(input_ids=ids).logits[0, -1]
        cap.clear()
        o = model(input_ids=ids, use_cache=False, logits_to_keep=1)
        mine = o.logits[0, -1]
        rec = model.lm_head(model.model.norm(cap[n_layers])).float()[0]
        bare = model.lm_head(model.model(input_ids=ids, use_cache=False)[0][:, -1, :]).float()[0]
        ref.append(lsm(full).cpu())
        print(f"len={len(e):5d} | wrapper(keep=1) vs full: {(lsm(mine) - lsm(full)).abs().max():.4f} nats"
              f" | lm_head(norm(hook)) vs full: {(lsm(rec) - lsm(full)).abs().max():.4f}"
              f" | BARE model.model vs full: {(lsm(bare) - lsm(full)).abs().max():.4f}"
              f" | argmax {tok.decode([int(full.argmax())])!r}")
    ml = max(len(e) for e in enc)
    ids = torch.full((len(enc), ml), tok.pad_token_id)
    att = torch.zeros((len(enc), ml), dtype=torch.long)
    for r, e in enumerate(enc):
        ids[r, ml - len(e):] = torch.tensor(e); att[r, ml - len(e):] = 1
    ids, att = ids.cuda(), att.cuda()
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    ob = model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False, logits_to_keep=1)
    for r in range(len(enc)):
        print(f"padded batch row {r}: drift vs single = {(lsm(ob.logits[r, -1]).cpu() - ref[r]).abs().max():.4f} nats")

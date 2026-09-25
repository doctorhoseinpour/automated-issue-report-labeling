#!/usr/bin/env python3
"""GPU runner for the newllms study: plain transformers + bitsandbytes NF4, batch size 1.

Modes
  states  One prefill of the zero-shot RAGTAG prompt per issue (no generation). Saves the
          answer-position hidden state (last prompt token, right after the "<label>" prefill)
          of EVERY decoder layer (the final layer post-norm, as rag_next), the log-softmax at
          the three label tokens, and the top-1 token. Inputs of the read-out and kNN vote.
  gen     The RAGTAG baseline. For every (uid, K) pair: the paper's K-shot prompt with PS
          neighbours from an npz (uid -> ranked index uids), greedy decoding (max 50 new
          tokens, stop at "</label>"), llm_labeler.parse_label, plus the first-step
          log-probs of the three label tokens (the constrained-decoding variant).

The prompt is the paper's: llm_labeler.build_chat_messages (system prompt, "Here are some
examples..." user turn, proportional truncation to ctx - 50 - 20 tokens), rendered with the
model's chat template (enable_thinking=False) + the "<label>" prefill, tokenized WITHOUT
adding special tokens (Gemma/Ministral templates already contain BOS).

Work is split into shards (--shard/--nshards or env NM_SHARD/NM_NSHARDS; round-robin over
items sorted by approximate length) and each shard into parts of --part_size items. A part
file is written atomically and skipped on restart, so a cancelled (preempted) job is simply
resubmitted. A done_<shard>.json marker is written when the whole shard is finished.
Query labels are never read.

  python run_llm.py --tag qw35_9b --mode states --roles inner,dev,test --run states
  python run_llm.py --tag qw35_9b --mode gen --uids_csv <val495.csv> --ks 0,1,3,5,7,9,10,12,15 \
      --neighbors <nb_PS_raw_dev.npz> --run val_sweep --nshards 4 --shard 0
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nm_common import LABELS, MAX_NEW, MAX_SEQ, MODELS, NM_RAW, load_pool  # noqa: E402

warnings.filterwarnings("ignore", message=".*generation flags.*")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- model
def find_decoder(model):
    """The text decoder (a module with .layers ModuleList and .norm) inside a *ForConditionalGeneration."""
    cands = []
    if hasattr(model, "get_decoder"):
        try:
            cands.append(model.get_decoder())
        except Exception:  # noqa: BLE001
            pass
    for path in ("model.language_model", "language_model", "model.text_model", "model"):
        obj = model
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                break
        cands.append(obj)
    for c in cands:
        if c is not None and isinstance(getattr(c, "layers", None), torch.nn.ModuleList) and hasattr(c, "norm"):
            return c
    raise RuntimeError("text decoder with .layers / .norm not found")


def eos_ids_for(model, tok):
    ids = set()
    g = getattr(model.generation_config, "eos_token_id", None)
    ids.update(g if isinstance(g, (list, tuple)) else ([g] if g is not None else []))
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    vocab = tok.get_vocab()
    for s in ("<|im_end|>", "<turn|>", "<end_of_turn>", "</s>", "<|endoftext|>"):
        if s in vocab:
            ids.add(vocab[s])
    return sorted(int(i) for i in ids)


def qwen_kernel_report():
    """Which gated-delta-rule implementation transformers resolved (fla vs torch fallback)."""
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as mq
        out = {}
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                     "causal_conv1d_update"):
            f = getattr(mq, name, None)
            if f is None:
                continue
            impl = inspect.getclosurevars(f).nonlocals.get("implementation", f)
            out[name] = f"{getattr(impl, '__module__', '?')}.{getattr(impl, '__name__', '?')}"
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


PARITY_TEXT = "⚠️ Bug: café नमस्ते — `x != y` ✔️ <label>bug</label>\n\n  tabs\tand  spaces"


def load_tokenizer(hf):
    """A tokenizer whose encoding matches the repo's tokenizer.json token-for-token.
    AutoTokenizer maps Qwen3.5 to Qwen2Tokenizer, which rebuilds a Qwen2 pre-tokenizer
    (\\p{L}+ instead of [\\p{L}\\p{M}]+); then fall back to the verbatim TokenizersBackend."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    ref = Tokenizer.from_file(hf_hub_download(hf, "tokenizer.json"))
    want = ref.encode(PARITY_TEXT, add_special_tokens=False).ids
    tok = AutoTokenizer.from_pretrained(hf)
    if tok.encode(PARITY_TEXT, add_special_tokens=False) != want:
        import transformers
        backend = getattr(transformers, "TokenizersBackend", None)
        if backend is None:
            from transformers.tokenization_utils_tokenizers import TokenizersBackend as backend
        tok = backend.from_pretrained(hf)
    assert tok.encode(PARITY_TEXT, add_special_tokens=False) == want, f"tokenizer parity failed: {type(tok)}"
    assert type(tok).__name__ != "MistralCommonBackend", "mistral-common backend has no jinja template"
    return tok, ref


class Runner:
    def __init__(self, tag, max_seq=MAX_SEQ):
        import transformers
        from transformers import AutoConfig, BitsAndBytesConfig
        self.tag, self.hf, self.max_seq = tag, MODELS[tag]["hf"], max_seq
        t0 = time.time()
        cfg = AutoConfig.from_pretrained(self.hf)
        cls = getattr(transformers, cfg.architectures[0])
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = cls.from_pretrained(self.hf, quantization_config=bnb, dtype=torch.bfloat16,
                                         device_map={"": 0}, attn_implementation="sdpa")
        self.model.eval()
        self.tok, self.tok_ref = load_tokenizer(self.hf)
        self.load_s = time.time() - t0
        self.dec = find_decoder(self.model)
        self.n_layers = len(self.dec.layers)
        assert self.n_layers == MODELS[tag]["n_layers"], (self.n_layers, MODELS[tag]["n_layers"])
        self.text_cfg = self.model.config.get_text_config() if hasattr(self.model.config, "get_text_config") \
            else self.model.config
        self.softcap = getattr(self.text_cfg, "final_logit_softcapping", None)
        self.eos = eos_ids_for(self.model, self.tok)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos[0]
        self.label_ids = None
        # Build the "</label>" stop criterion ONCE: generate(stop_strings=...) rebuilds it on every call
        # (a full-vocabulary scan, ~1 s per prompt).
        from transformers import StoppingCriteriaList, StopStringCriteria
        self.stopping = StoppingCriteriaList([StopStringCriteria(tokenizer=self.tok, stop_strings=["</label>"])])
        transformers.logging.set_verbosity_error()  # per-call max_length notices
        self._h, self._on = {}, False
        for i, layer in enumerate(self.dec.layers):
            layer.register_forward_hook(self._hook(i + 1))
        import bitsandbytes
        self.info = {"model": self.hf, "tag": tag, "arch": cfg.architectures[0], "cls": cls.__name__,
                     "decoder": type(self.dec).__name__, "n_layers": self.n_layers,
                     "hidden": int(self.text_cfg.hidden_size), "softcap": self.softcap, "eos": self.eos,
                     "pad": self.pad, "load_s": round(self.load_s, 1), "gpu": torch.cuda.get_device_name(0),
                     "torch": torch.__version__, "transformers": transformers.__version__,
                     "bitsandbytes": bitsandbytes.__version__, "quant": "bnb nf4, double quant, bf16 compute",
                     "attn": "sdpa", "max_seq": max_seq, "max_new_tokens": MAX_NEW,
                     "cudnn": torch.backends.cudnn.version(), "tokenizer": type(self.tok).__name__,
                     "node": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
                     "cluster": os.environ.get("SLURM_CLUSTER_NAME")}
        if "qwen3_5" in cfg.model_type:
            self.info["qwen_kernels"] = qwen_kernel_report()

    def _hook(self, l):
        def hook(_m, _inp, out):
            if self._on:
                t = out[0] if isinstance(out, (tuple, list)) else out
                self._h[l] = t[:, -1, :].detach().clone()
        return hook

    # --- prompts
    def render(self, msgs):
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False) + "<label>"

    def encode(self, prompt):
        """Tokens of the rendered prompt; hard cap as rag_next/llm_features.py (cut the end of the
        query, keep the last 60 tokens: template tail + prefill). Rare: truncation already fits
        the budget in almost every case."""
        e = self.tok.encode(prompt, add_special_tokens=False)
        if len(e) > self.max_seq:
            cut = len(e) - self.max_seq
            return e[: len(e) - 60 - cut] + e[-60:], True
        return e, False

    def set_label_ids(self, prompt):
        """First token of each label word, tokenized in context right after the prompt."""
        base = self.tok.encode(prompt, add_special_tokens=False)
        ids = []
        for lab in LABELS:
            full = self.tok.encode(prompt + lab + "</label>", add_special_tokens=False)
            if full[: len(base)] != base:
                raise RuntimeError(f"prompt is not a token prefix when followed by {lab!r}")
            ids.append(int(full[len(base)]))
        if len(set(ids)) != 3:
            raise RuntimeError(f"label first tokens are not distinct: {ids}")
        self.label_ids = ids
        self.info["label_ids"] = ids
        self.info["label_tokens"] = [self.tok.decode([i]) for i in ids]
        return ids

    # --- forward passes
    @torch.inference_mode()
    def states(self, ids):
        x = torch.tensor([ids], device="cuda")
        self._h.clear()
        self._on = True
        try:
            o = self.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
        finally:
            self._on = False
        lg = o.logits[0, -1].float()
        H = torch.stack([self._h[l][0] for l in range(1, self.n_layers + 1)])
        H[-1] = self.dec.norm(self._h[self.n_layers])[0]  # final state post-norm (what lm_head reads)
        lsm = torch.log_softmax(lg, -1)
        return H.float().cpu().numpy(), lsm[self.label_ids].cpu().numpy(), int(lg.argmax()), lg

    @torch.inference_mode()
    def gen(self, ids):
        x = torch.tensor([ids], device="cuda")
        out = self.model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                                  temperature=None, top_p=None, top_k=None, max_new_tokens=MAX_NEW,
                                  eos_token_id=self.eos, pad_token_id=self.pad, stopping_criteria=self.stopping,
                                  output_logits=True, return_dict_in_generate=True)
        g = out.sequences[0, x.shape[1]:]
        first = out.logits[0][0].float()
        lsm = torch.log_softmax(first, -1)
        text = self.tok.decode(g, skip_special_tokens=True).strip()
        return "<label>" + text, int(len(g)), lsm[self.label_ids].cpu().numpy(), int(first.argmax())


# ----------------------------------------------------------------------------- work items
def load_neighbors(path):
    z = np.load(path)
    return {int(u): [int(v) for v in row if v >= 0] for u, row in zip(z["uids"], z["nb"])}


def build_prompt(runner, by_uid, uid, k, neighbors):
    from llm_labeler import build_chat_messages
    r = by_uid.loc[uid]
    nbs = []
    if k > 0:
        for v in neighbors[uid][:k]:
            rv = by_uid.loc[v]  # a training issue: its label is a demonstration
            nbs.append({"title": rv["title"], "body": rv["body"], "label": rv["label"]})
    msgs, trunc = build_chat_messages(test_title=r["title"], test_body=r["body"], neighbors=nbs, k=k,
                                      is_thinking_model=False, max_prompt_tokens=runner.max_seq - MAX_NEW - 20,
                                      tokenizer=runner.tok)
    return runner.render(msgs), trunc, len(nbs)


def make_items(args, pool):
    if args.uids_csv:
        uids = pd.read_csv(args.uids_csv)["uid"].astype(int).tolist()
    else:
        uids = pool.loc[pool.role.isin(args.roles.split(",")), "uid"].astype(int).tolist()
    ks = [0] if args.mode == "states" else [int(k) for k in args.ks.split(",")]
    items = [(u, k) for u in uids for k in ks]
    if args.limit:
        items = items[: args.limit]
    return items


def approx_len(pool_len, neighbors, uid, k):
    n = pool_len[uid] + (sum(pool_len[v] for v in neighbors[uid][:k]) if k > 0 else 0)
    return min(n, 4 * MAX_SEQ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(MODELS))
    ap.add_argument("--mode", required=True, choices=["states", "gen"])
    ap.add_argument("--run", required=True, help="output sub-directory name, e.g. states / val_sweep / test_k0")
    ap.add_argument("--roles", default="inner,dev,test")
    ap.add_argument("--uids_csv", default=None)
    ap.add_argument("--ks", default="0")
    ap.add_argument("--neighbors", default=None)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("NM_SHARD", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("NM_NSHARDS", 1)))
    ap.add_argument("--part_size", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = NM_RAW / args.tag / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / f"done_{args.shard:02d}of{args.nshards:02d}.json"
    if done.exists():
        log(f"SKIP: {done} exists")
        return

    pool = load_pool()
    by_uid = pool.set_index("uid")
    items = make_items(args, pool)
    needs_nb = args.mode == "gen" and any(k > 0 for _, k in items)
    neighbors = load_neighbors(args.neighbors) if needs_nb else {}
    plen = {int(u): len(t) + len(b) for u, t, b in zip(pool.uid, pool.title, pool.body)}
    L = [approx_len(plen, neighbors, u, k) for u, k in items]
    order = sorted(range(len(items)), key=lambda i: (-L[i], items[i]))
    mine = [items[i] for i in order[args.shard::args.nshards]]
    parts = [mine[j:j + args.part_size] for j in range(0, len(mine), args.part_size)]
    ext = "npz" if args.mode == "states" else "parquet"
    pfile = lambda j: out_dir / f"part_{args.shard:02d}of{args.nshards:02d}_{j:03d}.{ext}"  # noqa: E731
    todo = [j for j in range(len(parts)) if not pfile(j).exists()]
    log(f"{args.tag} {args.mode} {args.run}: {len(items)} items, shard {args.shard}/{args.nshards} "
        f"-> {len(mine)} items in {len(parts)} parts, {len(todo)} to do")

    runner = Runner(args.tag)
    probe_prompt, _, _ = build_prompt(runner, by_uid, mine[0][0], 0, neighbors)
    runner.set_label_ids(probe_prompt)
    log(f"loaded in {runner.load_s:.0f}s | {json.dumps(runner.info)}")
    torch.cuda.reset_peak_memory_stats()

    for j in todo:
        t_part = time.time()
        recs, H = [], []
        for uid, k in parts[j]:
            prompt, trunc, nd = build_prompt(runner, by_uid, uid, k, neighbors)
            ids, capped = runner.encode(prompt)
            rec = {"uid": uid, "k": k, "n_demos": nd, "prompt_tokens": len(ids), "truncated": trunc.truncated,
                   "neighbors_truncated": trunc.neighbors_truncated, "query_truncated": trunc.query_truncated,
                   "tokens_removed": trunc.tokens_removed, "capped": capped}
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if args.mode == "states":
                h, lp, top1, _ = runner.states(ids)
                H.append(h)
            else:
                from llm_labeler import parse_label
                raw, n_gen, lp, top1 = runner.gen(ids)
                pred = parse_label(raw)
                has_xml = "<label>" in raw and "</label>" in raw
                rec.update(predicted_label=pred, raw_output=raw[:300], generated_tokens=n_gen,
                           parsed_via="xml" if (has_xml and pred != "invalid") else
                           ("regex" if pred != "invalid" else "failed"))
            torch.cuda.synchronize()
            rec.update(time_s=time.perf_counter() - t0, lp_bug=float(lp[0]), lp_feature=float(lp[1]),
                       lp_question=float(lp[2]), top1=top1, top1_is_label=top1 in runner.label_ids)
            recs.append(rec)
        df = pd.DataFrame(recs)
        f = pfile(j)
        tmp = f.with_name(f.name + ".tmp")
        if args.mode == "states":
            Hs = np.stack(H)
            H16 = Hs.astype(np.float16)
            Hout = H16 if np.isfinite(H16).all() else Hs  # fp16 overflow -> keep fp32 for this part
            with open(tmp, "wb") as fh:
                np.savez(fh, H=Hout, **{c: df[c].to_numpy() for c in df.columns})
        else:
            df.to_parquet(tmp, index=False)
        os.replace(tmp, f)
        log(f"part {j + 1}/{len(parts)}: {len(recs)} items, {time.time() - t_part:.0f}s wall, "
            f"compute {df.time_s.sum():.0f}s, mean prompt {df.prompt_tokens.mean():.0f} tok, "
            f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")

    meta = dict(runner.info, run=args.run, mode=args.mode, shard=args.shard, nshards=args.nshards,
                n_items=len(mine), n_parts=len(parts), roles=args.roles, uids_csv=args.uids_csv, ks=args.ks,
                neighbors=args.neighbors, gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    json.dump(meta, open(done, "w"), indent=1)
    log(f"DONE {done}")


if __name__ == "__main__":
    main()

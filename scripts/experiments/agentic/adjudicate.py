#!/usr/bin/env python3
"""Pilot arms for the agentic-IRC proposal, on SetFit's low-margin ("routed") dev issues.

Every arm ends with the same read-out: the model's log-probabilities of the three label
tokens right after an "<label>" prefix (constrained decoding: no invalid outputs), so the
arms differ only in what precedes that prefix.

  p0  paper RAGTAG prompt, k=9 raw PS neighbours (llm_labeler.build_chat_messages)
  p1  label-balanced contrastive examples (3 most similar past issues per label) + rubric
  p2  p1 + the SetFit scores of the target as a hint
  p3  p2 + a short written analysis before the label (chain of thought)
  p4  p2 + tools (agent): similar_issues / search_issues / label_stats / read_more, <= 3 calls
  p5  replay control for p4: p2 + p4's own tool outputs in one prompt, direct answer
  p6  p4 without the SetFit hint (agent starting from p1): the pure-LLM agent
  p7  replay control for p6 (p1 + p6's tool outputs, direct answer)
  p8  compute-matched bundle: p2 with 6 instead of 3 examples per label
  p9  paper BRAGTAG prompt (k=12, margin 3) at the same model size: the matched-size LLM reference

Factorial reading: p3-p2 reasoning; p5-p2 extra (agent-chosen) information;
p4-p5 the agentic process given identical information; p4-p3 tools given reasoning;
p6 vs p7 vs p1 the same questions for a classifier that never sees SetFit.

Greedy decoding (do_sample=False), repetition_penalty 1.0. Dev index = role inner.

Usage (lab machine, repo root):
  venv/bin/python scripts/experiments/agentic/adjudicate.py --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit \
      --arms p0,p1,p2,p3,p4,p5 --n_route 300 --tag q14
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))  # repo root (llm_labeler)
from common import EXP, INP, LABELS, Retriever, load_pool, load_setfit_dev, routed_uids, snapshot_inputs  # noqa: E402

LABEL_TOK = {"bug": 2313, "feature": 12753, "question": 7841}  # Qwen2.5 tokenizer, verified
MAX_TOOL_CALLS = 3


def rubric(repo: str) -> str:
    return f"""You are triaging GitHub issues for the {repo} repository. Decide which label this project's maintainers assigned to the target issue: bug, feature, or question.

Label meanings in this dataset:
- bug: {repo} itself behaves incorrectly (crash, error, wrong result, regression, broken build or documentation) and needs a fix in {repo}.
- feature: a request for something {repo} does not do yet: new functionality, an enhancement, or a change to intended behavior or design.
- question: the reporter needs help or information: how to do something, why something happens, or whether a behavior is expected. Reports of an error that stems from the reporter's own setup or usage are also labeled question.

The issue template a reporter picked (for example "Steps to reproduce", "Expected behavior", "Type: Bug") is evidence, but how much it counts differs by project: in some projects the template matches the label almost always, in others reporters often file questions or requests under the bug template. Use this project's labeled examples to see how its maintainers draw the line, and weigh what the reporter actually needs."""


def sys_direct(repo):
    return rubric(repo) + "\n\nRespond with only the label in XML tags, for example <label>question</label>."


def sys_cot(repo):
    return rubric(repo) + ("\n\nFirst write a short analysis of at most four sentences: what the reporter needs, whether the text "
                           "shows a defect in the project, and which labeled examples the target resembles most. Then give the "
                           "label in XML tags, for example <label>question</label>.")


def sys_agent(repo):
    return rubric(repo) + (f"\n\nYou can call tools that look up this project's past labeled issues, at most {MAX_TOOL_CALLS} calls "
                           "in total. Use them when the evidence is mixed, for example to see how the project labels issues "
                           "that share the target's template or topic. When you are ready, write at most three sentences of "
                           "justification and end with the label in XML tags, for example <label>question</label>.")


TOOLS = [
    {"type": "function", "function": {
        "name": "similar_issues",
        "description": "Return the k past issues of this project with the given label that are most similar to the target "
                       "(excluding examples already shown).",
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "enum": LABELS},
            "k": {"type": "integer", "description": "1 to 4"}}, "required": ["label"]}}},
    {"type": "function", "function": {
        "name": "search_issues",
        "description": "Semantic search over this project's past labeled issues with a free-text query, optionally "
                       "restricted to one label. Returns titles, excerpts and labels.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "label": {"type": "string", "enum": LABELS},
            "k": {"type": "integer", "description": "1 to 4"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "label_stats",
        "description": "Count how many of this project's past issues of each label match a case-insensitive regular "
                       "expression in their title or body, e.g. a template heading or a phrase. Shows how the project "
                       "labels issues with that feature.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "read_more",
        "description": "Return the part of the target issue's body that was omitted from the prompt because of length.",
        "parameters": {"type": "object", "properties": {}}}},
]


# ------------------------------------------------------------------ text views
class Views:
    def __init__(self, tok):
        self.tok = tok

    def ntok(self, s):
        return len(self.tok.encode(s, add_special_tokens=False))

    def head(self, s, n):
        ids = self.tok.encode(s, add_special_tokens=False)
        return s if len(ids) <= n else self.tok.decode(ids[:n]).rstrip() + " ..."

    def target(self, title, body, n_head=1200, n_tail=300):
        """Head+tail view of the target body; returns (text, omitted_middle_text)."""
        ids = self.tok.encode(body, add_special_tokens=False)
        if len(ids) <= n_head + n_tail:
            return f"Title: {title}\nBody: {body}", ""
        mid = ids[n_head:len(ids) - n_tail]
        txt = (self.tok.decode(ids[:n_head]).rstrip() + f"\n[... {len(mid)} tokens omitted ...]\n"
               + self.tok.decode(ids[len(ids) - n_tail:]).lstrip())
        return f"Title: {title}\nBody: {txt}", self.tok.decode(mid)

    def example(self, r, n=200):
        return f"Title: {r['title']}\nBody: {self.head(r['body'], n)}"


# ------------------------------------------------------------------ model wrapper
class LM:
    def __init__(self, name, max_seq_length):
        import torch
        from unsloth import FastLanguageModel
        self.torch = torch
        t0 = time.time()
        self.model, self.tok = FastLanguageModel.from_pretrained(model_name=name, max_seq_length=max_seq_length,
                                                                 dtype=None, load_in_4bit=True)
        FastLanguageModel.for_inference(self.model)
        self.load_s = time.time() - t0
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        for l, i in LABEL_TOK.items():
            assert self.tok.encode(l, add_special_tokens=False) == [i], l
        self.lab_ids = [LABEL_TOK[l] for l in LABELS]
        self.eos = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")]
        self.gen_batch = 6

    def render(self, messages, tools=None, prefix=""):
        s = self.tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        return s + prefix

    def _batches(self, enc, tok_budget, max_batch):
        order = np.argsort(-np.array([len(e) for e in enc]), kind="stable")
        i = 0
        while i < len(order):
            L = len(enc[order[i]])
            bs = max(1, min(max_batch, tok_budget // max(L, 1)))
            yield order[i:i + bs]
            i += bs

    def score(self, prompts):
        """Label log-probs at the end of each prompt (which must end with '<label>').

        One prompt per full causal-LM forward, last-position logits only. Calling Unsloth's
        base model (model.model) directly skips the causal mask that the causal-LM wrapper
        builds (its `causal_mask` argument): any row without a padding mask then attends
        bidirectionally and gets wrong logits. Padded batches are causal but drift by
        ~0.1-0.25 nats from the unbatched result. Prefill already runs near peak throughput
        at batch 1, so exactness costs little. Verified against model(...).logits[:, -1]."""
        torch = self.torch
        enc = [self.tok.encode(p, add_special_tokens=False) for p in prompts]
        out = np.zeros((len(enc), 3), dtype=np.float32)
        top1 = np.zeros(len(enc), dtype=np.int64)
        with torch.inference_mode():
            for j, e in enumerate(enc):
                lg = self.model(input_ids=torch.tensor([e]).cuda(), logits_to_keep=1).logits[0, -1].float()
                lsm = torch.log_softmax(lg, -1)
                out[j] = lsm[self.lab_ids].cpu().numpy()
                top1[j] = int(lg.argmax())
        return out, top1, np.array([len(e) for e in enc])

    def generate(self, prompts, max_new_tokens=256, tok_budget=12000, max_batch=None):
        max_batch = max_batch or self.gen_batch
        torch = self.torch
        enc = [self.tok.encode(p, add_special_tokens=False) for p in prompts]
        texts, ntoks = [None] * len(enc), [0] * len(enc)
        with torch.inference_mode():
            for idx in self._batches(enc, tok_budget, max_batch):
                batch = [enc[j] for j in idx]
                ml = max(len(b) for b in batch)
                ids = torch.full((len(batch), ml), self.tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(batch), ml), dtype=torch.long)
                for r, b in enumerate(batch):
                    ids[r, ml - len(b):] = torch.tensor(b)
                    att[r, ml - len(b):] = 1
                g = self.model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=max_new_tokens,
                                        do_sample=False, temperature=None, top_p=None, top_k=None,
                                        repetition_penalty=1.0, eos_token_id=self.eos,
                                        pad_token_id=self.tok.pad_token_id, use_cache=True)
                for r, j in enumerate(idx):
                    new = g[r, ml:].tolist()
                    cut = [k for k, t in enumerate(new) if t in self.eos]
                    new = new[:cut[0]] if cut else new
                    new = [t for t in new if t != self.tok.pad_token_id]
                    texts[j] = self.tok.decode(new, skip_special_tokens=False)
                    ntoks[j] = len(new)
        return texts, ntoks


# ------------------------------------------------------------------ arm builders
def label_prefix(text):
    """Keep the model's text before its first '<label>' and re-open the tag for scoring."""
    i = text.find("<label>")
    return (text[:i] if i >= 0 else text.rstrip() + "\n") + "<label>"


class Ctx:
    def __init__(self, pool, lm, n_route):
        self.pool = pool.set_index("uid", drop=False)
        self.lm = lm
        self.v = Views(lm.tok)
        self.R = Retriever(pool)
        sf = load_setfit_dev().set_index("uid")
        self.sf = sf
        self.uids = routed_uids(n_route)
        self.nb = np.load(INP / "nb_PS_raw_dev.npz")
        self.nbpos = {int(u): i for i, u in enumerate(self.nb["uids"])}

    def row(self, u):
        return self.pool.loc[int(u)]

    def balanced(self, u, m=3):
        ex = []
        for l in LABELS:
            ex += [(v, s, l) for v, s in self.R.similar_to_issue(int(u), l, m)]
        ex.sort(key=lambda t: -t[1])
        return ex

    def examples_block(self, ex):
        s = "Here are the most similar past issues from this project, with their labels:\n\n"
        for i, (v, _, l) in enumerate(ex, 1):
            s += f"--- Example {i} ---\n{self.v.example(self.row(v))}\nAnswer: <label>{l}</label>\n\n"
        return s

    def hint(self, u):
        r = self.sf.loc[int(u)]
        sc = ", ".join(f"{l} {r[f'p_{l}']:.2f}" for l in LABELS)
        return (f"A text classifier trained on this project's past labeled issues gives the target these scores: {sc}. "
                "The classifier is usually right, but it is less certain than usual about this issue, so treat the scores as a hint.\n\n")

    def user_msg(self, u, with_hint, extra="", m=3):
        r = self.row(u)
        tgt, _ = self.v.target(r["title"], r["body"])
        return (self.examples_block(self.balanced(u, m)) + (self.hint(u) if with_hint else "") + extra
                + "Now classify the following target issue:\n\n" + tgt)


def arm_p0(ctx, k=9, max_prompt_tokens=8122, debias_margin=None):
    from llm_labeler import _debias_neighbors, build_chat_messages
    prompts = []
    for u in ctx.uids:
        r = ctx.row(u)
        nbs = []
        for v in ctx.nb["nb"][ctx.nbpos[int(u)]][:k]:
            rv = ctx.row(v)
            nbs.append({"title": rv["title"], "body": rv["body"], "label": rv["label"]})
        if debias_margin is not None:
            nbs = _debias_neighbors(nbs, debias_margin)
        msgs, _ = build_chat_messages(test_title=r["title"], test_body=r["body"], neighbors=nbs, k=k,
                                      is_thinking_model=False, max_prompt_tokens=max_prompt_tokens, tokenizer=ctx.lm.tok)
        prompts.append(ctx.lm.render(msgs, prefix="<label>"))
    return prompts


def run_direct(ctx, prompts):
    t0 = time.time()
    lp, top1, nt = ctx.lm.score(prompts)
    return dict(logp=lp, top1=top1, prompt_tokens=nt, gen_tokens=np.zeros(len(nt), int),
                n_calls=np.ones(len(nt), int), wall=time.time() - t0)


def arm_single(ctx, with_hint, m=3):
    return [ctx.lm.render([{"role": "system", "content": sys_direct(ctx.row(u)["repo"])},
                           {"role": "user", "content": ctx.user_msg(u, with_hint, m=m)}], prefix="<label>") for u in ctx.uids]


def run_cot(ctx):
    msgs = [[{"role": "system", "content": sys_cot(ctx.row(u)["repo"])},
             {"role": "user", "content": ctx.user_msg(u, True)}] for u in ctx.uids]
    t0 = time.time()
    base = [ctx.lm.render(m) for m in msgs]
    texts, ng = ctx.lm.generate(base, max_new_tokens=256)
    final = [b + label_prefix(t) for b, t in zip(base, texts)]
    lp, top1, nt = ctx.lm.score(final)
    pt = np.array([ctx.v.ntok(b) for b in base]) + nt  # generation prefill + scoring prefill
    return dict(logp=lp, top1=top1, prompt_tokens=pt, gen_tokens=np.array(ng), n_calls=np.full(len(nt), 2),
                wall=time.time() - t0, texts=texts)


# ------------------------------------------------------------------ agent
class Tools:
    def __init__(self, ctx, u, shown):
        self.ctx, self.u, self.shown = ctx, int(u), set(shown)
        r = ctx.row(u)
        self.proj = r["proj"]
        _, self.omitted = ctx.v.target(r["title"], r["body"])

    def _fmt(self, hits, header):
        if not hits:
            return header + "\n(no matching issues)"
        s = header
        for i, (v, sim) in enumerate(hits, 1):
            rv = self.ctx.row(v)
            self.shown.add(v)
            s += f"\n{i}. [label: {rv['label']}] similarity {sim:.2f}\n{self.ctx.v.example(rv, 150)}\n"
        return s

    def call(self, name, args):
        k = int(args.get("k", 3) or 3) if isinstance(args, dict) else 3
        k = max(1, min(4, k))
        lab = args.get("label") if isinstance(args, dict) else None
        lab = lab if lab in LABELS else None
        if name == "similar_issues":
            if lab is None:
                return "error: label must be one of bug, feature, question"
            hits = [h for h in self.ctx.R.similar_to_issue(self.u, lab, k + len(self.shown)) if h[0] not in self.shown][:k]
            return self._fmt(hits, f"Past {lab} issues in this project most similar to the target:")
        if name == "search_issues":
            q = str(args.get("query", ""))[:500]
            if not q.strip():
                return "error: empty query"
            hits = [h for h in self.ctx.R.search(self.proj, q, lab, k + len(self.shown)) if h[0] not in self.shown][:k]
            return self._fmt(hits, f"Search results for '{q}'" + (f" among {lab} issues:" if lab else ":"))
        if name == "label_stats":
            pat = str(args.get("pattern", ""))[:120]
            st, err = self.ctx.R.label_stats(self.proj, pat)
            if err:
                return "error: " + err
            s = f"Past issues in this project whose title or body matches /{pat}/ (case-insensitive):"
            for l in LABELS:
                h, n, ex = st[l]
                s += f"\n- {l}: {h} of {n} ({100 * h / max(n, 1):.0f}%)" + (f"; e.g. \"{ex[0][:80]}\"" if ex else "")
            return s
        if name == "read_more":
            return self.ctx.v.head(self.omitted, 1000) if self.omitted else "Nothing was omitted; the full body is shown."
        return f"error: unknown tool {name}"


TOOL_RX = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def run_agent(ctx, with_hint=True, max_new_tokens=300):
    uids = list(ctx.uids)
    st = []
    for u in uids:
        ex = ctx.balanced(u)
        st.append(dict(u=int(u), msgs=[{"role": "system", "content": sys_agent(ctx.row(u)["repo"])},
                                       {"role": "user", "content": ctx.user_msg(u, with_hint)}],
                       tools=Tools(ctx, u, [v for v, _, _ in ex]), n_calls=0, n_tool=0, pt=0, gt=0,
                       final=None, trace=[], errors=0))
    t0 = time.time()
    for step in range(MAX_TOOL_CALLS + 2):
        active = [s for s in st if s["final"] is None]
        if not active:
            break
        force = step == MAX_TOOL_CALLS + 1 or False
        prompts = []
        for s in active:
            if s["n_tool"] >= MAX_TOOL_CALLS and s["msgs"][-1]["role"] == "tool":
                s["msgs"].append({"role": "user", "content": "Tool budget used up. Give your final answer now."})
            prompts.append(ctx.lm.render(s["msgs"], tools=TOOLS))
        texts, ng = ctx.lm.generate(prompts, max_new_tokens=max_new_tokens)
        for s, p, t, g in zip(active, prompts, texts, ng):
            s["n_calls"] += 1; s["pt"] += ctx.v.ntok(p); s["gt"] += g
            calls = TOOL_RX.findall(t)
            if calls and not force and s["n_tool"] < MAX_TOOL_CALLS and "<label>" not in t:
                s["msgs"].append({"role": "assistant", "content": t.strip()})
                for c in calls:
                    if s["n_tool"] >= MAX_TOOL_CALLS:
                        res, name, args = "error: tool budget used up", "?", {}
                    else:
                        try:
                            obj = json.loads(c)
                            name, args = obj.get("name"), obj.get("arguments", {}) or {}
                            if isinstance(args, str):
                                args = json.loads(args)
                            res = s["tools"].call(name, args)
                        except Exception as e:  # malformed call: report it back, count it
                            name, args, res = "?", {}, f"error: could not parse tool call ({e})"
                            s["errors"] += 1
                    s["n_tool"] += 1
                    s["msgs"].append({"role": "tool", "content": res})
                    s["trace"].append({"name": name, "args": args, "result": res})
            else:
                s["final"] = t
    for s in st:  # anyone still open answers without further tools
        if s["final"] is None:
            s["final"] = ""
    finals = [ctx.lm.render(s["msgs"], tools=TOOLS) + label_prefix(s["final"]) for s in st]
    lp, top1, nt = ctx.lm.score(finals)
    wall = time.time() - t0
    return dict(logp=lp, top1=top1, prompt_tokens=np.array([s["pt"] for s in st]) + nt,
                gen_tokens=np.array([s["gt"] for s in st]), n_calls=np.array([s["n_calls"] + 1 for s in st]),
                wall=wall, n_tool=np.array([s["n_tool"] for s in st]), errors=np.array([s["errors"] for s in st]),
                traces=[dict(uid=s["u"], trace=s["trace"], final=s["final"]) for s in st])


def arm_replay(ctx, traces, with_hint=True):
    by = {t["uid"]: t for t in traces}
    prompts = []
    for u in ctx.uids:
        tr = by[int(u)]["trace"]
        extra = ""
        if tr:
            extra = "Additional evidence looked up for this issue:\n\n" + "\n\n".join(x["result"] for x in tr) + "\n\n"
        prompts.append(ctx.lm.render([{"role": "system", "content": sys_direct(ctx.row(u)["repo"])},
                                      {"role": "user", "content": ctx.user_msg(u, with_hint, extra)}], prefix="<label>"))
    return prompts


# ------------------------------------------------------------------ main
def save(out, arm, ctx, res):
    lp = res["logp"]
    df = pd.DataFrame({"uid": ctx.uids, "label": [ctx.row(u)["label"] for u in ctx.uids],
                       "pred": [LABELS[i] for i in lp.argmax(1)],
                       **{f"lp_{l}": lp[:, i] for i, l in enumerate(LABELS)},
                       "top1_is_label": np.isin(res["top1"], list(LABEL_TOK.values())),
                       "prompt_tokens": res["prompt_tokens"], "gen_tokens": res["gen_tokens"], "n_calls": res["n_calls"]})
    for k in ["n_tool", "errors"]:
        if k in res:
            df[k] = res[k]
    df.to_csv(out / f"{arm}.csv", index=False)
    if "traces" in res:
        with open(out / f"{arm}_traces.jsonl", "w") as f:
            for t in res["traces"]:
                f.write(json.dumps(t) + "\n")
    if "texts" in res:
        with open(out / f"{arm}_texts.jsonl", "w") as f:
            for u, t in zip(ctx.uids, res["texts"]):
                f.write(json.dumps({"uid": int(u), "text": t}) + "\n")
    meta = dict(arm=arm, n=len(df), wall_s=round(res["wall"], 1), acc=float((df.pred == df.label).mean()),
                prompt_tokens=int(df.prompt_tokens.sum()), gen_tokens=int(df.gen_tokens.sum()),
                calls=int(df.n_calls.sum()), top1_is_label=float(df.top1_is_label.mean()),
                gpu_peak_mb=round(ctx.lm.torch.cuda.max_memory_allocated() / 2**20))
    json.dump(meta, open(out / f"{arm}.json", "w"), indent=1)
    print(json.dumps(meta), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-14B-Instruct-bnb-4bit")
    ap.add_argument("--arms", default="p0,p1,p2,p3,p4,p5,p6,p7,p8")
    ap.add_argument("--n_route", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N routed issues")
    ap.add_argument("--sample", type=int, default=None, help="random subset of N routed issues (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_seq_length", type=int, default=10240)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_root", default=str(EXP / "pilot"), help="smoke tests: a directory outside results/")
    ap.add_argument("--gen_batch", type=int, default=6, help="max sequences per generate() batch (rerun-noise checks use 1)")
    args = ap.parse_args()

    snapshot_inputs()
    out = Path(args.out_root) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    lm = LM(args.model, args.max_seq_length)
    lm.gen_batch = args.gen_batch
    ctx = Ctx(load_pool(), lm, args.n_route)
    if args.limit:
        ctx.uids = ctx.uids[: args.limit]
    if args.sample:
        keep = set(np.random.default_rng(args.seed).choice(ctx.uids, args.sample, replace=False).tolist())
        ctx.uids = np.array([u for u in ctx.uids if u in keep])
    print(f"model {args.model} loaded in {lm.load_s:.0f}s; {len(ctx.uids)} routed dev issues", flush=True)
    json.dump(dict(vars(args), model_load_s=round(lm.load_s, 1)), open(out / "run_args.json", "w"), indent=1)
    for arm in args.arms.split(","):
        if (out / f"{arm}.csv").exists():
            print("SKIP", arm)
            continue
        lm.torch.cuda.reset_peak_memory_stats()
        if arm == "p0":
            res = run_direct(ctx, arm_p0(ctx))
        elif arm == "p9":  # matched-size BRAGTAG: paper prompt, k=12, bug examples dropped at margin 3
            res = run_direct(ctx, arm_p0(ctx, k=12, debias_margin=3))
        elif arm in ("p1", "p2"):
            res = run_direct(ctx, arm_single(ctx, with_hint=arm == "p2"))
        elif arm == "p8":  # compute-matched bundle: p2 with 6 examples per label
            res = run_direct(ctx, arm_single(ctx, with_hint=True, m=6))
        elif arm == "p3":
            res = run_cot(ctx)
        elif arm in ("p4", "p6"):
            res = run_agent(ctx, with_hint=arm == "p4")
        elif arm in ("p5", "p7"):
            src = {"p5": "p4", "p7": "p6"}[arm]
            traces = [json.loads(l) for l in open(out / f"{src}_traces.jsonl")]
            res = run_direct(ctx, arm_replay(ctx, traces, with_hint=arm == "p5"))
        else:
            raise SystemExit(f"unknown arm {arm}")
        save(out, arm, ctx, res)


if __name__ == "__main__":
    main()

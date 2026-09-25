#!/usr/bin/env python3
"""Command-line versions of the pilot agent's four tools, for the frontier-agent audit
(Claude subagents acting as the agent on the 100 audit items). Standard library only.

The tools see only the project's `inner` issues (the dev phase's labelled history) through
agent_tooldata/{inner_issues.jsonl, items_tooldata.json}, exported by the lab-side step in
the proposal. Every call is logged to calls.jsonl, and each item may make at most 3 calls.

  python3 agent_tools_cli.py similar <item_id> <bug|feature|question> [k]
  python3 agent_tools_cli.py search  <item_id> "<free-text query>" [label|-] [k]
  python3 agent_tools_cli.py stats   <item_id> "<case-insensitive regex>"
  python3 agent_tools_cli.py more    <item_id>

Differences from adjudicate.Tools: `search` ranks by BM25 instead of MiniLM cosine (no
embedding model on this machine); excerpts are the first 600 characters of the body.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

DATA = Path(os.environ.get("AGENT_TOOLDATA", Path(__file__).resolve().parent / "agent_tooldata"))
LOG = Path(os.environ.get("AGENT_TOOLLOG", DATA / "calls.jsonl"))
LABELS = ["bug", "feature", "question"]
BUDGET = 3


def load():
    issues = {}
    with open(DATA / "inner_issues.jsonl") as f:
        for line in f:
            r = json.loads(line)
            issues[int(r["uid"])] = r
    items = json.load(open(DATA / "items_tooldata.json"))
    return issues, items


def past_calls(item):
    if not LOG.exists():
        return []
    return [c for c in (json.loads(l) for l in open(LOG) if l.strip()) if c["item"] == item]


def excerpt(r, n=600):
    b = r["body"] if len(r["body"]) <= n else r["body"][:n].rstrip() + " ..."
    return f"Title: {r['title']}\nBody: {b}"


def fmt(issues, hits, header):
    if not hits:
        return header + "\n(no matching issues)"
    s = header
    for i, (u, score) in enumerate(hits, 1):
        r = issues[u]
        s += f"\n{i}. [label: {r['label']}] score {score:.2f}\n{excerpt(r)}\n"
    return s


def bm25(docs, query, k1=1.2, b=0.75):
    tok = lambda t: re.findall(r"[a-z0-9_]+", t.lower())
    dt = [tok(d) for d in docs]
    avg = sum(len(d) for d in dt) / max(len(dt), 1)
    df = Counter(w for d in dt for w in set(d))
    n = len(dt)
    q = tok(query)
    out = []
    for d in dt:
        tf = Counter(d)
        s = 0.0
        for w in q:
            if tf[w]:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        out.append(s)
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    tool, item = sys.argv[1], sys.argv[2]
    args = sys.argv[3:]
    issues, items = load()
    if item not in items:
        print(f"error: unknown item {item}")
        sys.exit(1)
    it = items[item]
    prev = past_calls(item)
    if len(prev) >= BUDGET:
        print(f"error: tool budget used up ({BUDGET} calls) for {item}")
        return
    seen = set(it["shown"]) | {u for c in prev for u in c.get("returned", [])}
    proj_uids = [u for u, r in issues.items() if r["proj"] == it["proj"]]
    returned = []
    try:
        if tool == "similar":
            lab = args[0] if args else ""
            k = max(1, min(4, int(args[1]) if len(args) > 1 else 3))
            if lab not in LABELS:
                out = "error: label must be one of bug, feature, question"
            else:
                hits = [(u, s) for u, s in it["similar"][lab] if u not in seen][:k]
                returned = [u for u, _ in hits]
                out = fmt(issues, hits, f"Past {lab} issues in this project most similar to the target:")
        elif tool == "search":
            query = args[0] if args else ""
            lab = args[1] if len(args) > 1 and args[1] in LABELS else None
            k = max(1, min(4, int(args[2]) if len(args) > 2 else 3))
            cand = [u for u in proj_uids if (lab is None or issues[u]["label"] == lab) and u not in seen]
            sc = bm25([issues[u]["title"] + " " + issues[u]["body"] for u in cand], query[:500])
            hits = sorted(zip(cand, sc), key=lambda t: -t[1])[:k]
            hits = [(u, s) for u, s in hits if s > 0]
            returned = [u for u, _ in hits]
            out = fmt(issues, hits, f"Search results for '{query[:120]}'" + (f" among {lab} issues:" if lab else ":"))
        elif tool == "stats":
            pat = (args[0] if args else "")[:120]
            try:
                rx = re.compile(pat, re.IGNORECASE)
            except re.error as e:
                rx, out = None, f"error: invalid regex: {e}"
            if rx is not None:
                out = f"Past issues in this project whose title or body matches /{pat}/ (case-insensitive):"
                for lab in LABELS:
                    us = [u for u in proj_uids if issues[u]["label"] == lab]
                    hit = [u for u in us if rx.search((issues[u]["title"] + "\n" + issues[u]["body"])[:6000])]
                    ex = f'; e.g. "{issues[hit[0]]["title"][:80]}"' if hit else ""
                    out += f"\n- {lab}: {len(hit)} of {len(us)} ({100 * len(hit) / max(len(us), 1):.0f}%){ex}"
        elif tool == "more":
            out = it["omitted"] or "Nothing was omitted; the full body is shown."
        else:
            out = f"error: unknown tool {tool}"
    except Exception as e:  # malformed arguments count against the budget, like the pilot agent
        out = f"error: could not run tool ({e})"
    with open(LOG, "a") as f:
        f.write(json.dumps({"item": item, "tool": tool, "args": args, "returned": returned, "output": out,
                            "ts": time.time()}) + "\n")
    print(out)
    print(f"\n[{len(prev) + 1} of {BUDGET} tool calls used for {item}]")


if __name__ == "__main__":
    main()

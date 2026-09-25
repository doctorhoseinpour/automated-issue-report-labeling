#!/usr/bin/env python3
"""Produce TEST predictions for a configuration chosen on dev (does not evaluate).

Each component is refit in the test phase (fit on the full train split, predict
test); the stacker is fit on the dev-phase component outputs (fit on inner,
predicted on dev) and applied unchanged to the test-phase outputs. The output is a
3,300-row CSV in the paper's global test order and RAGTAG schema, so evaluate.py /
final_eval.py can score it. ground_truth is copied only to fill the schema.

Config (JSON):
  {"name": "...",
   "components": [
      {"type": "probe", "feat": "q7_k0", "layers": [18, 21, 24, 28], "C": 0.003, "scope": "AUG"},
      {"type": "tfidf", "C": 16.0},
      {"type": "label", "dev": "q7_k12b3_dev", "test": "q7_k12b3_test"},
      {"type": "knn", "setting": "PS", "variant": "center", "k": 15},
      {"type": "setfit"}],
   "stacker_C": 1.0}          # omit "stacker_C" with a single component -> its argmax
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import components as C
from common import EXP, LABELS, RES
from fusion import fit_stacker, stack_features


def comp_probs(spec, phase):
    t = spec["type"]
    if t == "probe":
        if "layers" in spec:
            return C.probe_ensemble(phase, spec["feat"], spec["layers"], spec["C"], spec.get("scope", "AUG"),
                                    kind=spec.get("kind", "h"), budget=spec.get("budget"),
                                    seed=spec.get("seed", 0))
        return C.probe(phase, spec["feat"], spec["layer"], spec["C"], spec.get("scope", "PA"),
                       kind=spec.get("kind", "h"))
    if t == "tfidf":
        return C.tfidf(phase, spec.get("C", 16.0))
    if t == "label":
        return C.label_scores(phase, spec[phase], spec.get("calibrate", "none"), spec.get("scope", "PA"))
    if t == "knn":
        return C.knn_vote(phase, spec["setting"], spec["variant"], spec["k"])
    if t == "setfit":
        return C.setfit(phase, spec.get("body", "Collab-uniba_github-issues-mpnet-st-e10"))
    raise ValueError(t)


def main():
    cfg = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2])
    test_P, timing = [], []
    for s in cfg["components"]:
        t0 = time.time()
        test_P.append(comp_probs(s, "test"))
        timing.append({"component": json.dumps(s), "fit_predict_s": round(time.time() - t0, 2)})
        print(f"  test-phase {s['type']}: {timing[-1]['fit_predict_s']} s", flush=True)
    if len(test_P) == 1 and "stacker_C" not in cfg:
        P = test_P[0]
    else:
        dev_P = [comp_probs(s, "dev") for s in cfg["components"]]
        y_dev = C.labels_of(C.query_uids("dev"))
        clf = fit_stacker(stack_features(dev_P), y_dev, cfg.get("stacker_C", 1.0))
        P = clf.predict_proba(stack_features(test_P))
        print("stacker coef:\n", np.round(clf.coef_, 3), "\nintercept", np.round(clf.intercept_, 3))
    q_uids = C.query_uids("test")
    pool = C.pool_cached().set_index("uid")
    rows = pool.loc[q_uids]
    order = np.argsort(rows["gidx"].to_numpy())
    assert (rows["gidx"].to_numpy()[order] == np.arange(3300)).all()
    pred = np.array(LABELS)[P.argmax(1)]
    df = pd.DataFrame({
        "test_idx": np.arange(3300),
        "title": rows["title"].to_numpy()[order],
        "body": rows["body"].to_numpy()[order],
        "ground_truth": rows["label"].to_numpy()[order],
        "predicted_label": pred[order],
        "raw_output": [json.dumps(dict(zip(LABELS, np.round(p, 5).tolist()))) for p in P[order]],
        "truncated": False, "neighbors_truncated": False, "query_truncated": False,
        "tokens_removed": 0, "parsed_via": "readout", "prompt_tokens": 0, "generated_tokens": 0,
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    json.dump({**cfg, "timing": timing}, open(out.with_suffix(".config.json"), "w"), indent=1)
    print(f"wrote {out} ({len(df)} rows); predicted label shares:",
          df["predicted_label"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()

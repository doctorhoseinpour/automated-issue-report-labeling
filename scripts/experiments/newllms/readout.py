#!/usr/bin/env python3
"""Decision-state read-out (R) and kNN vote (K) on the new models, FROZEN rag_next recipe (lab machine).

  R: answer-position states at depth fractions {0.625, 0.75, 0.875, 1.0}; per-layer AUG logistic
     regression (C = 0.003) on standardized states; probabilities averaged.
  K: final-layer state standardized with the index's (unlabeled) statistics, cosine, PS index,
     k = 9, similarity-weighted vote. Training-free.
Reuses rag_next/components.py and state_neighbors.py unchanged, with their FEATS pointed at
newllms/features. A regression check first reproduces rag_next's R-7B dev score (0.827) through
this path.

  dev : fit on inner (2,310), score dev (990)       -> printed + val/<tag>_readout_dev.json
  test: fit on train (3,300), predict test (3,300)  -> test_preds/{readout,stateknn}_<tag>.csv
        (predictions only; scoring is eval_test.py)
  --sensitivity: dev-only single-layer / C / scope variants and a PA depth curve (never tested)

  python readout.py qw35_9b gm4_12b mi3_8b [--sensitivity]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import nm_common as N  # noqa: F401  (puts rag_next on sys.path)
import components as C
import state_neighbors as SN
from common import LABELS, macro_f1, per_class_f1


def use_feats(path):
    C.FEATS = path
    SN.FEATS = path
    C.load_feats.cache_clear()


def rep(name, y, P):
    p = P.argmax(1)
    f = per_class_f1(y, p)
    r = {"macro_f1": macro_f1(y, p), "f1_bug": f[0], "f1_feature": f[1], "f1_question": f[2],
         "q_to_bug": float(np.mean(p[y == 2] == 0)), "pred_bug": float(np.mean(p == 0))}
    print(f"  {name:44s} F1={r['macro_f1']:.4f}  bug/feat/q={f[0]:.3f}/{f[1]:.3f}/{f[2]:.3f}  q->bug={r['q_to_bug']:.3f}",
          flush=True)
    return r


def write_test_csv(P, out, parsed_via, cfg):
    """3,300 rows in the paper's global test order and RAGTAG schema (as rag_next/final_predict.py);
    ground_truth is copied only to fill the schema."""
    q = C.query_uids("test")
    rows = C.pool_cached().set_index("uid").loc[q]
    order = np.argsort(rows["gidx"].to_numpy())
    assert (rows["gidx"].to_numpy()[order] == np.arange(3300)).all()
    pred = np.array(LABELS)[P.argmax(1)]
    pd.DataFrame({
        "test_idx": np.arange(3300), "title": rows["title"].to_numpy()[order], "body": rows["body"].to_numpy()[order],
        "ground_truth": rows["label"].to_numpy()[order], "predicted_label": pred[order],
        "raw_output": [json.dumps(dict(zip(LABELS, np.round(p, 5).tolist()))) for p in P[order]],
        "truncated": False, "neighbors_truncated": False, "query_truncated": False, "tokens_removed": 0,
        "parsed_via": parsed_via, "prompt_tokens": 0, "generated_tokens": 0,
    }).to_csv(out, index=False)
    json.dump(cfg, open(out.with_suffix(".config.json"), "w"), indent=1)
    print(f"  wrote {out.name}; predicted shares {pd.Series(pred).value_counts(normalize=True).round(3).to_dict()}")


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("--")]
    sens = "--sensitivity" in sys.argv
    out_val = N.NM / "val"
    out_test = N.NM / "test_preds"
    out_val.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    use_feats(N.RN_FEATS)
    yd = C.labels_of(C.query_uids("dev"))
    f7 = macro_f1(yd, C.probe_ensemble("dev", "q7_k0", [18, 21, 24, 28], 0.003, "AUG").argmax(1))
    print(f"regression check: rag_next R-7B dev through this path = {f7:.4f} (notebook 0.827)")
    assert abs(f7 - 0.827) < 0.0015, f7
    use_feats(N.NM_FEATS)

    for tag in tags:
        n = N.MODELS[tag]["n_layers"]
        layers = N.readout_layers(n)
        feat, vtag = f"{tag}_k0", f"{tag}L{n}"
        print(f"\n=== {tag} ({N.MODELS[tag]['name']}): read-out layers {layers} of {n}")
        sys_argv = sys.argv
        sys.argv = ["state_neighbors.py", feat, str(n), vtag]
        SN.main()  # label-free decision-state index (PS/PA, dev and test phases)
        sys.argv = sys_argv
        dev = {"layers": layers,
               "own_label": rep("own label, constrained (zero-shot)", yd, C.label_scores("dev", feat)),
               "readout": rep("R: read-out (frozen recipe)", yd, C.probe_ensemble("dev", feat, layers, 0.003, "AUG")),
               "knn": rep("K: decision-state kNN@9 PS (training-free)", yd, C.knn_vote("dev", "PS", vtag, 9))}
        json.dump(dev, open(out_val / f"{tag}_readout_dev.json", "w"), indent=1)

        t0 = time.time()
        Pt = C.probe_ensemble("test", feat, layers, 0.003, "AUG")
        t_r = time.time() - t0
        write_test_csv(Pt, out_test / f"readout_{tag}.csv", "readout",
                       {"name": f"readout_{tag}", "components": [{"type": "probe", "feat": feat, "layers": layers,
                                                                  "C": 0.003, "scope": "AUG"}],
                        "fit_predict_s": round(t_r, 1)})
        t0 = time.time()
        Kt = C.knn_vote("test", "PS", vtag, 9)
        t_k = time.time() - t0
        write_test_csv(Kt, out_test / f"stateknn_{tag}.csv", "stateknn",
                       {"name": f"stateknn_{tag}", "components": [{"type": "knn", "setting": "PS", "variant": vtag,
                                                                   "k": 9}], "fit_predict_s": round(t_k, 2)})
        print(f"  test-phase fit+predict: read-out {t_r:.0f}s, kNN {t_k:.2f}s (CPU)")

        if sens:  # dev only, descriptive; never evaluated on test
            s = {}
            for l in layers:
                s[f"AUG_L{l}"] = rep(f"  single layer {l}, AUG, C=0.003", yd, C.probe("dev", feat, l, 0.003, "AUG"))
            for c in (0.001, 0.01):
                s[f"AUG_C{c}"] = rep(f"  ensemble, AUG, C={c}", yd, C.probe_ensemble("dev", feat, layers, c, "AUG"))
            s["PA_C0.003"] = rep("  ensemble, PA, C=0.003", yd, C.probe_ensemble("dev", feat, layers, 0.003, "PA"))
            depth = {}
            for l in sorted(set(list(range(max(1, n // 8), n + 1, max(1, n // 8))) + [n])):
                depth[l] = rep(f"  depth curve: layer {l}, PA", yd, C.probe("dev", f"{tag}_k0_all", l, 0.003, "PA"))["macro_f1"]
            s["depth_PA"] = depth
            json.dump(s, open(out_val / f"{tag}_readout_sensitivity.json", "w"), indent=1)


if __name__ == "__main__":
    main()

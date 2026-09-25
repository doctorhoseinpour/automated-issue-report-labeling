#!/usr/bin/env python3
"""Score the rag_next decision-state read-outs (R-14B / R-32B, dev phase: fit on inner) on the
agentic pilot's routed dev slice and on the 100 audit items. Read-only use of rag_next code/features.
Usage (lab machine): venv/bin/python scripts/experiments/agentic/readout_on_routed.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "rag_next"))
import components as C  # noqa: E402

import importlib.util  # noqa: E402

# both study directories have a module called `common`; load this study's under its own name
_spec = importlib.util.spec_from_file_location("agentic_common", Path(__file__).resolve().parent / "common.py")
_ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ac)
EXP, INP, routed_uids = _ac.EXP, _ac.INP, _ac.routed_uids

L = np.array(["bug", "feature", "question"])
R = set(int(x) for x in routed_uids(300))
key = pd.read_csv(EXP / "audit" / "key.csv")
for size in ["14B", "32B"]:
    cfg = json.load(open(REPO / "scripts/experiments/rag_next/configs" / f"readout_{size}.json"))["components"][0]
    try:
        P = C.probe_ensemble("dev", cfg["feat"], cfg["layers"], cfg["C"], cfg["scope"])
    except Exception as e:  # features for this size may not exist on this machine
        print(size, "unavailable:", str(e)[:160])
        continue
    u = C.query_uids("dev")
    y = L[C.labels_of(u)]
    pred = L[P.argmax(1)]
    out = pd.DataFrame({"uid": u, "label": y, "pred": pred, "p_bug": P[:, 0], "p_feature": P[:, 1], "p_question": P[:, 2]})
    out.to_csv(INP / f"readout{size}_dev.csv", index=False)
    m = out.uid.isin(R).to_numpy()
    k = out.set_index("uid").loc[key.uid]
    f_all = f1_score(y, pred, labels=list(L), average="macro")
    f_r = f1_score(y[m], pred[m], labels=list(L), average="macro")
    q2b = np.mean(pred[m & (y == "question")] == "bug")
    print(f"R-{size} dev macroF1 {f_all:.4f} | SetFit-routed 300: acc {np.mean(pred[m] == y[m]):.3f} "
          f"macroF1 {f_r:.3f} q2bug {q2b:.3f} | outside: acc {np.mean(pred[~m] == y[~m]):.3f} | "
          f"audit-100 acc {np.mean(k.pred.values == key.label.values):.3f}")

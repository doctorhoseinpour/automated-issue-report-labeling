#!/usr/bin/env python3
"""Cross-hardware consistency: 14B features for the ansible dev issues extracted on the OSC
H100 (xcheck_q14_ansible_dev) vs the same issues from the lab RTX 4090 run (q14_k0).
Compares label log-probs, hidden-state cosine, and the default read-out's predictions
(head fit on the 4090 inner states, applied to both versions of the dev states)."""
import numpy as np

import components as C

a = C.load_feats("xcheck_q14_ansible_dev")
b = C.load_feats("q14_k0")
rows = C._rows(b, a["uids"])
d = np.abs(a["logp"] - b["logp"][rows])
print(f"label log-prob |diff|: median {np.median(d):.4f}  p95 {np.percentile(d, 95):.4f}  max {d.max():.4f}")
print("own-label argmax agreement:", np.mean(a["logp"].argmax(1) == b["logp"][rows].argmax(1)))
for l in [30, 36, 42, 48]:
    x, y = a[f"h_L{l}"].astype(np.float32), b[f"h_L{l}"][rows].astype(np.float32)
    cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    print(f"L{l}: hidden-state cosine min {cos.min():.5f} median {np.median(cos):.5f}")
# read-out: fit on 4090 inner (PA, per layer, C=0.003, averaged), apply to both
pool = C.pool_cached()
tr = pool[pool.role == "inner"].uid.to_numpy()
rt = C._rows(b, tr); ytr = C.labels_of(tr)
P_lab = np.mean([C._lr(b[f"h_L{l}"][rt].astype(np.float32), ytr, b[f"h_L{l}"][rows].astype(np.float32), 0.003)
                 for l in [30, 36, 42, 48]], 0)
P_osc = np.mean([C._lr(b[f"h_L{l}"][rt].astype(np.float32), ytr, a[f"h_L{l}"].astype(np.float32), 0.003)
                 for l in [30, 36, 42, 48]], 0)
print("read-out prediction agreement (4090 vs H100 dev states):", np.mean(P_lab.argmax(1) == P_osc.argmax(1)),
      " max |prob diff|:", np.abs(P_lab - P_osc).max().round(4))

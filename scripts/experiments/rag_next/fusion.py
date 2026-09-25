"""Log-linear fusion (stacking) of component probabilities.

Stacker = multinomial LR on the clipped log-probabilities of the selected components
(3 features per component, optional per-component scaling learned). Dev estimate =
repeated stratified K-fold cross-fitting *inside dev* (stratified by project x label);
final = stacker fit on all dev rows, applied to test-phase component outputs.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from common import macro_f1


def logp(P):
    return np.log(np.clip(P, 1e-4, 1.0))


def stack_features(comps):
    return np.concatenate([logp(P) for P in comps], 1)


def fit_stacker(X, y, C=1.0):
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(X, y)
    return clf


def cv_stack(comps, y, strata, C=1.0, n_splits=5, repeats=5, seed=0):
    """Mean macro F1 of the stacker under cross-fitting within dev; also returns the
    out-of-fold predictions of the first repeat."""
    X = stack_features(comps)
    scores, oof0 = [], None
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        oof = np.zeros(len(y), int)
        for a, b in skf.split(X, strata):
            oof[b] = fit_stacker(X[a], y[a], C).predict(X[b])
        scores.append(macro_f1(y, oof))
        if oof0 is None:
            oof0 = oof
    return float(np.mean(scores)), float(np.std(scores)), oof0


def product_of_experts(comps, weights=None):
    """Untrained fusion: weighted sum of log-probs (weights default 1)."""
    weights = weights or [1.0] * len(comps)
    S = sum(w * logp(P) for w, P in zip(weights, comps))
    return S.argmax(1)

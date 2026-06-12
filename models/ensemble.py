"""Ensemble of calibrated classifiers with validation-optimised weights.

Weights are parameterised through a softmax so the optimiser can run
unconstrained; the objective is multi-class log loss on the validation set.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss

_EPS = 1e-12


def optimize_weights(probas: dict[str, np.ndarray], y: np.ndarray,
                     seed: int = 42) -> dict[str, float]:
    """Find convex combination of member probabilities minimising log loss."""
    names = sorted(probas)
    stack = np.stack([probas[n] for n in names])      # (M, N, 3)
    y = np.asarray(y, dtype=int)

    def loss(theta: np.ndarray) -> float:
        w = np.exp(theta - theta.max())
        w /= w.sum()
        blend = np.tensordot(w, stack, axes=1)
        return log_loss(y, np.clip(blend, _EPS, 1.0), labels=[0, 1, 2])

    rng = np.random.default_rng(seed)
    best_w, best_val = None, np.inf
    # multi-start to dodge local minima in the tiny weight space
    starts = [np.zeros(len(names))] + [rng.normal(0, 1, len(names)) for _ in range(7)]
    for theta0 in starts:
        res = minimize(loss, theta0, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-7})
        if res.fun < best_val:
            best_val = res.fun
            w = np.exp(res.x - res.x.max())
            best_w = w / w.sum()
    return {n: float(w) for n, w in zip(names, best_w)}


def blend(probas: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.values())
    out = None
    for name, w in weights.items():
        p = probas[name] * (w / total)
        out = p if out is None else out + p
    return out

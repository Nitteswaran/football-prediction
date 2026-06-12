"""Probability calibration via temperature scaling.

Fitted on validation probabilities only; applied to any later predictions.
Temperature scaling preserves the argmax while fixing over/under-confidence,
which is what matters for log loss, Brier score and betting-style use.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-12


def _logits(proba: np.ndarray) -> np.ndarray:
    return np.log(np.clip(proba, _EPS, 1.0))


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    z = _logits(proba) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """Find T minimising NLL of temperature-scaled probabilities."""
    y = np.asarray(y, dtype=int)

    def nll(t: float) -> float:
        p = apply_temperature(proba, t)
        return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], _EPS, 1.0)))

    res = minimize_scalar(nll, bounds=(0.25, 4.0), method="bounded")
    return float(res.x)


class TemperatureScaler:
    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        self.temperature = fit_temperature(proba, y)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return apply_temperature(proba, self.temperature)

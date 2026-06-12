"""Evaluation metrics for classification, regression and calibration."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, log_loss, mean_absolute_error,
                             mean_squared_error, roc_auc_score)

_EPS = 1e-12


def brier_multiclass(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    """Mean squared error between one-hot outcomes and predicted probabilities."""
    y = np.asarray(y_true, dtype=int)
    onehot = np.eye(n_classes)[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """RPS for ordered outcomes (home win < draw < away win) — the standard
    proper score in football forecasting."""
    y = np.asarray(y_true, dtype=int)
    onehot = np.eye(proba.shape[1])[y]
    cum_p = np.cumsum(proba, axis=1)
    cum_o = np.cumsum(onehot, axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (proba.shape[1] - 1)))


def classification_report(y_true, proba) -> dict[str, float]:
    proba = np.clip(np.asarray(proba, dtype=float), _EPS, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    y = np.asarray(y_true, dtype=int)
    out = {
        "accuracy": float(accuracy_score(y, proba.argmax(axis=1))),
        "log_loss": float(log_loss(y, proba, labels=[0, 1, 2])),
        "brier": brier_multiclass(y, proba),
        "rps": ranked_probability_score(y, proba),
    }
    try:
        out["roc_auc_ovr"] = float(roc_auc_score(y, proba, multi_class="ovr"))
    except ValueError:
        out["roc_auc_ovr"] = float("nan")
    return out


def regression_report(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
    }


def reliability_curve(y_true, proba, class_idx: int, n_bins: int = 10):
    """(bin_centers, observed_freq, predicted_mean, counts) for one class."""
    y = (np.asarray(y_true, dtype=int) == class_idx).astype(float)
    p = np.asarray(proba, dtype=float)[:, class_idx]
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    centers, obs, pred, counts = [], [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() < 10:
            continue
        centers.append((edges[b] + edges[b + 1]) / 2)
        obs.append(float(y[mask].mean()))
        pred.append(float(p[mask].mean()))
        counts.append(int(mask.sum()))
    return centers, obs, pred, counts


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Mean |observed - predicted| across classes weighted by bin mass."""
    total, weight = 0.0, 0
    for c in range(proba.shape[1]):
        _, obs, pred, counts = reliability_curve(y_true, proba, c, n_bins)
        for o, p, n in zip(obs, pred, counts):
            total += abs(o - p) * n
            weight += n
    return total / max(weight, 1)

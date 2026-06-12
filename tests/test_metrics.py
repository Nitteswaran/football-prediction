import numpy as np

from evaluation.metrics import (brier_multiclass, classification_report,
                                ranked_probability_score)
from models.calibration import TemperatureScaler, apply_temperature


def test_brier_perfect_and_uniform():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert brier_multiclass(y, perfect) == 0.0
    uniform = np.full((3, 3), 1 / 3)
    assert np.isclose(brier_multiclass(y, uniform), 2 / 3)


def test_rps_orders_better_forecasts():
    y = np.array([0])
    near = np.array([[0.7, 0.2, 0.1]])
    far = np.array([[0.7, 0.1, 0.2]])   # same mass on the wrong side but further
    assert ranked_probability_score(y, near) < ranked_probability_score(y, far)


def test_classification_report_keys():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 500)
    p = rng.dirichlet([2, 1, 2], 500)
    rep = classification_report(y, p)
    for key in ("accuracy", "log_loss", "brier", "rps", "roc_auc_ovr"):
        assert key in rep


def test_temperature_scaling_improves_overconfident():
    rng = np.random.default_rng(2)
    n = 4000
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 1.0          # informative signal
    over = np.exp(logits * 3)               # overconfident probabilities
    over /= over.sum(1, keepdims=True)

    sc = TemperatureScaler().fit(over, y)
    assert sc.temperature > 1.5             # detects overconfidence
    from sklearn.metrics import log_loss
    assert log_loss(y, sc.transform(over)) < log_loss(y, over)


def test_apply_temperature_preserves_argmax():
    p = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    for t in (0.5, 2.0):
        assert (apply_temperature(p, t).argmax(1) == p.argmax(1)).all()

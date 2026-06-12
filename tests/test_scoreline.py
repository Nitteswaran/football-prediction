import numpy as np

from models.scoreline import (fit_rho, outcome_probs_from_grid, score_grid,
                              top_scorelines)


def test_grid_sums_to_one():
    grid = score_grid(1.5, 1.1, rho=-0.05)
    assert np.isclose(grid.sum(), 1.0)
    assert (grid >= 0).all()


def test_outcome_probs_consistent():
    grid = score_grid(2.0, 0.8, rho=0.0)
    h, d, a = outcome_probs_from_grid(grid)
    assert np.isclose(h + d + a, 1.0)
    assert h > a            # stronger home attack
    assert 0 < d < 0.5


def test_top_scorelines_sorted():
    grid = score_grid(1.3, 1.0)
    tops = top_scorelines(grid, k=5)
    probs = [p for _, p in tops]
    assert probs == sorted(probs, reverse=True)
    assert len(tops) == 5


def test_fit_rho_recovers_sign():
    rng = np.random.default_rng(0)
    n = 4000
    lam_h = np.full(n, 1.4)
    lam_a = np.full(n, 1.1)
    # Generate from independent Poissons -> rho should be near 0.
    gh = rng.poisson(lam_h)
    ga = rng.poisson(lam_a)
    rho = fit_rho(lam_h, lam_a, gh, ga)
    assert abs(rho) < 0.06

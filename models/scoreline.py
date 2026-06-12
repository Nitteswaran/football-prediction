"""Exact-score model: Dixon-Coles adjusted double Poisson.

Goal expectancies (lambda_home, lambda_away) come from the goal-regressor
ensemble; this module turns them into a full scoreline probability grid with
the Dixon-Coles low-score dependence correction, whose rho parameter is
estimated by maximum likelihood on historical matches.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

import config

_EPS = 1e-12


def _tau(i: np.ndarray, j: np.ndarray, lh: np.ndarray, la: np.ndarray,
         rho: float) -> np.ndarray:
    """Dixon-Coles correction factor for the four low-score cells."""
    t = np.ones(np.broadcast(i, j).shape)
    t = np.where((i == 0) & (j == 0), 1.0 - lh * la * rho, t)
    t = np.where((i == 0) & (j == 1), 1.0 + lh * rho, t)
    t = np.where((i == 1) & (j == 0), 1.0 + la * rho, t)
    t = np.where((i == 1) & (j == 1), 1.0 - rho, t)
    return np.clip(t, _EPS, None)


def fit_rho(lambda_h: np.ndarray, lambda_a: np.ndarray,
            goals_h: np.ndarray, goals_a: np.ndarray) -> float:
    """MLE of the Dixon-Coles rho on observed scores."""
    lh = np.clip(np.asarray(lambda_h, float), 0.05, 8.0)
    la = np.clip(np.asarray(lambda_a, float), 0.05, 8.0)
    gh = np.asarray(goals_h, int)
    ga = np.asarray(goals_a, int)
    base = poisson.logpmf(gh, lh) + poisson.logpmf(ga, la)

    def nll(rho: float) -> float:
        t = _tau(gh, ga, lh, la, rho)
        return -np.mean(base + np.log(t))

    res = minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded")
    return float(res.x)


def score_grid(lambda_h: float, lambda_a: float, rho: float = 0.0,
               max_goals: int = config.MAX_GOALS_GRID) -> np.ndarray:
    """(max_goals+1, max_goals+1) matrix of P(home=i, away=j), summing to 1."""
    lh = float(np.clip(lambda_h, 0.05, 8.0))
    la = float(np.clip(lambda_a, 0.05, 8.0))
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals, lh)
    pa = poisson.pmf(goals, la)
    grid = np.outer(ph, pa)
    ii, jj = np.meshgrid(goals, goals, indexing="ij")
    grid *= _tau(ii, jj, np.full_like(ii, lh, dtype=float),
                 np.full_like(jj, la, dtype=float), rho)
    grid /= grid.sum()
    return grid


def outcome_probs_from_grid(grid: np.ndarray) -> tuple[float, float, float]:
    """(P_home_win, P_draw, P_away_win) implied by a scoreline grid."""
    home = float(np.tril(grid, -1).sum())   # i > j
    draw = float(np.trace(grid))
    away = float(np.triu(grid, 1).sum())    # j > i
    return home, draw, away


def top_scorelines(grid: np.ndarray, k: int = 8) -> list[tuple[str, float]]:
    flat = [((i, j), grid[i, j]) for i in range(grid.shape[0])
            for j in range(grid.shape[1])]
    flat.sort(key=lambda kv: kv[1], reverse=True)
    return [(f"{i}-{j}", float(p)) for (i, j), p in flat[:k]]

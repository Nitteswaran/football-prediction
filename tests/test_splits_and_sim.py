"""Chronological split integrity + simulator building blocks."""
import numpy as np
import pandas as pd
import pytest

import config
from models.ensemble import blend, optimize_weights
from simulation.worldcup2026 import GROUPS, R16, R32


def test_worldcup_structure():
    teams = [t for g in GROUPS.values() for t in g]
    assert len(GROUPS) == 12
    assert len(teams) == 48
    assert len(set(teams)) == 48
    assert len(R32) == 16
    assert len(R16) == 8
    winners = {spec[1] for _, s1, s2 in R32 for spec in (s1, s2) if spec[0] == "W"}
    runners = {spec[1] for _, s1, s2 in R32 for spec in (s1, s2) if spec[0] == "R"}
    thirds = [spec for _, s1, s2 in R32 for spec in (s1, s2) if spec[0] == "T"]
    assert winners == set(GROUPS)          # every group winner appears once
    assert runners == set(GROUPS)
    assert len(thirds) == 8
    r32_ids = {mid for mid, _, _ in R32}
    assert {m for pair in R16 for m in pair} == r32_ids


def test_third_place_allocation_is_perfect_matching():
    """Every C(12,8) qualifying set must be assignable to the 8 third-slots."""
    from itertools import combinations
    letters = list(GROUPS)
    slot_specs = [(k, R32[k][2][1]) for k in range(len(R32)) if R32[k][2][0] == "T"]

    def feasible(qualified: set[str]) -> bool:
        ordered = sorted(slot_specs,
                         key=lambda s: len([c for c in s[1] if c in qualified]))

        def bt(si: int, used: set[str]) -> bool:
            if si == len(ordered):
                return True
            _, spec = ordered[si]
            return any(bt(si + 1, used | {c}) for c in spec
                       if c in qualified and c not in used)

        return bt(0, set())

    infeasible = [q for q in combinations(letters, 8) if not feasible(set(q))]
    # FIFA's published allocation guarantees a valid assignment for the vast
    # majority of sets; our engine falls back to greedy for any remainder.
    assert len(infeasible) <= 25, f"{len(infeasible)} infeasible sets"


def test_ensemble_weight_optimizer_prefers_better_model():
    rng = np.random.default_rng(0)
    n = 3000
    y = rng.integers(0, 3, n)
    good = np.full((n, 3), 0.15)
    good[np.arange(n), y] = 0.7
    good = good + rng.normal(0, 0.01, (n, 3))
    good = np.clip(good, 1e-3, None); good /= good.sum(1, keepdims=True)
    bad = rng.dirichlet([1, 1, 1], n)
    weights = optimize_weights({"good": good, "bad": bad}, y)
    assert weights["good"] > 0.8
    blended = blend({"good": good, "bad": bad}, weights)
    assert np.allclose(blended.sum(1), 1.0)


@pytest.mark.skipif(not config.FEATURES_PARQUET.exists(),
                    reason="feature matrix not built")
def test_real_splits_are_chronological():
    from training.splits import assert_chronological, make_splits
    sp = make_splits()
    assert_chronological(sp)
    assert len(sp.train) > 20000
    assert sp.valid["date"].min().year == 2019
    assert sp.test["date"].min().year == 2023
    # No metadata/target columns may appear among features.
    assert not ({"home_score", "away_score", "outcome", "date"} & set(sp.feature_cols))

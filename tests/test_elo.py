import math

from features.elo import EloEngine, expected_score, goal_multiplier


def test_expected_score_symmetry():
    assert math.isclose(expected_score(1500, 1500), 0.5)
    assert math.isclose(expected_score(1700, 1500) + expected_score(1500, 1700), 1.0)
    assert expected_score(1700, 1500) > 0.7


def test_goal_multiplier():
    assert goal_multiplier(0) == 1.0
    assert goal_multiplier(1) == 1.0
    assert goal_multiplier(2) == 1.5
    assert goal_multiplier(3) == (11 + 3) / 8
    assert goal_multiplier(-4) == (11 + 4) / 8


def test_update_zero_sum():
    elo = EloEngine()
    elo.update("A", "B", 3, 1, importance=5, neutral=True)
    assert math.isclose(elo.get("A") + elo.get("B"), 2 * elo.base)
    assert elo.get("A") > elo.base > elo.get("B")


def test_home_advantage_only_when_not_neutral():
    elo = EloEngine()
    assert elo.expectation("A", "B", neutral=True) == 0.5
    assert elo.expectation("A", "B", neutral=False) > 0.5


def test_upset_moves_more_points():
    elo = EloEngine()
    elo.ratings = {"Big": 1900.0, "Small": 1400.0}
    pre = elo.get("Small")
    elo.update("Small", "Big", 2, 0, importance=5, neutral=True)
    upset_gain = elo.get("Small") - pre

    elo2 = EloEngine()
    elo2.ratings = {"Big": 1900.0, "Small": 1400.0}
    pre2 = elo2.get("Big")
    elo2.update("Big", "Small", 2, 0, importance=5, neutral=True)
    expected_gain = elo2.get("Big") - pre2
    assert upset_gain > expected_gain > 0

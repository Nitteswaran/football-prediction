"""Elo rating engine for international teams.

Follows the eloratings.net convention: K-factor scaled by tournament
importance, goal-difference multiplier, and a fixed home advantage added to
the home side's rating when the match is not on neutral ground.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config


def expected_score(rating_a: float, rating_b: float) -> float:
    """Win expectancy of side A against side B (draws count half)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / config.ELO_SPREAD))


def goal_multiplier(goal_diff: int) -> float:
    d = abs(goal_diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0


@dataclass
class EloEngine:
    base: float = config.ELO_BASE
    home_advantage: float = config.ELO_HOME_ADVANTAGE
    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.base)

    def expectation(self, home: str, away: str, neutral: bool) -> float:
        """Pre-match win expectancy for the home side."""
        adv = 0.0 if neutral else self.home_advantage
        return expected_score(self.get(home) + adv, self.get(away))

    def update(self, home: str, away: str, home_score: int, away_score: int,
               importance: int, neutral: bool) -> tuple[float, float]:
        """Apply one match result; returns the post-match ratings."""
        exp_home = self.expectation(home, away, neutral)
        if home_score > away_score:
            actual = 1.0
        elif home_score == away_score:
            actual = 0.5
        else:
            actual = 0.0
        k = config.ELO_K_BY_IMPORTANCE.get(importance, 30.0)
        delta = k * goal_multiplier(home_score - away_score) * (actual - exp_home)
        new_home = self.get(home) + delta
        new_away = self.get(away) - delta
        self.ratings[home] = new_home
        self.ratings[away] = new_away
        return new_home, new_away

    def table(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda kv: kv[1], reverse=True)

"""Live prediction service.

Combines the trained model bundle with the end-of-data state snapshot of the
feature builder, so fixtures that haven't happened yet get features computed
exactly the way training rows were (same code path -> no train/serve skew).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import config
from features.builder import FeatureBuilder
from models.ensemble import blend
from models.scoreline import outcome_probs_from_grid, score_grid, top_scorelines

logger = logging.getLogger(__name__)


@dataclass
class MatchPrediction:
    home: str
    away: str
    probs: dict[str, float]              # home_win / draw / away_win
    expected_goals: dict[str, float]     # home / away
    grid: np.ndarray                     # scoreline probabilities
    top_scores: list[tuple[str, float]]
    drivers: dict[str, float]            # human-readable explanatory factors
    model_probs: dict[str, list[float]]  # per-member calibrated probabilities

    def as_dict(self) -> dict:
        return {
            "home_team": self.home,
            "away_team": self.away,
            "probabilities": self.probs,
            "expected_goals": self.expected_goals,
            "top_scorelines": [{"score": s, "probability": p} for s, p in self.top_scores],
            "scoreline_grid": self.grid.tolist(),
            "drivers": self.drivers,
            "model_probabilities": self.model_probs,
        }


class Predictor:
    def __init__(self,
                 bundle_path=config.ARTIFACTS_DIR / "model_bundle.joblib",
                 snapshot_path=config.STATE_SNAPSHOT):
        self.bundle = joblib.load(bundle_path)
        self.builder: FeatureBuilder = FeatureBuilder.load(snapshot_path)
        self.feature_cols: list[str] = self.bundle["feature_cols"]

    # ------------------------------------------------------------------
    def teams(self) -> list[str]:
        return sorted(t for t, st in self.builder.teams.items()
                      if st.matches_played >= config.MIN_TEAM_HISTORY)

    def elo_table(self, top: int = 100) -> list[dict]:
        table = self.builder.elo.table()[:top]
        return [{"rank": i + 1, "team": t, "elo": round(r, 1),
                 "matches": self.builder.teams[t].matches_played}
                for i, (t, r) in enumerate(table)]

    # ------------------------------------------------------------------
    def _feature_frame(self, home: str, away: str, date: datetime,
                       tournament: str, neutral: bool, country: str) -> pd.DataFrame:
        feats = self.builder.extract(home, away, date, tournament, neutral, country)
        row = {c: feats.get(c, np.nan) for c in self.feature_cols}
        return pd.DataFrame([row], columns=self.feature_cols)

    def predict(self, home: str, away: str, *, neutral: bool = True,
                tournament: str = config.DEFAULT_TOURNAMENT,
                date: datetime | None = None, country: str = "") -> MatchPrediction:
        date = date or datetime.now()
        X = self._feature_frame(home, away, date, tournament, neutral, country)

        member_probas, model_probs = {}, {}
        for name in config.ENSEMBLE_MEMBERS:
            raw = self.bundle["classifiers"][name].predict_proba(X)
            cal = self.bundle["scalers"][name].transform(raw)
            member_probas[name] = cal
            model_probs[name] = [round(float(p), 4) for p in cal[0]]
        ens = blend(member_probas, self.bundle["ensemble_weights"])[0]

        gnames = self.bundle["goal_model_names"]
        lam_h = float(np.mean([self.bundle["goal_models"][f"home_{n}"].predict(X)[0]
                               for n in gnames]))
        lam_a = float(np.mean([self.bundle["goal_models"][f"away_{n}"].predict(X)[0]
                               for n in gnames]))
        lam_h, lam_a = float(np.clip(lam_h, 0.05, 8.0)), float(np.clip(lam_a, 0.05, 8.0))

        grid = score_grid(lam_h, lam_a, self.bundle["rho"])
        # Reconcile the scoreline grid with the (better-calibrated) classifier
        # ensemble: rescale grid cells so implied W/D/L matches the ensemble.
        gh, gd, ga = outcome_probs_from_grid(grid)
        scale = np.ones_like(grid)
        ii, jj = np.meshgrid(range(grid.shape[0]), range(grid.shape[1]), indexing="ij")
        scale[ii > jj] = ens[0] / max(gh, 1e-9)
        scale[ii == jj] = ens[1] / max(gd, 1e-9)
        scale[ii < jj] = ens[2] / max(ga, 1e-9)
        grid = grid * scale
        grid /= grid.sum()

        feats = X.iloc[0]
        drivers = {
            "elo_home": round(float(feats["home_elo"]), 1),
            "elo_away": round(float(feats["away_elo"]), 1),
            "elo_delta": round(float(feats["elo_delta"]), 1),
            "elo_expectation_home": round(float(feats["elo_expectation_home"]), 3),
            "form10_ppg_home": _r(feats.get("home_form10_ppg")),
            "form10_ppg_away": _r(feats.get("away_form10_ppg")),
            "momentum_delta": _r(feats.get("momentum_points_delta")),
            "h2h5_weighted_score": _r(feats.get("h2h5_weighted_score")),
            "rest_delta_days": _r(feats.get("rest_delta")),
        }
        return MatchPrediction(
            home=home, away=away,
            probs={"home_win": round(float(ens[0]), 4),
                   "draw": round(float(ens[1]), 4),
                   "away_win": round(float(ens[2]), 4)},
            expected_goals={"home": round(lam_h, 3), "away": round(lam_a, 3)},
            grid=grid, top_scores=top_scorelines(grid),
            drivers=drivers, model_probs=model_probs,
        )

    def batch_predict(self, fixtures: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised prediction for many fixtures.

        Each fixture dict: {home, away, neutral, tournament, date, country}.
        Returns (probs (n,3) calibrated ensemble, lambda_home (n,), lambda_away (n,)).
        State is NOT updated between fixtures — they are all evaluated from the
        current snapshot, which is exactly what tournament simulation needs.
        """
        rows = []
        for fx in fixtures:
            feats = self.builder.extract(
                fx["home"], fx["away"], fx.get("date") or datetime.now(),
                fx.get("tournament", config.DEFAULT_TOURNAMENT),
                bool(fx.get("neutral", True)), fx.get("country", ""))
            rows.append({c: feats.get(c, np.nan) for c in self.feature_cols})
        X = pd.DataFrame(rows, columns=self.feature_cols)

        member_probas = {}
        for name in config.ENSEMBLE_MEMBERS:
            raw = self.bundle["classifiers"][name].predict_proba(X)
            member_probas[name] = self.bundle["scalers"][name].transform(raw)
        probs = blend(member_probas, self.bundle["ensemble_weights"])

        gnames = self.bundle["goal_model_names"]
        lam_h = np.clip(np.mean([self.bundle["goal_models"][f"home_{n}"].predict(X)
                                 for n in gnames], axis=0), 0.05, 8.0)
        lam_a = np.clip(np.mean([self.bundle["goal_models"][f"away_{n}"].predict(X)
                                 for n in gnames], axis=0), 0.05, 8.0)
        return probs, lam_h, lam_a

    def win_draw_loss(self, home: str, away: str, **kw) -> tuple[float, float, float]:
        p = self.predict(home, away, **kw).probs
        return p["home_win"], p["draw"], p["away_win"]


def _r(v, nd: int = 3):
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, nd)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pred = Predictor()
    p = pred.predict("Brazil", "Germany", neutral=True)
    print(p.probs, p.expected_goals)
    print(p.top_scores[:5])

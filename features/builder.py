"""Leakage-proof feature engineering.

The builder makes a single chronological pass over the match table. For each
match it first *extracts* features from the current team/H2H state (which by
construction only contains information from strictly earlier matches) and
only then *observes* the result to update state. The same `extract` code path
serves training and live prediction, which guarantees train/serve parity.

The end-of-pass state is snapshotted to disk so the API and the tournament
simulator can build features for future fixtures.
"""
from __future__ import annotations

import logging
import math
from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import config
from data.ingestion import MatchDataset, load_dataset, tournament_importance
from features.elo import EloEngine

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def _new_h2h_deque() -> deque:
    """Module-level factory so the state snapshot stays picklable."""
    return deque(maxlen=30)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


@dataclass
class MatchRecord:
    """One past match from a single team's perspective."""
    date: datetime
    gf: int
    ga: int
    points: int          # 3 / 1 / 0
    clean_sheet: bool
    opp_elo: float       # opponent Elo *before* that match
    importance: int


@dataclass
class TeamState:
    history: deque = field(default_factory=lambda: deque(maxlen=config.HISTORY_MAXLEN))
    recent_dates: deque = field(default_factory=lambda: deque(maxlen=80))
    matches_played: int = 0
    last_date: datetime | None = None


class FeatureBuilder:
    """Chronological feature extractor + state container."""

    def __init__(self, dataset: MatchDataset | None = None):
        self.elo = EloEngine()
        self.teams: dict[str, TeamState] = defaultdict(TeamState)
        # H2H keyed by alphabetically sorted pair; records stored as
        # (date, goals_first, goals_second) where "first" = sorted()[0].
        self.h2h: dict[tuple[str, str], deque] = defaultdict(_new_h2h_deque)
        self.centroids: dict[str, tuple[float, float]] = {}
        self._fifa: dict[str, tuple[list, list, list]] = {}
        if dataset is not None:
            self.centroids = dataset.centroids
            if dataset.fifa_rankings is not None:
                self._index_fifa(dataset.fifa_rankings)

    # ------------------------------------------------------------------
    # Optional FIFA ranking lookup (point-in-time, leak-free via bisect)
    # ------------------------------------------------------------------
    def _index_fifa(self, df: pd.DataFrame) -> None:
        for country, grp in df.groupby("country_full" if "country_full" in df.columns else "country"):
            grp = grp.sort_values("rank_date")
            self._fifa[str(country)] = (
                list(grp["rank_date"]), list(grp["rank"]),
                list(grp.get("total_points", grp["rank"])),
            )

    def _fifa_rank(self, team: str, date: datetime) -> tuple[float, float]:
        entry = self._fifa.get(team)
        if not entry:
            return (np.nan, np.nan)
        dates, ranks, points = entry
        i = bisect_right(dates, date) - 1
        if i < 0:
            return (np.nan, np.nan)
        return (float(ranks[i]), float(points[i]))

    # ------------------------------------------------------------------
    # Per-side feature blocks (all computed from pre-match state only)
    # ------------------------------------------------------------------
    def _form_block(self, hist: list[MatchRecord], n: int, prefix: str) -> dict:
        recent = hist[-n:]
        out: dict[str, float] = {}
        k = len(recent)
        if k == 0:
            keys = ("wins", "draws", "losses", "points", "ppg", "gf", "ga", "gd",
                    "gf_pg", "ga_pg", "clean_sheets", "cs_rate", "win_rate")
            return {f"{prefix}_form{n}_{key}": np.nan for key in keys}
        wins = sum(1 for m in recent if m.points == 3)
        draws = sum(1 for m in recent if m.points == 1)
        losses = k - wins - draws
        gf = sum(m.gf for m in recent)
        ga = sum(m.ga for m in recent)
        cs = sum(1 for m in recent if m.clean_sheet)
        pts = 3 * wins + draws
        out[f"{prefix}_form{n}_wins"] = wins
        out[f"{prefix}_form{n}_draws"] = draws
        out[f"{prefix}_form{n}_losses"] = losses
        out[f"{prefix}_form{n}_points"] = pts
        out[f"{prefix}_form{n}_ppg"] = pts / k
        out[f"{prefix}_form{n}_gf"] = gf
        out[f"{prefix}_form{n}_ga"] = ga
        out[f"{prefix}_form{n}_gd"] = gf - ga
        out[f"{prefix}_form{n}_gf_pg"] = gf / k
        out[f"{prefix}_form{n}_ga_pg"] = ga / k
        out[f"{prefix}_form{n}_clean_sheets"] = cs
        out[f"{prefix}_form{n}_cs_rate"] = cs / k
        out[f"{prefix}_form{n}_win_rate"] = wins / k
        return out

    @staticmethod
    def _streaks(hist: list[MatchRecord]) -> tuple[int, int, int]:
        win = 0
        for m in reversed(hist):
            if m.points == 3:
                win += 1
            else:
                break
        unbeaten = 0
        for m in reversed(hist):
            if m.points >= 1:
                unbeaten += 1
            else:
                break
        loss = 0
        for m in reversed(hist):
            if m.points == 0:
                loss += 1
            else:
                break
        return win, unbeaten, loss

    def _momentum(self, hist: list[MatchRecord], n: int = 10) -> tuple[float, float]:
        recent = hist[-n:]
        if not recent:
            return (np.nan, np.nan)
        wsum = psum = gsum = 0.0
        for k, m in enumerate(reversed(recent)):       # k=0 most recent
            w = config.MOMENTUM_DECAY ** k
            wsum += w
            psum += w * m.points
            gsum += w * (m.gf - m.ga)
        return (psum / wsum, gsum / wsum)

    def _sos(self, hist: list[MatchRecord], own_elo: float) -> dict[str, float]:
        out = {}
        for n in config.SOS_WINDOWS:
            recent = hist[-n:]
            if recent:
                opp = float(np.mean([m.opp_elo for m in recent]))
                out[f"sos{n}_opp_elo"] = opp
                out[f"sos{n}_elo_gap"] = own_elo - opp
            else:
                out[f"sos{n}_opp_elo"] = np.nan
                out[f"sos{n}_elo_gap"] = np.nan
        return out

    def _side_features(self, prefix: str, team: str, date: datetime,
                       venue_latlon: tuple[float, float] | None,
                       host_country: str) -> dict:
        st = self.teams[team]
        hist = list(st.history)
        elo = self.elo.get(team)
        f: dict[str, float] = {f"{prefix}_elo": elo}

        rank, rank_pts = self._fifa_rank(team, date)
        f[f"{prefix}_fifa_rank"] = rank
        f[f"{prefix}_fifa_points"] = rank_pts

        for n in config.FORM_WINDOWS:
            f.update(self._form_block(hist, n, prefix))

        win, unbeaten, loss = self._streaks(hist)
        f[f"{prefix}_win_streak"] = win
        f[f"{prefix}_unbeaten_streak"] = unbeaten
        f[f"{prefix}_loss_streak"] = loss

        mom_pts, mom_gd = self._momentum(hist)
        f[f"{prefix}_momentum_points"] = mom_pts
        f[f"{prefix}_momentum_gd"] = mom_gd

        for key, val in self._sos(hist, elo).items():
            f[f"{prefix}_{key}"] = val

        # Rest & congestion
        if st.last_date is not None:
            f[f"{prefix}_rest_days"] = min((date - st.last_date).days, 60)
        else:
            f[f"{prefix}_rest_days"] = np.nan
        dates = list(st.recent_dates)
        for horizon in (30, 90, 365):
            f[f"{prefix}_matches_last_{horizon}d"] = sum(
                1 for d in dates if (date - d).days <= horizon)
        f[f"{prefix}_experience"] = st.matches_played

        # Weighted average importance of recent matches (schedule profile)
        if hist:
            f[f"{prefix}_recent_importance"] = float(np.mean([m.importance for m in hist[-10:]]))
        else:
            f[f"{prefix}_recent_importance"] = np.nan

        # Travel
        home_latlon = self.centroids.get(team)
        if home_latlon and venue_latlon:
            f[f"{prefix}_travel_km"] = haversine_km(*home_latlon, *venue_latlon)
            f[f"{prefix}_tz_diff"] = abs(home_latlon[1] - venue_latlon[1]) / 15.0
        else:
            f[f"{prefix}_travel_km"] = np.nan
            f[f"{prefix}_tz_diff"] = np.nan
        f[f"{prefix}_is_host"] = float(team == host_country)
        return f

    def _h2h_features(self, home: str, away: str) -> dict:
        a, b = sorted((home, away))
        records = list(self.h2h[(a, b)])     # (date, goals_a, goals_b)
        sign = 1 if home == a else -1        # +1 when "a" is current home team
        f: dict[str, float] = {}
        for n in config.H2H_WINDOWS:
            recent = records[-n:]
            k = len(recent)
            if k == 0:
                for key in ("wins", "draws", "losses", "gd", "avg_goals", "weighted_score"):
                    f[f"h2h{n}_{key}"] = np.nan
                continue
            wins = draws = losses = 0
            gd_total = goals_total = 0
            wscore = wsum = 0.0
            for i, (_, ga_, gb_) in enumerate(reversed(recent)):  # most recent first
                diff = sign * (ga_ - gb_)                          # home-team perspective
                goals_total += ga_ + gb_
                gd_total += diff
                if diff > 0:
                    wins += 1; res = 1.0
                elif diff == 0:
                    draws += 1; res = 0.5
                else:
                    losses += 1; res = 0.0
                w = config.MOMENTUM_DECAY ** i                     # recent meetings weigh more
                wscore += w * res
                wsum += w
            f[f"h2h{n}_wins"] = wins
            f[f"h2h{n}_draws"] = draws
            f[f"h2h{n}_losses"] = losses
            f[f"h2h{n}_gd"] = gd_total
            f[f"h2h{n}_avg_goals"] = goals_total / k
            f[f"h2h{n}_weighted_score"] = wscore / wsum
        f["h2h_total_meetings"] = len(records)
        return f

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self, home: str, away: str, date: datetime, tournament: str,
                neutral: bool, country: str = "", importance: int | None = None) -> dict:
        """Feature vector for a fixture, using only past information."""
        if importance is None:
            importance = tournament_importance(tournament)
        venue = self.centroids.get(country) or (None if neutral else self.centroids.get(home))

        f: dict[str, float] = {}
        f.update(self._side_features("home", home, date, venue, country))
        f.update(self._side_features("away", away, date, venue, country))
        f.update(self._h2h_features(home, away))

        # Context
        f["importance"] = importance
        f["neutral"] = float(neutral)
        f["is_world_cup"] = float(importance == 5)
        f["is_friendly"] = float(importance == 1)
        f["is_qualifier"] = float("qualification" in tournament.lower())
        f["year"] = date.year
        f["month"] = date.month
        f["era"] = (date.year - 1900) / 10.0

        # Elo expectation & deltas (the strongest single signals)
        f["elo_expectation_home"] = self.elo.expectation(home, away, neutral)
        f["elo_delta"] = f["home_elo"] - f["away_elo"]
        f["fifa_rank_delta"] = f["away_fifa_rank"] - f["home_fifa_rank"]  # positive = home better
        for n in config.FORM_WINDOWS:
            f[f"form{n}_ppg_delta"] = f[f"home_form{n}_ppg"] - f[f"away_form{n}_ppg"]
            f[f"form{n}_gd_delta"] = f[f"home_form{n}_gd"] - f[f"away_form{n}_gd"]
            f[f"form{n}_gf_pg_delta"] = f[f"home_form{n}_gf_pg"] - f[f"away_form{n}_gf_pg"]
            f[f"form{n}_ga_pg_delta"] = f[f"home_form{n}_ga_pg"] - f[f"away_form{n}_ga_pg"]
        f["momentum_points_delta"] = f["home_momentum_points"] - f["away_momentum_points"]
        f["momentum_gd_delta"] = f["home_momentum_gd"] - f["away_momentum_gd"]
        f["rest_delta"] = f["home_rest_days"] - f["away_rest_days"]
        f["experience_delta"] = f["home_experience"] - f["away_experience"]
        f["sos10_delta"] = f["home_sos10_opp_elo"] - f["away_sos10_opp_elo"]
        f["travel_delta"] = f["home_travel_km"] - f["away_travel_km"]
        f["streak_delta"] = f["home_win_streak"] - f["away_win_streak"]
        return f

    def observe(self, home: str, away: str, date: datetime, home_score: int,
                away_score: int, tournament: str, neutral: bool,
                importance: int | None = None) -> None:
        """Update all state with a completed match. Call *after* extract."""
        if importance is None:
            importance = tournament_importance(tournament)
        elo_home_pre, elo_away_pre = self.elo.get(home), self.elo.get(away)
        self.elo.update(home, away, home_score, away_score, importance, neutral)

        if home_score > away_score:
            ph, pa = 3, 0
        elif home_score == away_score:
            ph, pa = 1, 1
        else:
            ph, pa = 0, 3
        for team, gf, ga, pts, opp_elo in (
            (home, home_score, away_score, ph, elo_away_pre),
            (away, away_score, home_score, pa, elo_home_pre),
        ):
            st = self.teams[team]
            st.history.append(MatchRecord(date, gf, ga, pts, ga == 0, opp_elo, importance))
            st.recent_dates.append(date)
            st.matches_played += 1
            st.last_date = date

        a, b = sorted((home, away))
        if home == a:
            self.h2h[(a, b)].append((date, home_score, away_score))
        else:
            self.h2h[(a, b)].append((date, away_score, home_score))

    def build(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Run the full chronological pass and return the feature matrix."""
        rows: list[dict] = []
        n = len(matches)
        for i, row in enumerate(matches.itertuples(index=False)):
            feats = self.extract(row.home_team, row.away_team, row.date,
                                 row.tournament, bool(row.neutral), row.country,
                                 int(row.importance))
            feats["match_id"] = row.match_id
            feats["min_history"] = min(self.teams[row.home_team].matches_played,
                                       self.teams[row.away_team].matches_played)
            rows.append(feats)
            self.observe(row.home_team, row.away_team, row.date,
                         int(row.home_score), int(row.away_score),
                         row.tournament, bool(row.neutral), int(row.importance))
            if (i + 1) % 10000 == 0:
                logger.info("processed %d / %d matches", i + 1, n)

        feat_df = pd.DataFrame(rows)
        meta_cols = ["match_id", "date", "home_team", "away_team", "tournament",
                     "home_score", "away_score", "outcome"]
        out = matches[meta_cols].merge(feat_df, on="match_id", how="inner")
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path=config.STATE_SNAPSHOT) -> None:
        joblib.dump(self, path)
        logger.info("State snapshot saved to %s", path)

    @staticmethod
    def load(path=config.STATE_SNAPSHOT) -> "FeatureBuilder":
        return joblib.load(path)


META_COLUMNS = ["match_id", "date", "home_team", "away_team", "tournament",
                "home_score", "away_score", "outcome", "min_history"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLUMNS]


def run_pipeline() -> pd.DataFrame:
    """Entry point: ingest -> build features -> persist parquet + snapshot."""
    ds = load_dataset()
    builder = FeatureBuilder(ds)
    features = builder.build(ds.matches)
    features.to_parquet(config.FEATURES_PARQUET, index=False)
    builder.save()
    logger.info("Wrote %d rows x %d cols to %s", len(features),
                features.shape[1], config.FEATURES_PARQUET)
    return features

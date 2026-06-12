"""Data ingestion pipeline.

Loads the core results dataset and merges any optional auxiliary sources
(FIFA rankings, market values, betting odds) when their files are present.
Every loader validates schema and types so downstream stages can rely on a
clean, chronologically sorted match table.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]


# ---------------------------------------------------------------------------
# Tournament importance encoding
# ---------------------------------------------------------------------------
def tournament_importance(tournament: str) -> int:
    """Map a free-text tournament name onto a 1..5 importance tier.

    5 = World Cup finals, 4 = continental finals / Confederations Cup,
    3 = qualifiers / Nations League, 2 = other competitive, 1 = friendly.
    """
    t = tournament.lower()
    if "friendly" in t:
        return 1
    if "world cup" in t and "qualification" not in t:
        return 5
    continental_finals = (
        "uefa euro", "copa américa", "copa america", "african cup of nations",
        "africa cup of nations", "afc asian cup", "gold cup",
        "concacaf championship", "oceania nations cup", "ofc nations cup",
        "confederations cup",
    )
    if any(name in t for name in continental_finals) and "qualification" not in t:
        return 4
    if "qualification" in t or "nations league" in t:
        return 3
    return 2


@dataclass
class MatchDataset:
    """Container for the merged match table plus auxiliary lookups."""
    matches: pd.DataFrame
    centroids: dict[str, tuple[float, float]] = field(default_factory=dict)
    fifa_rankings: pd.DataFrame | None = None
    market_values: pd.DataFrame | None = None
    odds: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_results(path: Path = config.RESULTS_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it from "
            "https://github.com/martj42/international_results"
        )
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"results.csv is missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    # Matches without a final score (abandoned / future fixtures) are unusable.
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE") | df["neutral"].eq(True)
    df["tournament"] = df["tournament"].fillna("Friendly").astype(str)
    df["country"] = df["country"].fillna("").astype(str)
    df["city"] = df["city"].fillna("").astype(str)

    df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    df = df.sort_values(["date", "home_team", "away_team"], kind="mergesort")
    df = df.reset_index(drop=True)
    df["match_id"] = df.index

    df["importance"] = df["tournament"].map(tournament_importance)
    df["outcome"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
        [0, 1],
        default=2,
    )
    logger.info("Loaded %d matches: %s .. %s", len(df), df["date"].min().date(),
                df["date"].max().date())
    return df


def load_centroids(path: Path = config.CENTROIDS_CSV) -> dict[str, tuple[float, float]]:
    """Country -> (lat, lon). Used for travel distance / timezone features."""
    if not path.exists():
        logger.warning("Centroid file missing; travel features will be NaN.")
        return {}
    df = pd.read_csv(path)
    out: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        out[str(row["COUNTRY"])] = (float(row["latitude"]), float(row["longitude"]))
    out.update(_CENTROID_ALIASES_RESOLVED(out))
    return out


def _CENTROID_ALIASES_RESOLVED(base: dict[str, tuple[float, float]]) -> dict:
    """Map dataset team/country names onto centroid table names."""
    aliases = {
        "United States": "United States of America" if "United States of America" in base else "United States",
        "USA": "United States",
        "South Korea": "Korea, Republic of" if "Korea, Republic of" in base else "South Korea",
        "North Korea": "Korea, Democratic People's Republic of",
        "DR Congo": "Congo DRC",
        "Republic of Ireland": "Ireland",
        "Ivory Coast": "Côte d'Ivoire",
        "Cape Verde": "Cabo Verde",
        "England": "United Kingdom",
        "Scotland": "United Kingdom",
        "Wales": "United Kingdom",
        "Northern Ireland": "United Kingdom",
        "Russia": "Russian Federation",
        "Vietnam": "Viet Nam",
        "Iran": "Iran, Islamic Republic of",
        "Syria": "Syrian Arab Republic",
        "Laos": "Lao People's Democratic Republic",
        "Tanzania": "Tanzania, United Republic of",
        "Venezuela": "Venezuela, Bolivarian Republic of",
        "Bolivia": "Bolivia, Plurinational State of",
        "Moldova": "Moldova, Republic of",
        "Czech Republic": "Czechia" if "Czechia" in base else "Czech Republic",
        "Türkiye": "Turkey",
        "Curaçao": "Curacao" if "Curacao" in base else "Curaçao",
    }
    resolved = {}
    for team, country in aliases.items():
        if country in base:
            resolved[team] = base[country]
    return resolved


def load_fifa_rankings(path: Path = config.FIFA_RANKING_CSV) -> pd.DataFrame | None:
    """Optional: long-format FIFA ranking history (rank_date, country, rank, points)."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["rank_date"] = pd.to_datetime(df["rank_date"])
    df = df.sort_values("rank_date")
    logger.info("Loaded FIFA rankings: %d rows", len(df))
    return df


def load_market_values(path: Path = config.MARKET_VALUES_CSV) -> pd.DataFrame | None:
    """Optional: squad market values (date, team, squad_value, avg_player_value, avg_age)."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def load_odds(path: Path = config.ODDS_CSV) -> pd.DataFrame | None:
    """Optional: pre-match odds (date, home_team, away_team, odds_h, odds_d, odds_a).

    Only *pre-match* (opening/closing) odds may live in this file; the feature
    builder joins them on match identity so no post-match information can leak.
    """
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_dataset() -> MatchDataset:
    """Load and merge every available source into one MatchDataset."""
    return MatchDataset(
        matches=load_results(),
        centroids=load_centroids(),
        fifa_rankings=load_fifa_rankings(),
        market_values=load_market_values(),
        odds=load_odds(),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ds = load_dataset()
    print(ds.matches.tail())
    print(f"{len(ds.matches)} matches, {ds.matches['home_team'].nunique()} home teams")

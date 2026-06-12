"""Feature-builder correctness, including the critical no-leakage property."""
from datetime import datetime

import numpy as np
import pandas as pd

from features.builder import FeatureBuilder, feature_columns


def _matches(rows):
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                     "home_score", "away_score", "tournament",
                                     "city", "country", "neutral"])
    df["date"] = pd.to_datetime(df["date"])
    df["importance"] = 1
    df["outcome"] = np.select(
        [df.home_score > df.away_score, df.home_score == df.away_score], [0, 1], 2)
    df = df.reset_index(drop=True)
    df["match_id"] = df.index
    return df


BASE = [
    ("2020-01-01", "A", "B", 2, 0, "Friendly", "x", "A", False),
    ("2020-02-01", "A", "C", 1, 1, "Friendly", "x", "A", False),
    ("2020-03-01", "B", "C", 0, 3, "Friendly", "x", "B", False),
    ("2020-04-01", "A", "B", 1, 0, "Friendly", "x", "", True),
    ("2020-05-01", "C", "A", 2, 2, "Friendly", "x", "C", False),
]


def test_no_future_leakage():
    """Feature rows must be identical whether or not later matches exist."""
    full = _matches(BASE)
    truncated = _matches(BASE[:3])

    f_full = FeatureBuilder().build(full)
    f_trunc = FeatureBuilder().build(truncated)

    cols = feature_columns(f_trunc)
    a = f_full.loc[:2, cols].reset_index(drop=True)
    b = f_trunc.loc[:, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_first_match_has_no_form():
    f = FeatureBuilder().build(_matches(BASE))
    first = f.iloc[0]
    assert np.isnan(first["home_form5_points"])
    assert np.isnan(first["home_rest_days"])
    assert first["home_experience"] == 0


def test_form_and_streaks_accumulate():
    f = FeatureBuilder().build(_matches(BASE))
    # Match 4 (2020-04-01): A has played 2 (W, D) -> 4 points, unbeaten streak 2.
    row = f.iloc[3]
    assert row["home_form5_points"] == 4
    assert row["home_unbeaten_streak"] == 2
    assert row["home_win_streak"] == 0      # last match was a draw
    # B has played 2 (L, L) -> loss streak 2
    assert row["away_loss_streak"] == 2
    assert row["away_form5_points"] == 0


def test_h2h_perspective():
    f = FeatureBuilder().build(_matches(BASE))
    # Match 4: A vs B again; A won the first meeting 2-0.
    row = f.iloc[3]
    assert row["h2h3_wins"] == 1
    assert row["h2h3_losses"] == 0
    assert row["h2h3_gd"] == 2
    # Match 5: C hosts A. Their meetings: A1-C1, B-C(no), so 1 draw.
    row5 = f.iloc[4]
    assert row5["h2h3_draws"] == 1


def test_elo_updates_after_extract():
    b = FeatureBuilder()
    df = _matches(BASE)
    out = b.build(df)
    # First A-B match must be rated at base Elo (no information yet)...
    assert out.iloc[0]["home_elo"] == 1500.0
    # ...and by the rematch A must be rated above B.
    rematch = out.iloc[3]
    assert rematch["home_elo"] > rematch["away_elo"]


def test_extract_matches_build_path():
    """Live extraction (API path) must equal the training-pass features."""
    df = _matches(BASE)
    b1 = FeatureBuilder()
    built = b1.build(df)

    b2 = FeatureBuilder()
    for row in df.iloc[:4].itertuples(index=False):
        b2.observe(row.home_team, row.away_team, row.date, row.home_score,
                   row.away_score, row.tournament, bool(row.neutral))
    last = df.iloc[4]
    live = b2.extract(last.home_team, last.away_team, last.date,
                      last.tournament, bool(last.neutral), last.country)
    for col in ("home_elo", "away_elo", "home_form5_points", "h2h3_draws",
                "elo_expectation_home"):
        built_val = built.iloc[4][col]
        if isinstance(built_val, float) and np.isnan(built_val):
            assert np.isnan(live[col])
        else:
            assert built_val == live[col], col

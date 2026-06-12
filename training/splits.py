"""Strict chronological train/validation/test splitting.

Matches are never shuffled. Train: ..2018, Validation: 2019-2022,
Test: 2023-present (configurable via config.SplitDates).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from features.builder import feature_columns

logger = logging.getLogger(__name__)


@dataclass
class Splits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    feature_cols: list[str]

    def xy(self, part: str, target: str = "outcome"):
        df = getattr(self, part)
        return df[self.feature_cols], df[target]

    def summary(self) -> str:
        return (f"train={len(self.train)} ({self.train['date'].min().date()}.."
                f"{self.train['date'].max().date()}) | "
                f"valid={len(self.valid)} | test={len(self.test)} | "
                f"features={len(self.feature_cols)}")


def make_splits(df: pd.DataFrame | None = None,
                splits: config.SplitDates = config.SplitDates()) -> Splits:
    if df is None:
        df = pd.read_parquet(config.FEATURES_PARQUET)
    df = df.sort_values("date").reset_index(drop=True)

    # Drop early-history matches with mostly-empty form features and the
    # pre-modern era; both stay in the *state* pass (the builder saw them),
    # they are merely excluded from supervised training.
    usable = df[(df["min_history"] >= config.MIN_TEAM_HISTORY)
                & (df["date"].dt.year >= config.TRAIN_START_YEAR)].copy()

    train_end = pd.Timestamp(splits.train_end)
    valid_end = pd.Timestamp(splits.valid_end)
    train = usable[usable["date"] <= train_end]
    valid = usable[(usable["date"] > train_end) & (usable["date"] <= valid_end)]
    test = usable[usable["date"] > valid_end]

    cols = feature_columns(df)
    # Drop optional-source columns that are entirely absent in training data.
    all_nan = [c for c in cols if train[c].isna().all()]
    if all_nan:
        logger.info("Dropping %d all-NaN feature columns: %s", len(all_nan), all_nan)
        cols = [c for c in cols if c not in all_nan]
    # Guard against accidental target leakage into the feature list.
    forbidden = {"home_score", "away_score", "outcome"}
    assert not (set(cols) & forbidden), "target columns leaked into features"

    sp = Splits(train=train, valid=valid, test=test, feature_cols=cols)
    logger.info(sp.summary())
    return sp


def assert_chronological(sp: Splits) -> None:
    """Sanity check used by tests: no validation/test match predates train end."""
    assert sp.train["date"].max() < sp.valid["date"].min()
    assert sp.valid["date"].max() < sp.test["date"].min()

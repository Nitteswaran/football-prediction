"""Global configuration for Pitchsense.

All paths, split dates, feature windows and model constants live here so that
every pipeline stage shares a single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Secrets (Stripe keys etc.) live in an untracked .env file in development;
# in production they come from the platform's secret manager.
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"
PARAMS_DIR = PROJECT_ROOT / "models" / "params"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RESULTS_CSV = RAW_DIR / "results.csv"
SHOOTOUTS_CSV = RAW_DIR / "shootouts.csv"
GOALSCORERS_CSV = RAW_DIR / "goalscorers.csv"
CENTROIDS_CSV = RAW_DIR / "country_centroids.csv"
# Optional data sources -- ingestion degrades gracefully when absent.
FIFA_RANKING_CSV = RAW_DIR / "fifa_ranking.csv"
MARKET_VALUES_CSV = RAW_DIR / "market_values.csv"
ODDS_CSV = RAW_DIR / "odds.csv"

FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
STATE_SNAPSHOT = PROCESSED_DIR / "state_snapshot.joblib"

for _d in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, PARAMS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Chronological splits (never shuffle)
# ---------------------------------------------------------------------------
TRAIN_END = "2018-12-31"      # 1872 .. 2018
VALID_END = "2022-12-31"      # 2019 .. 2022
# test = 2023 .. present

# Matches before a team has played MIN_TEAM_HISTORY games carry mostly-NaN
# form features; we keep them in the state pass but drop them from training.
MIN_TEAM_HISTORY = 5
TRAIN_START_YEAR = 1920       # pre-1920 football is a different sport statistically

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
FORM_WINDOWS = (3, 5, 10, 20)
H2H_WINDOWS = (3, 5, 10)
SOS_WINDOWS = (5, 10, 20)
MOMENTUM_DECAY = 0.9          # weight of match k matches ago = MOMENTUM_DECAY**k
HISTORY_MAXLEN = 30           # per-team rolling history buffer

# ---------------------------------------------------------------------------
# Elo engine
# ---------------------------------------------------------------------------
ELO_BASE = 1500.0
ELO_HOME_ADVANTAGE = 80.0     # added to home rating when not on neutral ground
ELO_SPREAD = 400.0

# K-factor by tournament importance tier (eloratings.net convention)
ELO_K_BY_IMPORTANCE = {5: 60.0, 4: 50.0, 3: 40.0, 2: 30.0, 1: 20.0}

# ---------------------------------------------------------------------------
# Targets / labels
# ---------------------------------------------------------------------------
OUTCOME_LABELS = ("home_win", "draw", "away_win")  # class indices 0, 1, 2
MAX_GOALS_GRID = 10           # scoreline grid is (0..10) x (0..10)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
OPTUNA_TRIALS = 100
OPTUNA_TIMEOUT_PER_STUDY = 3600  # seconds; safety budget per model study

CLASSIFIER_NAMES = ("logreg", "random_forest", "extra_trees", "xgboost",
                    "lightgbm", "catboost", "mlp")
ENSEMBLE_MEMBERS = ("xgboost", "lightgbm", "catboost", "mlp")
GOAL_MODEL_NAMES = ("xgboost", "lightgbm", "catboost")

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
N_SIMULATIONS = 100_000
DEFAULT_TOURNAMENT = "FIFA World Cup"


@dataclass(frozen=True)
class SplitDates:
    train_end: str = TRAIN_END
    valid_end: str = VALID_END


@dataclass
class PipelineConfig:
    """Bundle passed around by pipeline entry points."""
    splits: SplitDates = field(default_factory=SplitDates)
    seed: int = RANDOM_SEED
    optuna_trials: int = OPTUNA_TRIALS

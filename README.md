---
title: Pitchsense
emoji: ⚽
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Pitchsense — Football Prediction System

An end-to-end machine learning system for international football: match
outcome probabilities, expected goals, exact-score distributions, and Monte
Carlo World Cup simulation — wrapped in a FastAPI service with a clean web UI.

![stack](https://img.shields.io/badge/python-3.14-0B6E4F) ![models](https://img.shields.io/badge/models-XGB%20·%20LGBM%20·%20CatBoost%20·%20PyTorch-161D1A)

## What it does

| Capability | How |
|---|---|
| Win / Draw / Loss probabilities | Calibrated ensemble of XGBoost, LightGBM, CatBoost and a PyTorch MLP |
| Expected goals | Poisson-objective gradient-boosted regressors (home & away) |
| Exact score probabilities | Dixon-Coles adjusted double Poisson, reconciled with the classifier ensemble |
| Tournament simulation | Vectorised Monte Carlo over the official 2026 format (100,000 runs in seconds) |
| Explainability | SHAP top-50 features, per-prediction driver breakdown |
| Calibration | Per-model temperature scaling + reliability diagrams |

## Leakage prevention (the design constraint)

- **Single chronological pass**: features for each match are extracted from
  state *before* the result is observed. Rolling form, Elo, head-to-head,
  rest days, strength-of-schedule — all strictly past-only.
- **Train / serve parity**: the API and the simulator call the *same*
  `FeatureBuilder.extract()` used to create training rows, against a
  persisted end-of-data state snapshot.
- **Strict chronological splits**: train ≤ 2018, validation 2019–2022,
  test 2023–present. Nothing is ever shuffled.
- **Decisions on validation only**: hyperparameters (Optuna), calibration
  temperatures, ensemble weights and the Dixon-Coles rho are all fitted
  without touching the test window.
- **Tested**: `tests/test_builder.py::test_no_future_leakage` proves feature
  rows are bit-identical whether or not later matches exist.

## Project structure

```
├── config.py            # paths, splits, windows, constants
├── data/                # ingestion pipeline + raw/processed data
├── features/            # Elo engine + chronological feature builder (~200 features)
├── models/              # model zoo, NN, calibration, ensemble, Dixon-Coles
├── training/            # splits, Optuna tuning, training pipeline
├── prediction/          # live Predictor (bundle + state snapshot)
├── simulation/          # vectorised World Cup 2026 Monte Carlo
├── evaluation/          # metrics, reliability diagrams, SHAP reports
├── api/                 # FastAPI service (serves the frontend too)
├── frontend/            # web UI (vanilla JS, no build step)
├── reports/             # generated reports & figures
├── notebooks/           # exploration
└── tests/               # pytest suite incl. leakage tests
```

## Quickstart

```bash
make setup        # venv + dependencies
make data         # download results.csv (49k+ internationals since 1872)
make features     # build feature matrix + state snapshot
make tune         # optional: Optuna, 100 trials per model
make train        # train ensemble + goal models + calibration
make evaluate     # metrics, reliability diagrams, SHAP top-50
make simulate     # 100,000 World Cup 2026 simulations
make api          # serve API + frontend at http://localhost:8000
make test         # run the test suite
```

## Feature engineering (~200 features per match)

- **Strength**: Elo (importance-weighted K, goal-margin multiplier), Elo
  delta, Elo win expectancy, optional FIFA ranks.
- **Form**: wins/draws/losses, points, goals for/against, goal difference,
  clean sheets, win rate over 3/5/10/20-match windows, for both sides.
- **Streaks & momentum**: win/unbeaten/loss streaks, exponentially-decayed
  recent points and goal difference (most recent match weight 1.0, then 0.9…).
- **Head-to-head**: last 3/5/10 meetings — W/D/L, goal difference, average
  goals, recency-weighted score.
- **Schedule**: rest days, match congestion (30/90/365d), strength of
  schedule (mean opponent Elo over 5/10/20), recent match importance.
- **Context**: tournament importance tier, neutral venue, host nation,
  travel distance (haversine from country centroid), timezone shift.
- **Deltas**: home-vs-away differences of every key aggregate.

Optional sources (FIFA rankings, market values, pre-match odds) plug in by
dropping CSVs into `data/raw/` — loaders and point-in-time joins are wired.

## Models

| Family | Models |
|---|---|
| Baseline | Logistic regression |
| Trees | XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees |
| Deep | PyTorch MLP (BatchNorm+GELU+Dropout, early stopping), GRU temporal encoder |
| Score | Poisson goal regressors → Dixon-Coles scoreline grid |
| Ensemble | Softmax-parameterised weights optimised on validation log loss |

## API

```
GET  /api/teams        GET  /api/rankings      GET  /api/worldcup
POST /api/predict      GET  /api/evaluation    GET  /api/health
```

`POST /api/predict {"home_team": "Brazil", "away_team": "Germany"}` returns
calibrated W/D/L probabilities, expected goals, the full scoreline grid, the
most likely scores, per-model probabilities and the human-readable drivers.

## Monetization & deployment

Full prediction insights are gated behind a one-time $5 Stripe Checkout
payment (no user accounts; the paid Checkout session id is the unlock token,
stored in the buyer's browser). Locked responses contain no model output —
the blurred numbers in the UI are decoys.

Configuration is environment-only (see `.env.example`; copy to `.env` for
local dev — `.env` is gitignored and must never be committed):

| Variable | Purpose |
|---|---|
| `STRIPE_SECRET_KEY` | Stripe API key. Unset ⇒ billing disabled, all free. Use `sk_test_…` everywhere except production. |
| `PITCHSENSE_STRIPE_PRICE` | Price id of the unlock (defaults to the live $5 price). |
| `PUBLIC_BASE_URL` | Public origin (`https://…`) used for Stripe redirect URLs. **Required in production** — never derive it from the Host header behind a proxy. |
| `PITCHSENSE_CORS_ORIGINS` | Extra allowed origins, comma-separated. Empty (default) ⇒ no CORS, same-origin only. |

Production checklist:

- Store `STRIPE_SECRET_KEY` in your platform's secret manager (Fly/Render/
  Railway secrets, AWS SSM, …) — not in a file in the image, not in git.
- Serve over HTTPS behind a reverse proxy; run `make serve` (multi-worker,
  proxy headers enabled).
- Rate limits are per-process and in-memory; for serious scale move them to
  the proxy or a shared store.
- A refund does not automatically re-lock a paid session id; revoke manually
  by removing it from `data/processed/paid_sessions.json` if needed.

## Reports

After `make evaluate` / `make simulate`:

- `reports/evaluation_report.md` — model comparison table (accuracy, log
  loss, Brier, RPS, ROC AUC, ECE) + regression metrics
- `reports/figures/` — reliability diagrams, model comparison, SHAP top-50
- `reports/feature_importance_top50.csv`
- `reports/worldcup2026_simulation.json` — advancement → championship
  probabilities for all 48 teams

"""Main training pipeline.

Stages
------
1. Fit every classifier on train, early-stop / select on validation.
2. Temperature-calibrate each model on validation probabilities.
3. Optimise ensemble weights (validation log loss).
4. Fit Poisson goal regressors for home/away goals + average into expectancies.
5. Fit the Dixon-Coles rho on train+valid predictions.
6. Refit final models on train+valid with frozen hyperparameters, evaluate on
   the untouched test set, and persist every artifact.

Usage:
    python -m training.train [--skip-slow] [--no-refit]
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import config
from models.calibration import TemperatureScaler
from models.ensemble import blend, optimize_weights
from models.scoreline import fit_rho
from models.zoo import (fit_classifier, fit_goal_regressor, make_classifier,
                        make_goal_regressor)
from training.splits import Splits, make_splits
from training.tune import load_tuned_params

logger = logging.getLogger(__name__)


def train_classifiers(sp: Splits, names: list[str]) -> tuple[dict, dict]:
    X_tr, y_tr = sp.xy("train")
    X_va, y_va = sp.xy("valid")
    models, valid_probas = {}, {}
    for name in names:
        t0 = time.time()
        clf = make_classifier(name, load_tuned_params(name))
        fit_classifier(name, clf, X_tr, y_tr, X_va, y_va)
        proba = clf.predict_proba(X_va)
        ll = log_loss(y_va, np.clip(proba, 1e-12, 1), labels=[0, 1, 2])
        logger.info("classifier %-13s valid logloss=%.5f (%.0fs)",
                    name, ll, time.time() - t0)
        models[name] = clf
        valid_probas[name] = proba
    return models, valid_probas


def calibrate(valid_probas: dict, y_valid) -> dict[str, TemperatureScaler]:
    scalers = {}
    for name, proba in valid_probas.items():
        sc = TemperatureScaler().fit(proba, np.asarray(y_valid))
        scalers[name] = sc
        logger.info("calibration %-13s temperature=%.3f", name, sc.temperature)
    return scalers


def train_goal_models(sp: Splits, names: list[str]) -> dict:
    X_tr, _ = sp.xy("train")
    X_va, _ = sp.xy("valid")
    out = {}
    for side, target in (("home", "home_score"), ("away", "away_score")):
        y_tr = sp.train[target]
        y_va = sp.valid[target]
        for name in names:
            reg = make_goal_regressor(name)
            fit_goal_regressor(name, reg, X_tr, y_tr, X_va, y_va)
            out[f"{side}_{name}"] = reg
    return out


def goal_expectancies(goal_models: dict, X, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lh = np.mean([goal_models[f"home_{n}"].predict(X) for n in names], axis=0)
    la = np.mean([goal_models[f"away_{n}"].predict(X) for n in names], axis=0)
    return np.clip(lh, 0.05, 8.0), np.clip(la, 0.05, 8.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-slow", action="store_true",
                        help="train ensemble members only (skip RF/ET baselines)")
    parser.add_argument("--no-refit", action="store_true",
                        help="skip the final refit on train+valid")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sp = make_splits()
    logger.info(sp.summary())
    names = list(config.ENSEMBLE_MEMBERS) + ["logreg"]
    if not args.skip_slow:
        names += ["random_forest", "extra_trees"]

    # 1-2: classifiers + calibration -----------------------------------
    models, valid_probas = train_classifiers(sp, names)
    _, y_va = sp.xy("valid")
    scalers = calibrate(valid_probas, y_va)
    calibrated = {n: scalers[n].transform(p) for n, p in valid_probas.items()}

    # 3: ensemble weights over calibrated members -----------------------
    member_probas = {n: calibrated[n] for n in config.ENSEMBLE_MEMBERS}
    weights = optimize_weights(member_probas, np.asarray(y_va))
    ens_valid = blend(member_probas, weights)
    ens_ll = log_loss(y_va, np.clip(ens_valid, 1e-12, 1), labels=[0, 1, 2])
    logger.info("ensemble weights=%s valid logloss=%.5f", weights, ens_ll)

    # 4: goal models -----------------------------------------------------
    goal_models = train_goal_models(sp, list(config.GOAL_MODEL_NAMES))

    # 5: Dixon-Coles rho on train+valid predictions ----------------------
    tv = pd.concat([sp.train, sp.valid])
    lh, la = goal_expectancies(goal_models, tv[sp.feature_cols],
                               list(config.GOAL_MODEL_NAMES))
    rho = fit_rho(lh, la, tv["home_score"].values, tv["away_score"].values)
    logger.info("Dixon-Coles rho=%.4f", rho)

    # 6: final refit on train+valid (frozen hyperparameters) -------------
    final_models, final_goal_models = models, goal_models
    if not args.no_refit:
        X_tv = tv[sp.feature_cols]
        y_tv = tv["outcome"]
        final_models = {}
        for name in names:
            clf = make_classifier(name, load_tuned_params(name))
            # keep validation as the early-stopping set: it is in the past
            # relative to test, so this is still leak-free.
            fit_classifier(name, clf, X_tv, y_tv,
                           sp.valid[sp.feature_cols], sp.valid["outcome"])
            final_models[name] = clf
        final_goal_models = {}
        for side, target in (("home", "home_score"), ("away", "away_score")):
            for gname in config.GOAL_MODEL_NAMES:
                reg = make_goal_regressor(gname)
                fit_goal_regressor(gname, reg, X_tv, tv[target],
                                   sp.valid[sp.feature_cols], sp.valid[target])
                final_goal_models[f"{side}_{gname}"] = reg

    bundle = {
        "feature_cols": sp.feature_cols,
        "classifiers": final_models,
        "scalers": scalers,
        "ensemble_weights": weights,
        "goal_models": final_goal_models,
        "goal_model_names": list(config.GOAL_MODEL_NAMES),
        "rho": rho,
        "split_dates": {"train_end": sp.train["date"].max().isoformat(),
                        "valid_end": sp.valid["date"].max().isoformat()},
        "valid_logloss_ensemble": ens_ll,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    out = config.ARTIFACTS_DIR / "model_bundle.joblib"
    joblib.dump(bundle, out, compress=3)
    (config.ARTIFACTS_DIR / "ensemble_weights.json").write_text(
        json.dumps({"weights": weights, "valid_logloss": ens_ll,
                    "temperatures": {n: s.temperature for n, s in scalers.items()},
                    "rho": rho}, indent=2))
    logger.info("Saved model bundle -> %s", out)


if __name__ == "__main__":
    main()

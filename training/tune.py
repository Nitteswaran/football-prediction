"""Hyperparameter optimisation with Optuna.

Each model gets its own study; the objective is validation log loss (the
metric the ensemble ultimately optimises). Best params are written to
models/params/<model>.json and picked up automatically by training.

Usage:
    python -m training.tune --models xgboost lightgbm catboost mlp --trials 100
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import optuna
from sklearn.metrics import log_loss

import config
from models.zoo import fit_classifier, make_classifier
from training.splits import make_splits

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest(trial: optuna.Trial, model: str) -> dict:
    if model == "xgboost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 2000),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        )
    if model == "lightgbm":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 2500),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 120),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        )
    if model == "catboost":
        return dict(
            iterations=trial.suggest_int("iterations", 400, 2500),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            depth=trial.suggest_int("depth", 4, 9),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
        )
    if model == "mlp":
        n_layers = trial.suggest_int("n_layers", 2, 4)
        width = trial.suggest_categorical("width", [128, 256, 384])
        hidden = tuple(max(32, width // (2 ** i)) for i in range(n_layers))
        return dict(
            hidden=hidden,
            dropout=trial.suggest_float("dropout", 0.1, 0.5),
            lr=trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024]),
        )
    if model == "random_forest":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 900),
            max_depth=trial.suggest_int("max_depth", 6, 24),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 60),
            max_features=trial.suggest_float("max_features", 0.1, 0.9),
        )
    if model == "extra_trees":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000),
            max_depth=trial.suggest_int("max_depth", 6, 26),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 60),
            max_features=trial.suggest_float("max_features", 0.1, 0.9),
        )
    if model == "logreg":
        return dict(C=trial.suggest_float("C", 1e-4, 10.0, log=True))
    raise ValueError(model)


def tune_model(model_name: str, n_trials: int, timeout: int | None = None) -> dict:
    sp = make_splits()
    X_tr, y_tr = sp.xy("train")
    X_va, y_va = sp.xy("valid")

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, model_name)
        # hidden tuple is reconstructed from n_layers/width inside _suggest;
        # only pass model-constructor keys onward.
        clf = make_classifier(model_name, {k: v for k, v in params.items()
                                           if k not in ("n_layers", "width")})
        fit_classifier(model_name, clf, X_tr, y_tr, X_va, y_va)
        proba = clf.predict_proba(X_va)
        return log_loss(y_va, np.clip(proba, 1e-12, 1), labels=[0, 1, 2])

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        study_name=f"{model_name}_outcome",
    )
    study.optimize(objective, n_trials=n_trials,
                   timeout=timeout or config.OPTUNA_TIMEOUT_PER_STUDY,
                   show_progress_bar=False)

    best = _suggest(optuna.trial.FixedTrial(study.best_params), model_name)
    best = {k: v for k, v in best.items() if k not in ("n_layers", "width")}
    out = {"params": _jsonable(best), "valid_logloss": study.best_value,
           "n_trials": len(study.trials)}
    path = config.PARAMS_DIR / f"{model_name}.json"
    path.write_text(json.dumps(out, indent=2))
    logger.info("%s tuned: logloss=%.5f over %d trials -> %s",
                model_name, study.best_value, len(study.trials), path)
    return out


def _jsonable(params: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}


def load_tuned_params(model_name: str) -> dict:
    path = config.PARAMS_DIR / f"{model_name}.json"
    if path.exists():
        return json.loads(path.read_text())["params"]
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(config.ENSEMBLE_MEMBERS))
    parser.add_argument("--trials", type=int, default=config.OPTUNA_TRIALS)
    parser.add_argument("--timeout", type=int, default=config.OPTUNA_TIMEOUT_PER_STUDY,
                        help="per-study wall clock budget in seconds")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for m in args.models:
        tune_model(m, args.trials, args.timeout)


if __name__ == "__main__":
    main()

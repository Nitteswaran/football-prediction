"""Model factories.

Every classifier exposes fit / predict_proba; every regressor fit / predict.
Factories accept a params dict (typically produced by Optuna tuning) and fall
back to robust defaults. Models that cannot digest NaNs are wrapped in
imputation pipelines so the whole zoo accepts the same raw feature matrix.
"""
from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

import config
from models.nn import TorchMLPClassifier

SEED = config.RANDOM_SEED


# ---------------------------------------------------------------------------
# Classifiers (W/D/L)
# ---------------------------------------------------------------------------
def make_classifier(name: str, params: dict[str, Any] | None = None):
    params = dict(params or {})
    if name == "logreg":
        c = params.pop("C", 0.1)
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=c, max_iter=2000, random_state=SEED)),
        ])
    if name == "random_forest":
        defaults = dict(n_estimators=500, max_depth=14, min_samples_leaf=20,
                        max_features="sqrt", n_jobs=-1, random_state=SEED)
        defaults.update(params)
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(**defaults)),
        ])
    if name == "extra_trees":
        defaults = dict(n_estimators=600, max_depth=16, min_samples_leaf=15,
                        max_features="sqrt", n_jobs=-1, random_state=SEED)
        defaults.update(params)
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(**defaults)),
        ])
    if name == "xgboost":
        defaults = dict(n_estimators=900, learning_rate=0.03, max_depth=5,
                        min_child_weight=8, subsample=0.8, colsample_bytree=0.7,
                        reg_lambda=2.0, reg_alpha=0.1, objective="multi:softprob",
                        num_class=3, tree_method="hist", eval_metric="mlogloss",
                        n_jobs=-1, random_state=SEED, early_stopping_rounds=None)
        defaults.update(params)
        return XGBClassifier(**defaults)
    if name == "lightgbm":
        defaults = dict(n_estimators=1200, learning_rate=0.03, num_leaves=63,
                        min_child_samples=40, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=0.1,
                        objective="multiclass", num_class=3, n_jobs=-1,
                        random_state=SEED, verbosity=-1)
        defaults.update(params)
        return LGBMClassifier(**defaults)
    if name == "catboost":
        defaults = dict(iterations=1500, learning_rate=0.03, depth=6,
                        l2_leaf_reg=4.0, loss_function="MultiClass",
                        random_seed=SEED, verbose=0, allow_writing_files=False)
        defaults.update(params)
        return CatBoostClassifier(**defaults)
    if name == "mlp":
        defaults = dict(hidden=(256, 128, 64), dropout=0.3, lr=1e-3,
                        weight_decay=1e-4, batch_size=512, max_epochs=80)
        defaults.update(params)
        if isinstance(defaults.get("hidden"), list):
            defaults["hidden"] = tuple(defaults["hidden"])
        return TorchMLPClassifier(**defaults)
    raise ValueError(f"unknown classifier {name!r}")


def fit_classifier(name: str, model, X_train, y_train, X_valid=None, y_valid=None):
    """Fit with early stopping on the validation set where supported."""
    if name == "xgboost" and X_valid is not None:
        model.set_params(early_stopping_rounds=60)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    elif name == "lightgbm" and X_valid is not None:
        import lightgbm as lgb
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    elif name == "catboost" and X_valid is not None:
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                  early_stopping_rounds=80)
    elif name == "mlp" and X_valid is not None:
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    else:
        model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Goal regressors (Poisson objectives)
# ---------------------------------------------------------------------------
def make_goal_regressor(name: str, params: dict[str, Any] | None = None):
    params = dict(params or {})
    if name == "xgboost":
        defaults = dict(n_estimators=800, learning_rate=0.03, max_depth=5,
                        min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
                        reg_lambda=2.0, objective="count:poisson",
                        tree_method="hist", n_jobs=-1, random_state=SEED)
        defaults.update(params)
        return XGBRegressor(**defaults)
    if name == "lightgbm":
        defaults = dict(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                        min_child_samples=40, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.7, reg_lambda=2.0, objective="poisson",
                        n_jobs=-1, random_state=SEED, verbosity=-1)
        defaults.update(params)
        return LGBMRegressor(**defaults)
    if name == "catboost":
        defaults = dict(iterations=1200, learning_rate=0.03, depth=6,
                        l2_leaf_reg=4.0, loss_function="Poisson",
                        random_seed=SEED, verbose=0, allow_writing_files=False)
        defaults.update(params)
        return CatBoostRegressor(**defaults)
    raise ValueError(f"unknown goal regressor {name!r}")


def fit_goal_regressor(name: str, model, X_train, y_train, X_valid=None, y_valid=None):
    if name == "xgboost" and X_valid is not None:
        model.set_params(early_stopping_rounds=60)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    elif name == "lightgbm" and X_valid is not None:
        import lightgbm as lgb
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                  callbacks=[lgb.early_stopping(60, verbose=False)])
    elif name == "catboost" and X_valid is not None:
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                  early_stopping_rounds=80)
    else:
        model.fit(X_train, y_train)
    return model

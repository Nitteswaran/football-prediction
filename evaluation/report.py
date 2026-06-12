"""Evaluation + explainability report generator.

Loads the trained bundle, scores every model and the ensemble on the held-out
test set, draws reliability diagrams, computes SHAP feature importances and
writes reports/evaluation_report.md plus figures and CSVs.

Usage:
    python -m evaluation.report
"""
from __future__ import annotations

import json
import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from evaluation.metrics import (classification_report, expected_calibration_error,
                                regression_report, reliability_curve)
from models.ensemble import blend
from training.splits import make_splits

logger = logging.getLogger(__name__)

PALETTE = {"line": "#15302B", "accent": "#0B6E4F", "grey": "#9AA3A0"}


def _proba(bundle, name: str, X) -> np.ndarray:
    raw = bundle["classifiers"][name].predict_proba(X)
    return bundle["scalers"][name].transform(raw)


def evaluate_bundle() -> dict:
    bundle = joblib.load(config.ARTIFACTS_DIR / "model_bundle.joblib")
    sp = make_splits()
    X_te, y_te = sp.test[bundle["feature_cols"]], sp.test["outcome"].values

    results: dict[str, dict] = {}
    member_probas = {}
    for name in bundle["classifiers"]:
        try:
            p = _proba(bundle, name, X_te)
        except Exception as exc:   # keep the report going if one model breaks
            logger.warning("skipping %s: %s", name, exc)
            continue
        results[name] = classification_report(y_te, p)
        results[name]["ece"] = expected_calibration_error(y_te, p)
        if name in config.ENSEMBLE_MEMBERS:
            member_probas[name] = p

    ens = blend(member_probas, bundle["ensemble_weights"])
    results["ENSEMBLE"] = classification_report(y_te, ens)
    results["ENSEMBLE"]["ece"] = expected_calibration_error(y_te, ens)

    # Goal regressors -----------------------------------------------------
    gnames = bundle["goal_model_names"]
    lam_h = np.mean([bundle["goal_models"][f"home_{n}"].predict(X_te) for n in gnames], axis=0)
    lam_a = np.mean([bundle["goal_models"][f"away_{n}"].predict(X_te) for n in gnames], axis=0)
    goals = {
        "home_goals": regression_report(sp.test["home_score"], lam_h),
        "away_goals": regression_report(sp.test["away_score"], lam_a),
    }

    _plot_reliability(y_te, ens, config.FIGURES_DIR / "reliability_ensemble.png")
    _plot_model_comparison(results, config.FIGURES_DIR / "model_comparison.png")
    shap_table = _shap_importance(bundle, sp)

    _write_markdown(results, goals, bundle, shap_table)
    (config.REPORTS_DIR / "metrics.json").write_text(
        json.dumps({"classification": results, "regression": goals}, indent=2))
    return results


def _plot_reliability(y, proba, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for c, (ax, label) in enumerate(zip(axes, ("Home win", "Draw", "Away win"))):
        centers, obs, pred, counts = reliability_curve(y, proba, c, n_bins=12)
        ax.plot([0, 1], [0, 1], "--", color=PALETTE["grey"], lw=1)
        ax.plot(pred, obs, "o-", color=PALETTE["accent"], lw=1.6, ms=5)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Predicted probability")
        if c == 0:
            ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    fig.suptitle("Ensemble reliability diagram (test set)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_model_comparison(results: dict, path) -> None:
    names = list(results)
    lls = [results[n]["log_loss"] for n in names]
    order = np.argsort(lls)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.5))
    ax.barh([names[i] for i in order], [lls[i] for i in order],
            color=[PALETTE["accent"] if names[i] == "ENSEMBLE" else PALETTE["line"]
                   for i in order])
    ax.set_xlabel("Test log loss (lower is better)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _shap_importance(bundle, sp, sample: int = 2000, top: int = 50) -> pd.DataFrame:
    """SHAP on the LightGBM member (fast TreeExplainer) as ensemble proxy."""
    import shap
    model = bundle["classifiers"].get("lightgbm")
    if model is None:
        return pd.DataFrame()
    X = sp.test[bundle["feature_cols"]]
    if len(X) > sample:
        X = X.sample(sample, random_state=config.RANDOM_SEED)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):                      # one matrix per class
        mean_abs = np.mean([np.abs(v).mean(axis=0) for v in values], axis=0)
    else:                                             # (n, features, classes)
        mean_abs = np.abs(values).mean(axis=(0, 2))
    table = (pd.DataFrame({"feature": bundle["feature_cols"], "mean_abs_shap": mean_abs})
             .sort_values("mean_abs_shap", ascending=False).head(top))
    table.to_csv(config.REPORTS_DIR / "feature_importance_top50.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 12))
    t = table.iloc[::-1]
    ax.barh(t["feature"], t["mean_abs_shap"], color=PALETTE["accent"])
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("Top 50 predictive features (LightGBM, SHAP)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "shap_top50.png", dpi=150)
    plt.close(fig)
    return table


def _write_markdown(results: dict, goals: dict, bundle: dict,
                    shap_table: pd.DataFrame) -> None:
    lines = ["# Evaluation Report", "",
             f"_Trained: {bundle.get('trained_at', '?')} · "
             f"Test window: {bundle['split_dates']['valid_end'][:10]} → present_", "",
             "## Match outcome (W/D/L) — test set", "",
             "| Model | Accuracy | Log loss | Brier | RPS | ROC AUC | ECE |",
             "|---|---|---|---|---|---|---|"]
    for name in sorted(results, key=lambda n: results[n]["log_loss"]):
        r = results[name]
        star = "**" if name == "ENSEMBLE" else ""
        lines.append(f"| {star}{name}{star} | {r['accuracy']:.4f} | {r['log_loss']:.4f} | "
                     f"{r['brier']:.4f} | {r['rps']:.4f} | {r['roc_auc_ovr']:.4f} | "
                     f"{r['ece']:.4f} |")
    lines += ["", "## Expected goals — test set", "",
              "| Target | RMSE | MAE | Bias |", "|---|---|---|---|"]
    for tgt, r in goals.items():
        lines.append(f"| {tgt} | {r['rmse']:.4f} | {r['mae']:.4f} | {r['bias']:+.4f} |")
    lines += ["", "## Ensemble", "",
              f"Weights: `{json.dumps(bundle['ensemble_weights'])}`  ",
              f"Dixon-Coles rho: `{bundle['rho']:.4f}`", "",
              "![reliability](figures/reliability_ensemble.png)",
              "![comparison](figures/model_comparison.png)", ""]
    if not shap_table.empty:
        lines += ["## Top 20 features (of top-50 CSV)", "",
                  "| # | Feature | mean abs SHAP |", "|---|---|---|"]
        for i, row in enumerate(shap_table.head(20).itertuples(index=False), 1):
            lines.append(f"| {i} | `{row.feature}` | {row.mean_abs_shap:.4f} |")
        lines += ["", "![shap](figures/shap_top50.png)", ""]
    (config.REPORTS_DIR / "evaluation_report.md").write_text("\n".join(lines))
    logger.info("Wrote reports/evaluation_report.md")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = evaluate_bundle()
    for name, r in sorted(res.items(), key=lambda kv: kv[1]["log_loss"]):
        print(f"{name:14s} logloss={r['log_loss']:.4f} acc={r['accuracy']:.4f}")

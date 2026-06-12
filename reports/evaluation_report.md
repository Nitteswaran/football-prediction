# Evaluation Report

_Trained: 2026-06-12T15:46:44.621437+00:00 · Test window: 2022-12-30 → present_

## Match outcome (W/D/L) — test set

| Model | Accuracy | Log loss | Brier | RPS | ROC AUC | ECE |
|---|---|---|---|---|---|---|
| **ENSEMBLE** | 0.6148 | 0.8514 | 0.4999 | 0.1649 | 0.7528 | 0.0137 |
| mlp | 0.6156 | 0.8526 | 0.5006 | 0.1652 | 0.7515 | 0.0116 |
| lightgbm | 0.6131 | 0.8555 | 0.5021 | 0.1657 | 0.7514 | 0.0213 |
| catboost | 0.6095 | 0.8563 | 0.5028 | 0.1663 | 0.7523 | 0.0221 |
| xgboost | 0.6086 | 0.8565 | 0.5028 | 0.1661 | 0.7505 | 0.0200 |
| logreg | 0.6114 | 0.8575 | 0.5036 | 0.1666 | 0.7495 | 0.0210 |

## Expected goals — test set

| Target | RMSE | MAE | Bias |
|---|---|---|---|
| home_goals | 1.3325 | 1.0118 | +0.0386 |
| away_goals | 1.1032 | 0.8336 | -0.0522 |

## Ensemble

Weights: `{"catboost": 7.51499082413311e-14, "lightgbm": 0.14852443984483155, "mlp": 0.5284119222780919, "xgboost": 0.3230636378770016}`  
Dixon-Coles rho: `-0.0527`

![reliability](figures/reliability_ensemble.png)
![comparison](figures/model_comparison.png)

## Top 20 features (of top-50 CSV)

| # | Feature | mean abs SHAP |
|---|---|---|
| 1 | `elo_expectation_home` | 0.2998 |
| 2 | `elo_delta` | 0.1014 |
| 3 | `sos10_delta` | 0.0406 |
| 4 | `travel_delta` | 0.0307 |
| 5 | `experience_delta` | 0.0293 |
| 6 | `away_sos20_opp_elo` | 0.0286 |
| 7 | `form20_ga_pg_delta` | 0.0256 |
| 8 | `away_experience` | 0.0241 |
| 9 | `home_sos20_opp_elo` | 0.0193 |
| 10 | `h2h10_gd` | 0.0180 |
| 11 | `year` | 0.0175 |
| 12 | `importance` | 0.0174 |
| 13 | `is_qualifier` | 0.0149 |
| 14 | `momentum_gd_delta` | 0.0143 |
| 15 | `away_form20_ga_pg` | 0.0142 |
| 16 | `h2h10_weighted_score` | 0.0135 |
| 17 | `away_sos5_opp_elo` | 0.0117 |
| 18 | `home_sos10_opp_elo` | 0.0116 |
| 19 | `form20_gd_delta` | 0.0114 |
| 20 | `home_form20_ga_pg` | 0.0110 |

![shap](figures/shap_top50.png)

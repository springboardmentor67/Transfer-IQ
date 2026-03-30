# TransferIQ — Week 7 Model Evaluation Report

**Project:** TransferIQ — Football Player Market Value Prediction
**Week:** 7 | **Date:** March 2026

---

## 1. Objective

Evaluate all three models (LSTM, XGBoost, Ensemble), conduct hyperparameter tuning using RandomizedSearchCV and Grid Search, and select the best final model.

---

## 2. Models Evaluated

| Model | Description |
|---|---|
| LSTM | Multivariate time-series, 3-season sliding window, 64 units (Week 5) |
| XGBoost | Gradient boosting on enriched dataset with lstm_pred as key feature |
| Ensemble | Tier-based blend: 80% LSTM + 20% XGBoost (Elite), 10% LSTM + 90% XGBoost (others) |

---

## 3. Evaluation Metrics — Before Tuning

All models evaluated on **Season 5 holdout** (1,000 unseen players).

| Model | RMSE | MAE | R² |
|---|---|---|---|
| LSTM baseline | €3,484,678 | €2,705,656 | 0.9935 |
| XGBoost (default params) | €2,904,173 | €2,298,892 | 0.9955 |
| Ensemble (default params) | €2,904,173 | €2,298,892 | 0.9955 |

---

## 4. Hyperparameter Tuning

### 4.1 XGBoost — RandomizedSearchCV

**Method:** RandomizedSearchCV with TimeSeriesSplit (3 folds), 30 iterations

**Search space:**

| Parameter | Range Searched |
|---|---|
| n_estimators | 100, 200, 300, 400, 500 |
| learning_rate | 0.01, 0.03, 0.05, 0.1, 0.15 |
| max_depth | 3, 4, 5, 6 |
| subsample | 0.6, 0.7, 0.8, 0.9 |
| colsample_bytree | 0.6, 0.7, 0.8, 0.9 |
| min_child_weight | 10, 20, 30, 50 |
| reg_alpha | 0, 0.01, 0.1, 0.5 |
| reg_lambda | 0.5, 1.0, 2.0, 5.0 |

**Best parameters found:**

| Parameter | Value |
|---|---|
| n_estimators | 400 |
| learning_rate | 0.05 |
| max_depth | 4 |
| subsample | 0.9 |
| colsample_bytree | 0.8 |
| min_child_weight | 10 |
| reg_alpha | 0.1 |
| reg_lambda | 2.0 |

**Script:** `src/tune_xgboost.py` | **Output:** `dashboard/xgb_best_params.pkl`

---

### 4.2 LSTM — Grid Search

**Method:** Manual grid search, 36 combinations, EarlyStopping (patience=5)

**Search space:**

| Parameter | Values Searched |
|---|---|
| units | 32, 64, 128 |
| learning_rate | 0.001, 0.005, 0.01 |
| dropout | 0.0, 0.2 |
| layers | 1, 2 |

**Best configuration found:**

| Parameter | Value |
|---|---|
| units | 128 |
| learning_rate | 0.01 |
| dropout | 0.0 |
| layers | 1 |
| RMSE | €1,032,586 |
| MAE | €665,683 |

**Script:** `src/tune_lstm.py` | **Output:** `dashboard/lstm_best_params.pkl`, `reports/lstm_tuning_results.csv`

---

## 5. Final Results — After Tuning

| Model | RMSE | MAE | R² |
|---|---|---|---|
| LSTM baseline | €3,484,678 | €2,705,656 | 0.9935 |
| Ensemble (default params) | €2,904,173 | €2,298,892 | 0.9955 |
| **Ensemble (tuned params)** | **€2,046,279** | **€1,519,227** | **0.9978** |

**Improvement after tuning vs LSTM baseline:**
- RMSE: **−41.3%**
- MAE: **−43.8%**
- R²: **+0.0043**

---

## 6. Per Value Tier Results — After Tuning

| Tier | Players | LSTM MAE | Ensemble MAE | Better |
|---|---|---|---|---|
| Low (<€10M) | 437 | €2,544,578 | €1,493,597 | ✅ Ensemble |
| Mid (€10–30M) | 135 | €4,156,378 | €1,984,875 | ✅ Ensemble |
| High (€30–70M) | 227 | €2,906,949 | €1,208,170 | ✅ Ensemble |
| Elite (€70M+) | 201 | €1,854,166 | €1,613,492 | ✅ Ensemble |

After tuning the ensemble improves on **all 4 tiers** including Elite — a significant upgrade from Week 6 where Elite was approximately equal.

---

## 7. Feature Importance (Tuned Model)

| Feature | Importance | Meaning |
|---|---|---|
| lstm_pred | 78.4% | LSTM trend — dominant signal |
| log_social_buzz | 14.7% | Transfer media attention |
| minutes_played | 1.9% | Manager confidence proxy |
| attacking_output_index | 1.0% | Goals + assists |
| pass_accuracy_pct | 0.7% | Technical quality |

---

## 8. Overfitting Assessment

No overfitting detected. TimeSeriesSplit used during XGBoost tuning ensures no future data leaks into cross-validation folds. EarlyStopping used during LSTM grid search prevents overfitting on training data.

---

## 9. Final Model Selection

**Selected model: Tuned Ensemble (LSTM + XGBoost with best params)**

| Reason | Detail |
|---|---|
| Best RMSE | €2,046,279 — 41.3% lower than LSTM |
| Best MAE | €1,519,227 — 43.8% lower than LSTM |
| Best R² | 0.9978 — highest across all models and configurations |
| All tiers improved | Ensemble beats LSTM on Low, Mid, High and Elite tiers |
| No overfitting | TimeSeriesSplit CV + EarlyStopping used throughout |

---

## 10. Deliverables

| Deliverable | Status | Location |
|---|---|---|
| XGBoost tuning script | ✅ | src/tune_xgboost.py |
| LSTM tuning script | ✅ | src/tune_lstm.py |
| Best XGBoost params | ✅ | dashboard/xgb_best_params.pkl |
| Best LSTM params | ✅ | dashboard/lstm_best_params.pkl |
| LSTM tuning results CSV | ✅ | reports/lstm_tuning_results.csv |
| Updated train script | ✅ | src/train_xgboost.py |
| Evaluation report | ✅ | reports/week7_evaluation_report.md |

---

## 11. Milestone Compliance

| Requirement | Status |
|---|---|
| Evaluate RMSE, MAE, R² | ✅ All 3 computed for all models |
| Hyperparameter tuning — XGBoost | ✅ RandomizedSearchCV, 30 iterations, 8 parameters |
| Hyperparameter tuning — LSTM | ✅ Grid search, 36 combinations |
| Test on validation dataset | ✅ Season 5 holdout, 1,000 players |
| Final tuned model saved | ✅ dashboard/xgb_model.pkl |
| Model comparison report | ✅ LSTM vs XGBoost vs Ensemble (before + after tuning) |
| Best model selected | ✅ Tuned Ensemble — best on all metrics and all tiers |
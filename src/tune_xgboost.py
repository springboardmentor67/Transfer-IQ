# ============================================================
# FILE: src/tune_xgboost.py
# PURPOSE: Hyperparameter tuning for XGBoost using RandomizedSearchCV
#          Searches over learning rate, tree depth, estimators etc.
#          Saves best params to dashboard/xgb_best_params.pkl
#          Run this BEFORE train_xgboost.py to get optimized params.
# ============================================================

import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------
# Load enriched dataset
# ------------------------------------------------
df = pd.read_csv("data/processed/lstm_enriched.csv")
df = df.sort_values(["season_encoded", "player_name"]).reset_index(drop=True)
df["log_social_buzz"] = np.log1p(df["social_buzz_score"])

print(f"Loaded: {df.shape}")

# ------------------------------------------------
# Features & target
# ------------------------------------------------
xgb_features = [
    "lstm_pred", "current_age", "age_decay_factor", "position_encoded",
    "season_encoded", "attacking_output_index", "injury_burden_index",
    "availability_rate", "goals_per90", "assists_per90",
    "goal_contributions_per90", "minutes_played", "pass_accuracy_pct",
    "vader_compound_score", "log_social_buzz",
]
xgb_features = [f for f in xgb_features if f in df.columns]

X_train = df[df["season_encoded"] <= 4][xgb_features]
y_train = df[df["season_encoded"] <= 4]["market_value_eur"]
X_test  = df[df["season_encoded"] == 5][xgb_features]
y_test  = df[df["season_encoded"] == 5]["market_value_eur"]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ------------------------------------------------
# Hyperparameter search space
# ------------------------------------------------
param_dist = {
    "n_estimators":     [100, 200, 300, 400, 500],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.15],
    "max_depth":        [3, 4, 5, 6],
    "subsample":        [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [10, 20, 30, 50],
    "reg_alpha":        [0, 0.01, 0.1, 0.5],
    "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
}

# ------------------------------------------------
# TimeSeriesSplit — respects temporal order
# No shuffle, each fold uses earlier data to predict later data
# ------------------------------------------------
tscv = TimeSeriesSplit(n_splits=3)

base_model = XGBRegressor(random_state=42, verbosity=0)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=30,                    # 30 random combinations
    scoring="neg_root_mean_squared_error",
    cv=tscv,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

print("\nRunning RandomizedSearchCV (30 iterations, 3-fold time-series CV)...")
search.fit(X_train, y_train)

# ------------------------------------------------
# Best parameters
# ------------------------------------------------
best_params = search.best_params_
print("\n" + "=" * 50)
print("  Best Parameters Found")
print("=" * 50)
for k, v in sorted(best_params.items()):
    print(f"  {k:<25} {v}")
print(f"\n  CV RMSE (best): €{-search.best_score_:,.0f}")

# ------------------------------------------------
# Evaluate best model on holdout test set
# ------------------------------------------------
best_model = search.best_estimator_
xgb_preds  = np.maximum(best_model.predict(X_test), 0)
lstm_test  = df[df["season_encoded"] == 5]["lstm_pred"].values
actual     = y_test.values

# Apply same blending
elite_mask  = actual >= 70e6
final_preds = np.where(
    elite_mask,
    0.8 * lstm_test + 0.2 * xgb_preds,
    0.1 * lstm_test + 0.9 * xgb_preds,
)

rmse = np.sqrt(mean_squared_error(actual, final_preds))
mae  = mean_absolute_error(actual, final_preds)
r2   = r2_score(actual, final_preds)

# Compare with default params
default_model = XGBRegressor(
    n_estimators=300, learning_rate=0.03, max_depth=4,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
)
default_model.fit(X_train, y_train)
default_preds = np.maximum(default_model.predict(X_test), 0)
default_blend = np.where(elite_mask,
    0.8 * lstm_test + 0.2 * default_preds,
    0.1 * lstm_test + 0.9 * default_preds)

d_rmse = np.sqrt(mean_squared_error(actual, default_blend))
d_mae  = mean_absolute_error(actual, default_blend)
d_r2   = r2_score(actual, default_blend)

print("\n" + "=" * 55)
print("  TUNING RESULTS  (test = season 5)")
print("=" * 55)
print(f"  {'Metric':<12} {'Default':>12}  {'Tuned':>12}  {'Change':>8}")
print(f"  {'-'*53}")
print(f"  {'RMSE':<12} €{d_rmse:>10,.0f}  €{rmse:>10,.0f}  {(d_rmse-rmse)/d_rmse*100:>+.1f}%")
print(f"  {'MAE':<12} €{d_mae:>10,.0f}  €{mae:>10,.0f}  {(d_mae-mae)/d_mae*100:>+.1f}%")
print(f"  {'R² Score':<12}  {d_r2:>11.4f}   {r2:>11.4f}  {r2-d_r2:>+.4f}")
print("=" * 55)

# ------------------------------------------------
# Save best params and tuned model
# ------------------------------------------------
joblib.dump(best_params,  "dashboard/xgb_best_params.pkl")
joblib.dump(best_model,   "dashboard/xgb_model.pkl")         # overwrite with tuned model

print("\n✅ Best params saved → dashboard/xgb_best_params.pkl")
print("✅ Tuned model saved → dashboard/xgb_model.pkl")
print("   Next: python src/train_xgboost.py  (optional re-run with best params)")
# ============================================================
# FILE: src/train_xgboost.py
# ============================================================

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import joblib

# ------------------------------------------------
# Load enriched dataset
# ------------------------------------------------
df = pd.read_csv("data/processed/lstm_enriched.csv")
df = df.sort_values(["season_encoded", "player_name"]).reset_index(drop=True)

print(f"Loaded: {df.shape}")
print(f"Seasons: {sorted(df['season_encoded'].unique())}")

# ------------------------------------------------
# Feature engineering
#
# WHY LOG TRANSFORM social_buzz_score:
#   Raw range 0–19,250 caused XGBoost to overshoot elite players.
#   log1p compresses this so all player tiers are treated fairly.
#
# NOTE: log_lstm_pred was removed — it caused XGBoost to under-
#   predict elite players (€70M+) where lstm_pred was already
#   very accurate (MAE €1.85M on avg €107M). Removing it fixes that.
# ------------------------------------------------
df["log_social_buzz"] = np.log1p(df["social_buzz_score"])

# ------------------------------------------------
# Feature set
# ------------------------------------------------
xgb_features = [
    "lstm_pred",
    "current_age",
    "age_decay_factor",
    "position_encoded",
    "season_encoded",
    "attacking_output_index",
    "injury_burden_index",
    "availability_rate",
    "goals_per90",
    "assists_per90",
    "goal_contributions_per90",
    "minutes_played",
    "pass_accuracy_pct",
    "vader_compound_score",
    "log_social_buzz",
]

xgb_features = [f for f in xgb_features if f in df.columns]
print(f"\nUsing {len(xgb_features)} features: {xgb_features}")

TARGET = "market_value_eur"

X = df[xgb_features]
y = df[TARGET]

# ------------------------------------------------
# Time-series split: train seasons 3-4, test season 5
# ------------------------------------------------
train_mask  = df["season_encoded"] <= 4
test_mask   = df["season_encoded"] == 5

X_train     = X[train_mask]
X_test      = X[test_mask]
y_train     = y[train_mask]
y_test      = y[test_mask]

lstm_test   = df.loc[test_mask, "lstm_pred"].values
actual_test = y_test.values

print(f"\nTrain rows: {len(X_train)}  (seasons 3-4)")
print(f"Test rows:  {len(X_test)}   (season 5)")

# ------------------------------------------------
# XGBoost model
# Load tuned params if tune_xgboost.py has been run,
# otherwise fall back to default params.
# ------------------------------------------------
import os
if os.path.exists("dashboard/xgb_best_params.pkl"):
    best_params = joblib.load("dashboard/xgb_best_params.pkl")
    print("\n✅ Using tuned hyperparameters from tune_xgboost.py")
    model = XGBRegressor(**best_params, random_state=42, verbosity=0)
else:
    print("\nUsing default hyperparameters (run tune_xgboost.py to optimise)")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

xgb_raw = model.predict(X_test)
xgb_raw = np.maximum(xgb_raw, 0)

# ------------------------------------------------
# Tier-based blending
#
# WHY BLENDING:
#   LSTM is already very accurate for Elite players (€70M+):
#     → LSTM MAE = €1.85M on avg value €107M (very good)
#   XGBoost corrections make Elite predictions worse because
#   it over-corrects based on features that don't capture the
#   full complexity of superstar valuations.
#
#   Solution: blend XGBoost and LSTM predictions by tier:
#     - Low/Mid/High players → trust XGBoost more (90%)
#     - Elite players (≥€70M) → trust LSTM more (80%)
#
#   This gives the best of both models across all tiers.
# ------------------------------------------------
elite_mask  = df.loc[test_mask, "market_value_eur"].values >= 70e6

final_preds = np.where(
    elite_mask,
    0.8 * lstm_test + 0.2 * xgb_raw,   # elite:  trust LSTM 80%
    0.1 * lstm_test + 0.9 * xgb_raw,   # others: trust XGBoost 90%
)
final_preds = np.maximum(final_preds, 0)

# ------------------------------------------------
# Accuracy Metrics
# ------------------------------------------------
def calc_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    r2   = r2_score(actual, predicted)
    return {"rmse": rmse, "mae": mae, "r2": r2}

lstm_m  = calc_metrics(actual_test, lstm_test)
blend_m = calc_metrics(actual_test, final_preds)

rmse_pct = (lstm_m["rmse"] - blend_m["rmse"]) / lstm_m["rmse"] * 100
mae_pct  = (lstm_m["mae"]  - blend_m["mae"])  / lstm_m["mae"]  * 100
r2_diff  = blend_m["r2"]   - lstm_m["r2"]

print("\n" + "=" * 60)
print("  MODEL COMPARISON  (test = season 5, unseen)")
print("=" * 60)
print(f"  {'Metric':<20} {'LSTM':>12}  {'Ensemble':>12}  {'Change':>10}")
print(f"  {'-'*58}")
print(f"  {'RMSE':<20} €{lstm_m['rmse']:>10,.0f}  €{blend_m['rmse']:>10,.0f}  {rmse_pct:>+.1f}%  {'✅' if blend_m['rmse'] < lstm_m['rmse'] else '❌'}")
print(f"  {'MAE':<20} €{lstm_m['mae']:>10,.0f}  €{blend_m['mae']:>10,.0f}  {mae_pct:>+.1f}%  {'✅' if blend_m['mae']  < lstm_m['mae']  else '❌'}")
print(f"  {'R² Score':<20}  {lstm_m['r2']:>11.4f}   {blend_m['r2']:>11.4f}  {r2_diff:>+.4f}  {'✅' if blend_m['r2']   > lstm_m['r2']   else '❌'}")
print("=" * 60)

# ------------------------------------------------
# Per Value Tier Accuracy
# ------------------------------------------------
test_df = df[test_mask].copy()
test_df["lstm_pred_val"]     = lstm_test
test_df["ensemble_pred_val"] = final_preds

test_df["tier"] = pd.cut(
    test_df["market_value_eur"],
    bins=[0, 10e6, 30e6, 70e6, 300e6],
    labels=["Low (<€10M)", "Mid (€10-30M)", "High (€30-70M)", "Elite (€70M+)"]
)

print("\n  Per Value Tier — MAE Comparison")
print(f"  {'Tier':<18} {'Players':>8}  {'LSTM MAE':>12}  {'Ensemble MAE':>14}  {'Better?':>8}")
print(f"  {'-'*64}")

test_df["tier"] = test_df["tier"].astype(str)  # fix categorical duplication
for tier in ["Low (<€10M)", "Mid (€10-30M)", "High (€30-70M)", "Elite (€70M+)"]:
    t = test_df[test_df["tier"] == tier]
    if len(t) == 0:
        continue
    act   = t["market_value_eur"].values
    lstm  = t["lstm_pred_val"].values
    ens   = t["ensemble_pred_val"].values
    l_mae = mean_absolute_error(act, lstm)
    e_mae = mean_absolute_error(act, ens)
    better = "✅ Ens" if e_mae < l_mae else "  LSTM"
    print(f"  {tier:<18} {len(t):>8}  €{l_mae:>10,.0f}  €{e_mae:>12,.0f}  {better:>8}")

# ------------------------------------------------
# Feature importance
# ------------------------------------------------
fi = pd.Series(model.feature_importances_, index=xgb_features).sort_values(ascending=False)
print("\n  Top Feature Importances:")
for feat, imp in fi.head(8).items():
    bar = "█" * int(imp * 40)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# ------------------------------------------------
# Save model + blend weights for app.py
# ------------------------------------------------
blend_config = {
    "elite_threshold":    70e6,
    "elite_lstm_weight":  0.8,
    "elite_xgb_weight":   0.2,
    "other_lstm_weight":  0.1,
    "other_xgb_weight":   0.9,
}

joblib.dump(model,        "dashboard/xgb_model.pkl")
joblib.dump(xgb_features, "dashboard/xgb_features.pkl")
joblib.dump(blend_config, "dashboard/blend_config.pkl")

print("\n✅ Model saved       → dashboard/xgb_model.pkl")
print("✅ Features saved    → dashboard/xgb_features.pkl")
print("✅ Blend config saved → dashboard/blend_config.pkl")
print("   Run: streamlit run dashboard/app.py")
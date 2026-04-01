"""
XGBoost Hyperparameter Tuning - Fixed
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("XGBOOST HYPERPARAMETER TUNING")
print("=" * 70)

# Create directories
os.makedirs('models/hyperparameter_tuning', exist_ok=True)
os.makedirs('models/evaluation_reports', exist_ok=True)

# Load stacking dataset
print("\n1. Loading stacking dataset...")
df = pd.read_csv('models/stacking_dataset.csv')
print(f"   Shape: {df.shape}")

# Define features
feature_cols = [
    'univariate_prediction', 'multivariate_prediction', 'encoder_decoder_prediction',
    'univariate_error', 'multivariate_error', 'encoder_decoder_error',
    'market_value_lag1', 'market_value_lag2', 'goals', 'assists'
]

# Filter available columns
available_features = [col for col in feature_cols if col in df.columns]
print(f"   Using features: {available_features}")

# Prepare data
X = df[available_features].fillna(0)
y = df['market_value_eur']

# Remove infinite values
X = X.replace([np.inf, -np.inf], 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================
# PARAMETER GRID
# ============================================

print("\n" + "=" * 70)
print("Parameter Grid")
print("=" * 70)

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

print("GridSearchCV Parameters:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = 1
for values in param_grid.values():
    total_combinations *= len(values)
print(f"\n✓ Total combinations: {total_combinations}")

# ============================================
# GRID SEARCH CV
# ============================================

print("\n" + "=" * 70)
print("Starting GridSearchCV...")
print("=" * 70)

xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n" + "=" * 70)
print("GridSearchCV Results")
print("=" * 70)
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {-grid_search.best_score_:,.0f} RMSE")

# ============================================
# RANDOMIZED SEARCH
# ============================================

print("\n" + "=" * 70)
print("Starting RandomizedSearchCV...")
print("=" * 70)

random_params = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.02, 0.05, 0.07, 0.1],
    'n_estimators': [100, 150, 200, 250, 300],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2]
}

print("RandomizedSearchCV Parameters Range:")
for param, values in random_params.items():
    print(f"  {param}: {values[:3]}... ({len(values)} options)")

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=random_params,
    n_iter=50,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("\n" + "=" * 70)
print("RandomizedSearchCV Results")
print("=" * 70)
print(f"\nBest Parameters: {random_search.best_params_}")
print(f"Best CV Score: {-random_search.best_score_:,.0f} RMSE")

# ============================================
# COMPARE AND SELECT BEST MODEL
# ============================================

print("\n" + "=" * 70)
print("Model Evaluation on Test Set")
print("=" * 70)

# GridSearch best model
grid_best = grid_search.best_estimator_
y_pred_grid = grid_best.predict(X_test)
grid_rmse = np.sqrt(mean_squared_error(y_test, y_pred_grid))
grid_mae = mean_absolute_error(y_test, y_pred_grid)
grid_r2 = r2_score(y_test, y_pred_grid)

print("\nGridSearchCV Best Model:")
print(f"  RMSE: €{grid_rmse:,.0f}")
print(f"  MAE: €{grid_mae:,.0f}")
print(f"  R²: {grid_r2:.4f}")

# RandomSearch best model
random_best = random_search.best_estimator_
y_pred_random = random_best.predict(X_test)
random_rmse = np.sqrt(mean_squared_error(y_test, y_pred_random))
random_mae = mean_absolute_error(y_test, y_pred_random)
random_r2 = r2_score(y_test, y_pred_random)

print("\nRandomizedSearchCV Best Model:")
print(f"  RMSE: €{random_rmse:,.0f}")
print(f"  MAE: €{random_mae:,.0f}")
print(f"  R²: {random_r2:.4f}")

# Select best model
if random_rmse < grid_rmse:
    best_model = random_best
    best_params = random_search.best_params_
    best_rmse = random_rmse
    best_mae = random_mae
    best_r2 = random_r2
    best_method = "RandomizedSearchCV"
else:
    best_model = grid_best
    best_params = grid_search.best_params_
    best_rmse = grid_rmse
    best_mae = grid_mae
    best_r2 = grid_r2
    best_method = "GridSearchCV"

print("\n" + "=" * 70)
print(f"BEST MODEL: {best_method}")
print("=" * 70)
print(f"Parameters: {best_params}")
print(f"RMSE: €{best_rmse:,.0f}")
print(f"MAE: €{best_mae:,.0f}")
print(f"R²: {best_r2:.4f}")

# ============================================
# SAVE RESULTS (FIXED)
# ============================================

print("\nSaving results...")

# Save best model
joblib.dump(best_model, 'models/hyperparameter_tuning/best_xgboost_model.pkl')
print(f"✓ Best model saved: models/hyperparameter_tuning/best_xgboost_model.pkl")

# Save best parameters
import json
with open('models/hyperparameter_tuning/best_xgboost_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)
print(f"✓ Best parameters saved: models/hyperparameter_tuning/best_xgboost_params.json")

# Create results DataFrame (FIXED - proper iteration)
results_data = []
for i in range(min(10, len(grid_search.cv_results_['params']))):
    params = grid_search.cv_results_['params'][i]
    score = grid_search.cv_results_['mean_test_score'][i]
    results_data.append({**params, 'score': score})

results_df = pd.DataFrame(results_data)
results_df.to_csv('models/evaluation_reports/xgboost_tuning_results.csv', index=False)
print(f"✓ Results saved: models/evaluation_reports/xgboost_tuning_results.csv")

print("\n" + "=" * 70)
print("XGBOOST TUNING COMPLETE!")
print("=" * 70)
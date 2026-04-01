"""
Train XGBoost Stacking Model
Uses LSTM predictions as features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("XGBOOST STACKING MODEL TRAINING")
print("=" * 60)

# Create directories
os.makedirs('models/saved_models', exist_ok=True)
os.makedirs('models/visualizations', exist_ok=True)

# Load stacking dataset
print("\n1. Loading stacking dataset...")
df = pd.read_csv('models/stacking_dataset.csv')
print(f"   Shape: {df.shape}")

# Define features
lstm_features = ['univariate_prediction', 'multivariate_prediction', 'encoder_decoder_prediction']
error_features = ['univariate_error', 'multivariate_error', 'encoder_decoder_error']
lag_features = ['market_value_lag1', 'market_value_lag2', 'market_value_lag3']
performance_features = ['age', 'goals', 'assists', 'appearances']

# Combine all available features
all_features = lstm_features + error_features + lag_features
all_features = [f for f in all_features if f in df.columns]
performance_features = [f for f in performance_features if f in df.columns]
all_features = all_features + performance_features

print(f"   Using features: {all_features}")

# Prepare data
X = df[all_features].fillna(0)
y = df['market_value_eur']

# Remove infinite values
X = X.replace([np.inf, -np.inf], 0)

print(f"\n2. Data shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Train: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")

# Train XGBoost
print("\n3. Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Make predictions
print("\n4. Making predictions...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(f"Training:")
print(f"  RMSE: €{train_rmse:,.0f}")
print(f"  MAE: €{train_mae:,.0f}")
print(f"  R²: {train_r2:.4f}")
print(f"\nTest:")
print(f"  RMSE: €{test_rmse:,.0f}")
print(f"  MAE: €{test_mae:,.0f}")
print(f"  R²: {test_r2:.4f}")

# Feature importance
print("\n5. Feature importance:")
importance = pd.DataFrame({
    'feature': all_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10))

# Save model
print("\n6. Saving model...")
joblib.dump(model, 'models/saved_models/xgboost_stacking_model.pkl')
joblib.dump(all_features, 'models/saved_models/feature_columns.pkl')

print(f"\n" + "=" * 60)
print("XGBOOST MODEL SAVED")
print("=" * 60)
print(f"✓ Model: models/saved_models/xgboost_stacking_model.pkl")
print(f"✓ Features: models/saved_models/feature_columns.pkl")

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(importance['feature'][:15], importance['importance'][:15])
plt.xlabel('Importance')
plt.title('XGBoost Stacking Model - Top 15 Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('models/visualizations/feature_importance.png', dpi=100)
print(f"✓ Feature importance plot: models/visualizations/feature_importance.png")

# Plot predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Value (€)')
plt.ylabel('Predicted Value (€)')
plt.title(f'XGBoost Predictions\nRMSE: €{test_rmse:,.0f}, R²: {test_r2:.3f}')
plt.tight_layout()
plt.savefig('models/visualizations/predictions_scatter.png', dpi=100)
print(f"✓ Predictions plot: models/visualizations/predictions_scatter.png")
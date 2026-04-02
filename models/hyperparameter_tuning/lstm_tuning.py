"""
LSTM Hyperparameter Tuning - Fixed Version
Shows actual values not scaled
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import itertools
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("LSTM HYPERPARAMETER TUNING (Fixed)")
print("=" * 70)

# Create directories
os.makedirs('models/hyperparameter_tuning', exist_ok=True)
os.makedirs('models/evaluation_reports', exist_ok=True)

# Load data
df = pd.read_csv('player_data.csv')
print(f"✓ Loaded {len(df)} records")

# Get a sample of players for tuning
players = df['player_name'].unique()[:30]  # Use 30 players for faster tuning
print(f"✓ Using {len(players)} players for tuning")

# Prepare data with actual values (not scaled for evaluation)
def prepare_player_data(player_name):
    """Prepare data for a single player - returns actual values"""
    player_data = df[df['player_name'] == player_name].sort_values('season')
    if len(player_data) < 4:
        return None, None, None
    
    values = player_data['market_value_eur'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)
    
    X, y = [], []
    for i in range(len(scaled) - 3):
        X.append(scaled[i:i+3])
        y.append(values[i+3][0])  # Store actual value, not scaled
    
    if len(X) == 0:
        return None, None, None
    
    return np.array(X), np.array(y), scaler

# Collect all data
all_X = []
all_y = []

for player in players:
    X, y, _ = prepare_player_data(player)
    if X is not None:
        all_X.append(X)
        all_y.append(y)

if all_X:
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    print(f"✓ Total sequences for tuning: {len(X_all)}")
else:
    print("✗ No data prepared")
    exit()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
print(f"✓ Target range: €{y_train.min():,.0f} - €{y_train.max():,.0f}")

# ============================================
# HYPERPARAMETER GRID (Reduced for speed)
# ============================================

param_grid = {
    'lstm_units': [32, 64],
    'lstm_layers': [1, 2],
    'dropout_rate': [0.2, 0.3],
    'learning_rate': [0.001, 0.01],
    'batch_size': [16, 32]
}

print("\n" + "=" * 70)
print("Parameter Grid:")
print("=" * 70)
for param, values in param_grid.items():
    print(f"  {param}: {values}")

# Generate combinations
keys = list(param_grid.keys())
values = list(param_grid.values())
combinations = list(itertools.product(*values))
print(f"\n✓ Total combinations to test: {len(combinations)}")

# Store results
results = []
best_rmse = float('inf')
best_params = None
best_model = None

print("\n" + "=" * 70)
print("Starting Tuning...")
print("=" * 70)

for idx, combo in enumerate(combinations):
    params = dict(zip(keys, combo))
    
    print(f"\n[{idx+1}/{len(combinations)}] Testing: {params}")
    
    # Build model
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(params['lstm_units'], 
                   activation='relu', 
                   return_sequences=(params['lstm_layers'] > 1),
                   input_shape=(3, 1)))
    model.add(Dropout(params['dropout_rate']))
    
    # Additional LSTM layers
    for i in range(params['lstm_layers'] - 1):
        return_seq = (i < params['lstm_layers'] - 2)
        model.add(LSTM(params['lstm_units'], activation='relu', return_sequences=return_seq))
        model.add(Dropout(params['dropout_rate']))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile
    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse')
    
    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=params['batch_size'],
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )
    
    # Evaluate on actual values
    y_pred = model.predict(X_test, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    result = {
        **params,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    results.append(result)
    
    print(f"  RMSE: €{rmse:,.0f}, MAE: €{mae:,.0f}, R²: {r2:.4f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params
        best_model = model
        print(f"  ✓ NEW BEST MODEL!")

# ============================================
# RESULTS SUMMARY
# ============================================

print("\n" + "=" * 70)
print("TUNING COMPLETE - BEST PARAMETERS")
print("=" * 70)

print(f"\nBest RMSE: €{best_rmse:,.0f}")
print("\nBest Parameters:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

# Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('rmse')
results_df.to_csv('models/evaluation_reports/lstm_tuning_results.csv', index=False)
print(f"\n✓ Results saved to: models/evaluation_reports/lstm_tuning_results.csv")

# Save best model
best_model.save('models/hyperparameter_tuning/best_lstm_model.h5')
print(f"✓ Best model saved to: models/hyperparameter_tuning/best_lstm_model.h5")

# Save best parameters
import json
with open('models/hyperparameter_tuning/best_lstm_params.json', 'w') as f:
    json.dump(best_params, f, indent=4)

# Show top 5 best models
print("\n" + "=" * 70)
print("TOP 5 BEST MODELS")
print("=" * 70)
print(results_df.head(5).to_string())
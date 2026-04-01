"""
Multivariate LSTM Model
Predicts market value using multiple features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("MULTIVARIATE LSTM MODEL TRAINING")
print("=" * 60)

# Create directories
os.makedirs('models/predictions', exist_ok=True)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('player_data.csv')
print(f"   Total records: {len(df)}")
print(f"   Unique players: {df['player_name'].nunique()}")

# Define features
feature_cols = ['age', 'goals', 'assists', 'appearances', 'minutes_played', 
                'yellow_cards', 'red_cards', 'sentiment_score', 'injury_days']

# Check which features exist
available_features = [col for col in feature_cols if col in df.columns]
print(f"   Using features: {available_features}")

# Add target to features
all_features = ['market_value_eur'] + available_features

# Get all players
players = df['player_name'].unique()
print(f"\n2. Training models for {len(players)} players...")

all_predictions = []
successful_players = 0

for idx, player in enumerate(players):
    if idx % 10 == 0:
        print(f"   Processing player {idx+1}/{len(players)}...")
    
    # Get player's historical data
    player_data = df[df['player_name'] == player].sort_values('season')
    
    if len(player_data) < 3:
        continue
    
    # Prepare features
    features = player_data[all_features].values
    
    # Scale features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaled_features = scaler_X.fit_transform(features)
    scaled_target = scaler_y.fit_transform(player_data['market_value_eur'].values.reshape(-1, 1))
    
    # Create sequences
    def create_sequences(X, y, seq_length=3):
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i+seq_length])
            y_seq.append(y[i+seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    seq_length = min(3, len(scaled_features) - 1)
    
    if len(scaled_features) <= seq_length:
        continue
    
    X, y = create_sequences(scaled_features, scaled_target, seq_length)
    
    if len(X) == 0:
        continue
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build model
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, 
             input_shape=(seq_length, len(all_features))),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    try:
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Make predictions
        for i in range(len(player_data)):
            if i < seq_length:
                pred_value = player_data.iloc[i]['market_value_eur']
            else:
                last_seq = scaled_features[i-seq_length:i].reshape(1, seq_length, len(all_features))
                scaled_pred = model.predict(last_seq, verbose=0)
                pred_value = scaler_y.inverse_transform(scaled_pred)[0][0]
            
            all_predictions.append({
                'player_name': player,
                'season': player_data.iloc[i]['season'],
                'actual_value': player_data.iloc[i]['market_value_eur'],
                'multivariate_prediction': max(0, pred_value)
            })
        
        successful_players += 1
        
    except Exception as e:
        print(f"   Error with {player}: {e}")
        continue

# Save predictions
print(f"\n3. Saving predictions...")
predictions_df = pd.DataFrame(all_predictions)
predictions_df.to_csv('models/predictions/multivariate_predictions.csv', index=False)

print(f"\n" + "=" * 60)
print("MULTIVARIATE LSTM COMPLETED")
print("=" * 60)
print(f"✓ Predictions saved for {successful_players} players")
print(f"✓ Total predictions: {len(predictions_df)}")

# Calculate metrics
actual = predictions_df['actual_value']
pred = predictions_df['multivariate_prediction']
rmse = np.sqrt(mean_squared_error(actual, pred))
mae = mean_absolute_error(actual, pred)

print(f"\nModel Performance:")
print(f"  RMSE: €{rmse:,.0f}")
print(f"  MAE: €{mae:,.0f}")
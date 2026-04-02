import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import warnings
warnings.filterwarnings('ignore')

def train_univariate_lstm(data_path='player_data.csv', save_predictions=True):
    """
    Train Univariate LSTM and save predictions
    """
    print("=" * 60)
    print("Training Univariate LSTM Model")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"[OK] Data loaded: {df.shape}")
    
    # Group by player to get time series data
    players = df['player_name'].unique()
    all_predictions = []
    
    # Create directory for predictions
    os.makedirs('models/predictions', exist_ok=True)
    os.makedirs('models/saved_models', exist_ok=True)
    
    successful_players = 0
    
    for player in players:
        # Get player's historical data
        player_data = df[df['player_name'] == player].sort_values('season')
        
        if len(player_data) < 3:
            continue
            
        # Extract market value time series
        values = player_data['market_value_eur'].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(values)
        
        # Prepare sequences
        def create_sequences(data, seq_length=3):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        seq_length = min(3, len(scaled_values) - 1)
        if len(scaled_values) <= seq_length:
            continue
            
        X, y = create_sequences(scaled_values, seq_length)
        
        if len(X) == 0:
            continue
            
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
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
                    pred_value = values[i][0]
                else:
                    last_seq = scaled_values[i-seq_length:i].reshape(1, seq_length, 1)
                    scaled_pred = model.predict(last_seq, verbose=0)
                    pred_value = scaler.inverse_transform(scaled_pred)[0][0]
                
                all_predictions.append({
                    'player_name': player,
                    'season': player_data.iloc[i]['season'],
                    'actual_value': values[i][0],
                    'univariate_prediction': max(0, pred_value)
                })
            
            successful_players += 1
            if successful_players % 10 == 0:
                print(f"  Processed {successful_players} players...")
                
        except Exception as e:
            print(f"  Error with {player}: {e}")
            continue
    
    # Save predictions
    predictions_df = pd.DataFrame(all_predictions)
    
    if save_predictions:
        predictions_df.to_csv('models/predictions/univariate_predictions.csv', index=False)
        print(f"\n[OK] Univariate LSTM: {len(predictions_df)} predictions saved for {successful_players} players")
    
    return predictions_df

if __name__ == "__main__":
    train_univariate_lstm()
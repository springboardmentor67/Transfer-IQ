import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def train_multivariate_lstm(data_path='player_data.csv', save_predictions=True):
    """
    Train Multivariate LSTM using multiple features
    """
    print("=" * 60)
    print("Training Multivariate LSTM Model")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"[OK] Data loaded: {df.shape}")
    
    # Define features to use
    feature_columns = [
        'market_value_eur', 'age', 'goals', 'assists', 'appearances',
        'minutes_played', 'yellow_cards', 'red_cards', 'sentiment_score', 'injury_days'
    ]
    
    # Filter available columns
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"[OK] Using features: {available_features}")
    
    # Group by player
    players = df['player_name'].unique()
    all_predictions = []
    
    # Create directories
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('models/predictions', exist_ok=True)
    
    successful_players = 0
    
    for player in players:
        # Get player's historical data
        player_data = df[df['player_name'] == player].sort_values('season')
        
        if len(player_data) < 3:
            continue
            
        # Prepare multivariate features
        features = player_data[available_features].values
        
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
        
        # Build multivariate LSTM
        model = Sequential([
            LSTM(100, activation='relu', return_sequences=True, input_shape=(seq_length, len(available_features))),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
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
            
            # Make predictions for all seasons
            for i in range(len(player_data)):
                if i < seq_length:
                    pred_value = player_data.iloc[i]['market_value_eur']
                else:
                    last_seq = scaled_features[i-seq_length:i].reshape(1, seq_length, len(available_features))
                    scaled_pred = model.predict(last_seq, verbose=0)
                    pred_value = scaler_y.inverse_transform(scaled_pred)[0][0]
                
                all_predictions.append({
                    'player_name': player,
                    'season': player_data.iloc[i]['season'],
                    'actual_value': player_data.iloc[i]['market_value_eur'],
                    'multivariate_prediction': max(0, pred_value)
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
        predictions_df.to_csv('models/predictions/multivariate_predictions.csv', index=False)
        print(f"\n[OK] Multivariate LSTM: {len(predictions_df)} predictions saved for {successful_players} players")
    
    return predictions_df

if __name__ == "__main__":
    train_multivariate_lstm()
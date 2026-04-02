"""
Encoder-Decoder LSTM Model
Sequence-to-sequence prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ENCODER-DECODER LSTM MODEL TRAINING")
print("=" * 60)

# Create directories
os.makedirs('models/predictions', exist_ok=True)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('player_data.csv')
print(f"   Total records: {len(df)}")
print(f"   Unique players: {df['player_name'].nunique()}")

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
    
    if len(player_data) < 4:
        continue
    
    # Extract market value
    values = player_data['market_value_eur'].values.reshape(-1, 1)
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    
    # Create sequences for encoder-decoder
    def create_sequences(data, encoder_length=3, decoder_length=1):
        X_encoder, X_decoder, y = [], [], []
        for i in range(len(data) - encoder_length - decoder_length):
            X_encoder.append(data[i:i+encoder_length])
            X_decoder.append(data[i+encoder_length:i+encoder_length+decoder_length])
            y.append(data[i+encoder_length+decoder_length])
        return np.array(X_encoder), np.array(X_decoder), np.array(y)
    
    encoder_length = min(3, len(scaled_values) - 2)
    decoder_length = 1
    
    X_enc, X_dec, y = create_sequences(scaled_values, encoder_length, decoder_length)
    
    if len(X_enc) == 0:
        continue
    
    # Split data
    split_idx = int(len(X_enc) * 0.8)
    X_enc_train, X_enc_test = X_enc[:split_idx], X_enc[split_idx:]
    X_dec_train, X_dec_test = X_dec[:split_idx], X_dec[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build encoder-decoder model
    # Encoder
    encoder_inputs = Input(shape=(encoder_length, 1))
    encoder = LSTM(64, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(decoder_length, 1))
    decoder_lstm = LSTM(64, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(1))
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='mse')
    
    # Reshape y
    y_train_reshaped = y_train.reshape(-1, decoder_length, 1)
    y_test_reshaped = y_test.reshape(-1, decoder_length, 1)
    
    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    try:
        model.fit(
            [X_enc_train, X_dec_train], y_train_reshaped,
            epochs=50,
            batch_size=16,
            validation_data=([X_enc_test, X_dec_test], y_test_reshaped),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Make predictions
        for i in range(len(player_data)):
            if i < encoder_length + decoder_length:
                pred_value = player_data.iloc[i]['market_value_eur']
            else:
                enc_input = scaled_values[i-encoder_length-decoder_length:i-decoder_length].reshape(1, encoder_length, 1)
                dec_input = scaled_values[i-decoder_length:i].reshape(1, decoder_length, 1)
                
                scaled_pred = model.predict([enc_input, dec_input], verbose=0)
                pred_value = scaler.inverse_transform(scaled_pred[0])[0][0]
            
            all_predictions.append({
                'player_name': player,
                'season': player_data.iloc[i]['season'],
                'actual_value': player_data.iloc[i]['market_value_eur'],
                'encoder_decoder_prediction': max(0, pred_value)
            })
        
        successful_players += 1
        
    except Exception as e:
        print(f"   Error with {player}: {e}")
        continue

# Save predictions
print(f"\n3. Saving predictions...")
predictions_df = pd.DataFrame(all_predictions)
predictions_df.to_csv('models/predictions/encoder_decoder_predictions.csv', index=False)

print(f"\n" + "=" * 60)
print("ENCODER-DECODER LSTM COMPLETED")
print("=" * 60)
print(f"✓ Predictions saved for {successful_players} players")
print(f"✓ Total predictions: {len(predictions_df)}")

# Calculate metrics
actual = predictions_df['actual_value']
pred = predictions_df['encoder_decoder_prediction']
rmse = np.sqrt(mean_squared_error(actual, pred))
mae = mean_absolute_error(actual, pred)

print(f"\nModel Performance:")
print(f"  RMSE: €{rmse:,.0f}")
print(f"  MAE: €{mae:,.0f}")
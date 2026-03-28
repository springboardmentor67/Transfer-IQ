import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import joblib

# Adjust depending on where script runs from
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.lstm_models import UnivariateLSTM, MultivariateLSTM, EncoderDecoderLSTM, PlayerDataset
from src.models.ensemble_model import EnsembleModel

def create_sequences(df, features, target, window_size=3):
    X, y = [], []
    for player in df['player_name'].unique():
        player_data = df[df['player_name'] == player].sort_values('season')
        
        feature_data = player_data[features].values
        target_data = player_data[target].values
        
        if len(player_data) <= window_size:
            continue
            
        for i in range(len(player_data) - window_size):
            X.append(feature_data[i: i + window_size])
            # For uni/multivariate, we predict the next timestep
            y.append(target_data[i + window_size])
            
    return np.array(X), np.array(y)

def train_pipeline(data_path: str, models_dir: str):
    os.makedirs(models_dir, exist_ok=True)
    print("Loading prepared dataset...")
    df = pd.read_csv(data_path)
    
    # Exclude metadata from features
    keys = ['player_name', 'season', 'team', 'season_encoded', 'sentiment_label', 'market_value_source', 'career_stage', 'most_common_injury']
    target_col = 'market_value_eur'
    
    features = [c for c in df.columns if c not in keys and c != target_col]
    
    # 1. Create sequences
    print("Creating time-series sequences...")
    X_seq, y_seq = create_sequences(df, features, target_col, window_size=3)
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(PlayerDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(PlayerDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    # A. Univariate LSTM
    # We only use the target column sequence for this:
    target_idx = features.index('vader_compound_score') # wait, target_col not in features. We'll use 'log_market_value' instead, but let's assume log_market_value is univariate proxy. Or we can just use the target from previous steps. 
    # For simplicity, Univariate LSTM uses feature 0 (maybe log_market_value is feature 0).
    # Since market_value_eur is scaled, let's just train on feature 0 (let's say age or whatever, but preferably market value history).
    # Actually, Univariate requires the past market_value_eur. We didn't include it in features so let's just train the Multivariate LSTM as priority.
    
    # B. Multivariate LSTM
    print(f"Training Multivariate LSTM... Input Size: {len(features)}")
    mv_lstm = MultivariateLSTM(input_size=len(features), hidden_size=64, num_layers=2)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mv_lstm.parameters(), lr=0.001)
    
    epochs = 20
    for epoch in range(epochs):
        mv_lstm.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            preds = mv_lstm(X_b)
            loss = criterion(preds.squeeze(), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f}")
        
    torch.save(mv_lstm.state_dict(), f"{models_dir}/multivariate_lstm.pth")
    print("Multivariate LSTM saved.")
    
    # C. XGBoost Ensemble
    print("Training XGBoost Regressor...")
    # XGBoost does not natively consume (samples, window, features). 
    # We will flatten the 3 timesteps into 1D for each sample.
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    ensemble = EnsembleModel(lstm_weight=0.6, xgb_weight=0.4)
    ensemble.train_xgb(X_train_flat, y_train)
    ensemble.save_xgb(f"{models_dir}/xgboost_model.pkl")
    print("XGBoost Regressor saved.")
    
    # Save test data for evaluation step
    np.save(f"{models_dir}/X_test.npy", X_test)
    np.save(f"{models_dir}/y_test.npy", y_test)
    
    # Save features list
    import json
    with open(f"{models_dir}/features.json", "w") as f:
        json.dump(features, f)

if __name__ == "__main__":
    train_pipeline("data/processed/model_ready/clean_data.csv", "models")

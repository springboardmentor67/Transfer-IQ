import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

# 1. Baseline Models
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 2. Main Model
def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

# 3. Time-Series Model (LSTM in PyTorch)
class PlayerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(PlayerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x is (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # take last timestep
        return out.squeeze(1)

def train_lstm(df_seq, target_values_map, feature_cols, epochs=50, lr=0.001):
    # df_seq should have player_id and sequence of features
    # target_values_map is a dict of player_id -> target value
    
    # Sort sequences properly
    df_s = df_seq.sort_values(['player_id', 'match_num'])
    
    player_ids = df_s['player_id'].unique()
    
    # Prepare data tensors
    X, y = [], []
    for pid in player_ids:
        player_data = df_s[df_s['player_id'] == pid][feature_cols].values
        if len(player_data) > 0 and pid in target_values_map:
            X.append(player_data)
            y.append(target_values_map[pid])
            
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model
    model = PlayerLSTM(input_dim=len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    return model, X_tensor, y_tensor # return tensors for eval

# 4. Evaluator
def evaluate_model(y_true, y_pred, model_name):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    }

# 5. Predict with LSTM
def predict_lstm(model, X_tensor):
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).numpy()
    return preds

# Ensemble
def predict_ensemble(xgb_preds, lstm_preds, xgb_weight=0.6, lstm_weight=0.4):
    return (xgb_preds * xgb_weight) + (lstm_preds * lstm_weight)

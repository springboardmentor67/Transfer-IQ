import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost_model import PlayerValueXGBoost
from lstm_model import PlayerValueLSTM

# Set path relative to backend
import sys
base_path = 'e:/PROJECT/INFOSYS-AI/backend'
sys.path.append(os.path.join(base_path, 'utils'))
sys.path.append(os.path.join(base_path, 'model'))
from preprocessing import load_data, handle_missing_values, encode_categorical, scale_features
from feature_engineering import apply_feature_engineering

def train_models():
    # 1. Load data
    df = load_data(os.path.join(base_path, 'data/dataset.csv'))
    if df is None: return

    # 2. Preprocessing & Feature Engineering
    df = handle_missing_values(df)
    df, le = encode_categorical(df)
    df = apply_feature_engineering(df)

    # 3. Scale numerical columns
    features = ['goals', 'assists', 'matches', 'age', 'sentiment_score', 'performance_score', 'goal_ratio', 'age_factor']
    target = 'market_value'
    
    # Save the scaler and features for later use
    scaler = MinMaxScaler()
    df[features + [target]] = scaler.fit_transform(df[features + [target]])
    joblib.dump(scaler, 'e:/PROJECT/INFOSYS-AI/backend/data/scaler.pkl')
    
    # 4. Preparing data for XGBoost (regression)
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    xgb_model = PlayerValueXGBoost()
    xgb_model.train(X_train, y_train)
    xgb_model.save('e:/PROJECT/INFOSYS-AI/backend/model/xgboost_model.pkl')

    # Evaluate XGBoost
    xgb_preds = xgb_model.predict(X_test)
    print(f"XGBoost - RMSE: {np.sqrt(mean_squared_error(y_test, xgb_preds)):.4f}, R2: {r2_score(y_test, xgb_preds):.4f}")

    # 5. Preparing data for LSTM (time-series)
    # Each sequence will have 3 seasons of data to predict the target for that player
    sequence_data = []
    target_data = []
    
    players = df['player_id'].unique()
    for p in players:
        player_df = df[df['player_id'] == p].sort_values('season')
        if len(player_df) >= 3:
            # We take all historical features from first 3 entries for prediction
            # simplified: use first 3 steps to predict first 3 targets or similar
            # In simple demo, let's just create a sequence window
            sequence_data.append(player_df[features].values[:3])
            target_data.append(player_df[target].values[2]) # target after 3 entries
    
    if sequence_data:
        X_seq = torch.tensor(np.array(sequence_data), dtype=torch.float32)
        y_seq = torch.tensor(np.array(target_data), dtype=torch.float32).unsqueeze(1)
        
        lstm = PlayerValueLSTM(input_size=len(features))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)

        # Basic Training
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = lstm(X_seq)
            loss = criterion(outputs, y_seq)
            loss.backward()
            optimizer.step()
        
        torch.save(lstm.state_dict(), 'e:/PROJECT/INFOSYS-AI/backend/model/lstm_model.pth')
        print(f"LSTM trained with final loss: {loss.item():.4f}")

    print("Training Complete!")

if __name__ == "__main__":
    train_models()

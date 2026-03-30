import os
import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

# Set path relative to backend
base_path = 'e:/PROJECT/INFOSYS-AI/backend'
sys.path.append(os.path.join(base_path, 'model'))
from xgboost_model import PlayerValueXGBoost
from lstm_model import PlayerValueLSTM

def create_mock_models():
    print("Creating mock models for quick start...")
    
    # Create random scaler
    scaler = MinMaxScaler()
    dummy_data = np.random.rand(10, 9) # 8 features + 1 target
    scaler.fit(dummy_data)
    joblib.dump(scaler, os.path.join(base_path, 'data/scaler.pkl'))
    
    # Create mock XGBoost
    xgb_model = PlayerValueXGBoost()
    X_dummy = np.random.rand(10, 8)
    y_dummy = np.random.rand(10)
    xgb_model.train(X_dummy, y_dummy)
    xgb_model.save(os.path.join(base_path, 'model/xgboost_model.pkl'))
    
    # Create mock LSTM
    lstm = PlayerValueLSTM(input_size=8)
    torch.save(lstm.state_dict(), os.path.join(base_path, 'model/lstm_model.pth'))
    
    print("Mock models created at e:/PROJECT/INFOSYS-AI/backend/model/")

if __name__ == "__main__":
    create_mock_models()

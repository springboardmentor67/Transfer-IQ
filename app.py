
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np
import os
from typing import List

# Define the Model Class architecture (Must match training script)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.out = nn.Linear(16, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.relu(out)
        out = self.out(out)
        return out

app = FastAPI(title="Player Market Value Prediction API")

# Global variables for model and scaler
model = None
scaler = None

# Constants
MODEL_PATH = "models/lstm_model.pth"
SCALER_PATH = "data/processed/lstm_ready/scaler.pkl"

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        # Load scaler
        scaler = joblib.load(SCALER_PATH)
        input_size = scaler.n_features_in_
        
        # Load model
        model = LSTMModel(input_size=input_size)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("Model and Scaler loaded successfully.")
    else:
        print("Error: Model or Scaler not found.")

class PredictionInput(BaseModel):
    data: List[List[float]] # Expecting 7x9 array

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    data = np.array(input_data.data)
    
    # Validation
    if data.shape != (7, 9):
        raise HTTPException(status_code=400, detail=f"Expected input shape (7, 9), got {data.shape}")
        
    try:
        # Scale
        data_scaled = scaler.transform(data)
        
        # Convert to Tensor
        tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            pred_scaled = model(tensor).item()
            
        # Inverse Transform
        # Create dummy row with prediction in target column (last column)
        dummy_row = np.zeros(scaler.n_features_in_)
        dummy_row[-1] = pred_scaled
        pred_actual = scaler.inverse_transform([dummy_row])[0][-1]
        
        return {"predicted_market_value_eur": float(pred_actual)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

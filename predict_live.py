
import torch
import torch.nn as nn
import joblib
import numpy as np
import os

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

def main():
    # Paths
    MODEL_PATH = "models/lstm_model.pth"
    SCALER_PATH = "data/processed/lstm_ready/scaler.pkl"
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"Error: Model or Scaler not found. Checks paths:\n{MODEL_PATH}\n{SCALER_PATH}")
        return

    # 1. Load Scaler
    print(f"Loading scaler from {SCALER_PATH}...")
    scaler = joblib.load(SCALER_PATH)
    
    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    # Determine input size from scaler
    input_size = scaler.n_features_in_
    
    model = LSTMModel(input_size=input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # 3. Create Dummy Input (7 days, 9 features)
    features = [
        'goals', 'assists', 'shots', 'passes', 'xg',
        'daily_sentiment', 'daily_impact', 'daily_tweet_vol',
        'market_value_eur'
    ]
    print(f"\nGeneratin dummy input for features: {features}")
    
    # Random data resembling normalized data [0, 1] (or use actual vals and scale them)
    # Let's use raw values and scale them to be realistic
    # e.g. 0 goals, 20 passes, 1000000 market value
    
    # Create random raw data for 7 days
    raw_data = []
    for _ in range(7):
        day_stats = [
            np.random.randint(0, 2), # goals
            np.random.randint(0, 2), # assists
            np.random.randint(0, 5), # shots
            np.random.randint(20, 100), # passes
            np.random.uniform(0, 1.0), # xg
            np.random.uniform(-0.5, 0.5), # sentiment
            np.random.randint(1000, 50000), # impact
            np.random.randint(10, 500), # tweet vol
            50000000 # current market value (approx)
        ]
        raw_data.append(day_stats)
        
    raw_data = np.array(raw_data)
    print("Raw Input Data (Day 7):", raw_data[-1])
    
    # Scale
    data_scaled = scaler.transform(raw_data)
    
    # Convert to Tensor (Batch Size = 1, Seq Len = 7, Features = 9)
    input_tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)
    
    # 4. Predict
    with torch.no_grad():
        prediction_scaled = model(input_tensor).item()
        
    # Inverse Transform
    # We need to inverse transform just the target column.
    # We can create a dummy row with the prediction in the target column place
    # Target feature index is last one (8) based on week4_generate_sequences.py
    
    dummy_row = np.zeros(input_size)
    dummy_row[-1] = prediction_scaled
    
    prediction_actual = scaler.inverse_transform([dummy_row])[0][-1]
    
    print(f"\nScaled Prediction: {prediction_scaled:.6f}")
    print(f"Predicted Market Value: €{prediction_actual:,.2f}")

if __name__ == "__main__":
    main()

import numpy as np
import os
import argparse
import logging
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LSTM Model Definition (Must match Week 5) ---
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.relu(out)
        out = self.out(out)
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/lstm_ready", help="Directory with X.npy and y.npy")
    parser.add_argument("--models-dir", default="models", help="Directory to save/load models")
    parser.add_argument("--reports-dir", default="reports/figures", help="Directory to save plots")
    args = parser.parse_args()
    
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)
    
    # 1. Load Data
    logging.info("Loading sequence data...")
    x_path = os.path.join(args.data_dir, "X.npy")
    y_path = os.path.join(args.data_dir, "y.npy")
    scaler_path = os.path.join(args.data_dir, "scaler.pkl")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        logging.error("Data files not found. Run week4_generate_sequences.py first.")
        return
        
    X = np.load(x_path)
    y = np.load(y_path)
    scaler = joblib.load(scaler_path)
    
    # 2. Split Data (Must match LSTM split: shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 3. XGBoost Data Preparation
    # Flatten sequences? Or just take the last time step?
    # Using last time step features (t-1) to predict t
    X_train_xgb = X_train[:, -1, :]
    X_test_xgb = X_test[:, -1, :]
    
    logging.info(f"XGBoost Training Input Shape: {X_train_xgb.shape}")
    
    # 4. Train XGBoost
    logging.info("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    xgb_model.fit(X_train_xgb, y_train)
    
    # Predict XGBoost
    y_pred_xgb = xgb_model.predict(X_test_xgb)
    
    # Save XGBoost
    joblib.dump(xgb_model, os.path.join(args.models_dir, "ensemble_xgboost.pkl"))
    
    # 5. Generate LSTM Predictions
    logging.info("Generating LSTM predictions...")
    input_size = X_train.shape[2]
    lstm_model = LSTMModel(input_size=input_size).to(DEVICE)
    lstm_path = os.path.join(args.models_dir, "lstm_model.pth")
    
    if not os.path.exists(lstm_path):
        logging.error(f"LSTM model not found at {lstm_path}. Run Week 5 script first.")
        return
        
    lstm_model.load_state_dict(torch.load(lstm_path))
    lstm_model.eval()
    
    X_test_tensor = torch.Tensor(X_test).to(DEVICE)
    with torch.no_grad():
        y_pred_lstm = lstm_model(X_test_tensor).cpu().numpy().flatten()
        
    # 6. Ensemble (Weighted Average)
    # Simple average for now. Could be optimized.
    w_lstm = 0.5
    w_xgb = 0.5
    y_pred_ensemble = (w_lstm * y_pred_lstm) + (w_xgb * y_pred_xgb)
    
    # 7. Evaluation
    def eval_metrics(y_true, y_pred, name):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae
    
    rmse_lstm, mae_lstm = eval_metrics(y_test, y_pred_lstm, "LSTM")
    rmse_xgb, mae_xgb = eval_metrics(y_test, y_pred_xgb, "XGBoost")
    rmse_ens, mae_ens = eval_metrics(y_test, y_pred_ensemble, "Ensemble")
    
    logging.info(f"LSTM     - RMSE: {rmse_lstm:.6f}, MAE: {mae_lstm:.6f}")
    logging.info(f"XGBoost  - RMSE: {rmse_xgb:.6f}, MAE: {mae_xgb:.6f}")
    logging.info(f"Ensemble - RMSE: {rmse_ens:.6f}, MAE: {mae_ens:.6f}")

    # Save metrics to file
    metrics_path = os.path.join(args.reports_dir, "ensemble_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"LSTM RMSE: {rmse_lstm:.6f}\n")
        f.write(f"XGBoost RMSE: {rmse_xgb:.6f}\n")
        f.write(f"Ensemble RMSE: {rmse_ens:.6f}\n")
    logging.info(f"Metrics saved to {metrics_path}")
    
    # 8. Plotting Comparison
    # Inverse transform to EUR
    num_features = scaler.n_features_in_
    
    def inverse(pred_arr):
        dummy = np.zeros((len(pred_arr), num_features))
        dummy[:, -1] = pred_arr
        return scaler.inverse_transform(dummy)[:, -1]
    
    act_eur = inverse(y_test)
    lstm_eur = inverse(y_pred_lstm)
    xgb_eur = inverse(y_pred_xgb)
    ens_eur = inverse(y_pred_ensemble)
    
    plt.figure(figsize=(15, 7))
    subset = 150
    plt.plot(act_eur[:subset], label='Actual', color='black', linewidth=2)
    plt.plot(lstm_eur[:subset], label=f'LSTM (RMSE={rmse_lstm:.4f})', alpha=0.7)
    plt.plot(xgb_eur[:subset], label=f'XGBoost (RMSE={rmse_xgb:.4f})', alpha=0.7)
    plt.plot(ens_eur[:subset], label=f'Ensemble (RMSE={rmse_ens:.4f})', color='red', linestyle='--', linewidth=2)
    
    plt.title("Model Comparison: LSTM vs XGBoost vs Ensemble")
    plt.xlabel("Sample Index")
    plt.ylabel("Market Value (EUR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(args.reports_dir, "ensemble_comparison.png")
    plt.savefig(plot_path)
    logging.info(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()

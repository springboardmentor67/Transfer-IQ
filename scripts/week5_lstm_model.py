import numpy as np
import os
import argparse
import logging
import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0)) # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.relu(out)
        out = self.out(out)
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/lstm_ready", help="Directory with X.npy and y.npy")
    parser.add_argument("--models-dir", default="models", help="Directory to save model")
    parser.add_argument("--reports-dir", default="reports/figures", help="Directory to save plots")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
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
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.Tensor(X_train).to(DEVICE)
    y_train_tensor = torch.Tensor(y_train).to(DEVICE)
    X_test_tensor = torch.Tensor(X_test).to(DEVICE)
    y_test_tensor = torch.Tensor(y_test).to(DEVICE)
    
    # Create DataLoaders
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
    logging.info(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # 3. Build Model
    input_size = X_train.shape[2] # Number of features
    model = LSTMModel(input_size=input_size).to(DEVICE)
    logging.info(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Train
    logging.info("Starting training...")
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation on test set (simplification for this script)
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            val_loss = criterion(test_outputs.squeeze(), y_test_tensor).item()
        
        val_losses.append(val_loss)
            
        logging.info(f"Epoch [{epoch+1}/{args.epochs}], Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early Stopping Check
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(args.models_dir, "lstm_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break
                
    # Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(args.reports_dir, "lstm_loss_curve.png")
    plt.savefig(loss_plot_path)
    logging.info(f"Loss curve saved to {loss_plot_path}")

    # 5. Evaluate
    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.models_dir, "lstm_model.pth")))
    model.eval()
    
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.cpu().numpy().flatten()
        
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    logging.info(f"Test MSE: {mse:.6f}")
    logging.info(f"Test RMSE: {rmse:.6f}")
    logging.info(f"Test MAE: {mae:.6f}")
    
    # 6. Predictions & Plotting
    # Inverse transform logic
    num_features = scaler.n_features_in_
    
    dummy_pred = np.zeros((len(y_pred), num_features))
    dummy_pred[:, -1] = y_pred
    pred_inverse = scaler.inverse_transform(dummy_pred)[:, -1]
    
    dummy_actual = np.zeros((len(y_test), num_features))
    dummy_actual[:, -1] = y_test
    actual_inverse = scaler.inverse_transform(dummy_actual)[:, -1]
    
    plt.figure(figsize=(14, 6))
    subset_size = 100
    plt.plot(actual_inverse[:subset_size], label='Actual Value (EUR)', color='blue')
    plt.plot(pred_inverse[:subset_size], label='Predicted Value (EUR)', color='red', linestyle='--')
    plt.title('LSTM Model Prediction (PyTorch): Market Value Subset')
    plt.xlabel('Sample Index')
    plt.ylabel('Market Value (EUR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(args.reports_dir, "lstm_predictions.png")
    plt.savefig(plot_path)
    logging.info(f"Prediction plot saved to {plot_path}")
    logging.info("Model saved to models/lstm_model.pth")

if __name__ == "__main__":
    main()

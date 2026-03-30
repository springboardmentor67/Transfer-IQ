import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PlayerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class UnivariateLSTM(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2):
        super(UnivariateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Univariate: input size is 1
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.out = nn.Linear(16, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] # Last time step
        out = self.fc(out)
        out = self.relu(out)
        out = self.out(out)
        return out

class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(MultivariateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.out = nn.Linear(16, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.relu(out)
        out = self.out(out)
        return out

class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_steps=3, dropout=0.2):
        super(EncoderDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Decoder (Predicting multiple steps ahead)
        self.decoder = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        _, (h, c) = self.encoder(x)
        
        # decoder input, starts with the last known target value (we use zeros as simple approach or pass the last value)
        dec_input = torch.zeros(x.size(0), 1, 1).to(x.device)
        
        outputs = []
        for _ in range(self.output_steps):
            out, (h, c) = self.decoder(dec_input, (h, c))
            pred = self.fc(out[:, -1, :])
            outputs.append(pred)
            dec_input = pred.unsqueeze(1)
            
        outputs = torch.cat(outputs, dim=1) # Shape: (batch, output_steps)
        return outputs

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch.view_as(preds))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch.view_as(preds))
                val_loss += loss.item()
        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
    return model

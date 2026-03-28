import torch
import torch.nn as nn
import os

class PlayerValueLSTM(nn.Module):
    """
    Simple LSTM model for player performance/value time-series prediction.
    Uses PyTorch as it's compatible with Python 3.14 on this machine.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(PlayerValueLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # use the last hidden state for prediction
        out = self.fc(out[:, -1, :])
        return out

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

class modelDoPredykcji(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(modelDoPredykcji, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,dropout=0.2,dtype=torch.float64)
        self.fc = nn.Linear(hidden_dim, output_dim, dtype=torch.float64)

        self.to(self.device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=self.device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach() ))
        out = self.fc(out[:, -1, :])

        return out
    
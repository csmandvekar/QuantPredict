import torch
import torch.nn as nn

class ParallelCNNLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim: int = 0):
        super().__init__()
        # CNN branch
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        # LSTM branch
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        # Fusion head
        self.fc = nn.Sequential(
            nn.Linear(32 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [B, T, F]
        c = x.permute(0,2,1)         # [B, F, T]
        c = self.cnn(c).squeeze(-1)  # [B, 32]
        l,_ = self.lstm(x)
        l,_ = self.lstm2(l)
        l = l[:, -1, :]              # [B, 32]
        out = torch.cat([c, l], dim=1)
        return self.fc(out).squeeze(-1)
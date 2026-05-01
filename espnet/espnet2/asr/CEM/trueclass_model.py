"""Re-implementations of TruCLeS baselines used in the paper.

Reference:
    Ravi, Arora et al. "TeLeS: Temporal lexeme similarity score to estimate
    confidence in end-to-end ASR." IEEE/ACM TASLP, 2024.

These are baselines for SR-CEM evaluation on CTC-only and RNN-T models
(see Tables 7 and 8 in the paper).
"""

import torch.nn as nn
from torch.utils.data import Dataset


class trueclass_rnnt_model(nn.Module):
    """TruCLeS-RNNT baseline: two stacked bi-LSTMs with a sigmoid head.

    Approximate parameter count: ~5.4M with default settings.
    """

    def __init__(self, input_dim: int = 2560, hidden_dim: int = 512, num_layers: int = 1):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            hidden_dim * 2, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        """
        Args:
            x: padded input tensor (batch_size, max_seq_len, input_dim)
            lengths: pre-padding lengths of each sequence
        """
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.fc(x).squeeze(-1)
        return self.sigmoid(x)


class trucles_ctc_model(nn.Module):
    """TruCLeS-CTC baseline: 4-layer MLP with a sigmoid head.

    Approximate parameter count: ~7.4M with the configurations from the paper.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim * 4)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.relu1 = nn.ReLU()
        self.dense3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dense4 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu1(x)
        x = self.dense3(x)
        x = self.relu2(x)
        x = self.dense4(x)
        return self.sigmoid(x)


class VariableLengthDataset(Dataset):
    """Variable-length dataset of (features, targets) tensor pairs."""

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

import torch
import torch.nn as nn

import pickle
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch
import numpy as np



class trueclass_rnnt_model(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=512, num_layers=1):
        super(trueclass_rnnt_model, self).__init__()

        # First LSTM
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Second LSTM
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Fully connected layer for confidence score prediction
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional LSTM

        # Sigmoid activation for confidence estimation (output between 0 and 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        """
        x: Padded input tensor (batch_size, max_seq_len, input_dim)
        lengths: Actual lengths of sequences before padding
        """
        # Pack padded sequence
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # First LSTM
        x, _ = self.lstm1(x)

        # Second LSTM (directly taking output of first LSTM)
        x, _ = self.lstm2(x)

        # Unpack sequence
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Fully connected layer
        x = self.fc(x).squeeze(-1)  # Shape: (batch_size, max_seq_len)

        # Sigmoid activation
        return self.sigmoid(x)



class trucles_ctc_model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(trucles_ctc_model, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim*4)
        
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.relu1 = nn.ReLU()
        self.dense3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.relu2 = nn.ReLU()
        
        #self.tanh = nn.Tanh()
        self.dense4 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dense1(x)
        #x = self.tanh(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu1(x)
        x = self.dense3(x)
        x = self.relu2(x)
        x = self.dense4(x)
        x = self.sigmoid(x)
        return x
    
from torch.utils.data import Dataset

class VariableLengthDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features  # list of tensors
        self.targets = targets    # list of tensors

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
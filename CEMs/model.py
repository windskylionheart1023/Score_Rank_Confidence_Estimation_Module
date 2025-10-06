import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward

class Score_CEM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
            # nn.Linear(input_size, 1),
            # nn.ReLU(),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)

# Baselines
# Three layers
class CEM_MLP(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.fc(x)

class WPXformer(nn.Module):
    """
    WP Xformer: project input to d_model, run one Transformer DecoderLayer, sigmoid output.
    """
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 1) Project input to model dimension
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.ReLU(),
        )
        # 2) One Transformer decoder layer (single block)
        self.decoder_block = DecoderLayer(
            size=d_model,
            self_attn=MultiHeadedAttention(nhead, d_model, dropout),
            src_attn=MultiHeadedAttention(nhead, d_model, dropout),
            feed_forward=PositionwiseFeedForward(d_model, dim_feedforward, dropout),
            dropout_rate=dropout,
            normalize_before=True,
            concat_after=False,
        )
        # 3) Final projection to score
        self.out = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_size)
        Returns:
            Tensor of shape (batch, 1) with values in [0,1]
        """
        # Project and activate
        h = self.layer1(x)               # (B, d_model)
        # Wrap as length-1 sequence
        tgt = h.unsqueeze(1)             # (B, 1, d_model)
        mem = h.unsqueeze(1)             # (B, 1, d_model)
        # Single decoder layer forward (returns y, tgt_mask, mem, mem_mask)
        y, *_ = self.decoder_block(tgt, None, mem, None)
        y = y.squeeze(1)                 # (B, d_model)
        return self.out(y)               # (B, 1)


class E2EXformer(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        token_dim: int,
        d_model: int = 64,
        nhead: int = 1,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, d_model)
        self.token_proj = nn.Linear(token_dim, d_model)
        self.decoder_layer = DecoderLayer(
            size=d_model,
            self_attn=MultiHeadedAttention(nhead, d_model, dropout),
            src_attn=MultiHeadedAttention(nhead, d_model, dropout),
            feed_forward=PositionwiseFeedForward(d_model, dim_feedforward, dropout),
            dropout_rate=dropout,
            normalize_before=True,
            concat_after=False,
        )
        self.out = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, enc, enc_lengths, b, b_lengths):
        """
        enc: [B, max_enc_len, encoder_dim]
        enc_lengths: [B]
        b: [B, max_feat_len, token_dim]
        b_lengths: [B]
        """
        B, max_enc_len, _ = enc.size()
        B, max_b_len, _ = b.size()
        device = enc.device

        # Project
        memory = self.encoder_proj(enc)  # [B, max_enc_len, d_model]
        tgt = self.token_proj(b)         # [B, max_b_len, d_model]

        # ---- Create Padding Masks ----
        # Padding mask: True where pad
        enc_mask = (torch.arange(max_enc_len, device=device)[None, :] >= enc_lengths[:, None])  # [B, max_enc_len]
        b_mask   = (torch.arange(max_b_len, device=device)[None, :] >= b_lengths[:, None])      # [B, max_b_len]

        # memory_mask should be [B, 1, max_enc_len]
        enc_mask = enc_mask.unsqueeze(1)  # [B, 1, max_enc_len]
        b_mask = b_mask.unsqueeze(1)    # [B, 1, max_b_len]
        # tgt_mask should be [B, max_b_len]
        # b_mask: [B, max_b_len], already correct

        # Decoder layer expects: (tgt, tgt_mask, memory, memory_mask)
        y, *_ = self.decoder_layer(tgt, b_mask, memory, enc_mask)

        # Pick last non-padded output for each sample in batch
        idx = (b_lengths - 1).clamp(min=0)
        y_last = y[torch.arange(B, device=device), idx, :]  # [B, d_model]

        return self.out(y_last)  # [B, 1]


# Other helpful classes
from torch.utils.data import Dataset

class TripleVariableLengthDataset(Dataset):
    def __init__(self, enc, features, target):
        self.enc = enc
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        return self.enc[idx], self.features[idx], self.target[idx]


import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    enc_batch, feat_batch, tgt_batch = zip(*batch)
    enc_lengths = torch.tensor([e.shape[0] for e in enc_batch])
    feat_lengths = torch.tensor([f.shape[0] for f in feat_batch])
    padded_enc = pad_sequence(enc_batch, batch_first=True)   # [B, max_enc_len, D_enc]
    padded_feat = pad_sequence(feat_batch, batch_first=True) # [B, max_feat_len, D_feat]
    tgt_batch = torch.tensor(tgt_batch)
    return padded_enc, enc_lengths, padded_feat, feat_lengths, tgt_batch

class NamedTensorDataset(Dataset):
    def __init__(self, data, targets, feature_names=None):
        self.data = data
        self.targets = targets
        self.feature_names = feature_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'features': self.data[idx],
            'target': self.targets[idx],
            'feature_names': self.feature_names
        }
    
"""Model definitions and dataset wrappers used by SR-CEM.

The proposed module is :class:`Score_CEM` (Eq. 6 / Eq. 10 in the paper):
a single hidden layer with ReLU activation followed by a sigmoid head.
Other modules (:class:`CEM_MLP`, :class:`WPXformer`, :class:`E2EXformer`)
are baselines re-implemented for comparison.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)


class Score_CEM(nn.Module):
    """SR-CEM scoring head: Linear -> ReLU -> Linear -> Sigmoid."""

    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)


class CEM_MLP(nn.Module):
    """Three-layer MLP baseline (Qiu et al., 2021)."""

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
    """Single-block Transformer baseline.

    Projects per-token features to ``d_model``, runs one decoder layer with
    self/source attention pointing back to the same projected vector, and
    outputs a sigmoid confidence.
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
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.ReLU(),
        )
        self.decoder_block = DecoderLayer(
            size=d_model,
            self_attn=MultiHeadedAttention(nhead, d_model, dropout),
            src_attn=MultiHeadedAttention(nhead, d_model, dropout),
            feed_forward=PositionwiseFeedForward(d_model, dim_feedforward, dropout),
            dropout_rate=dropout,
            normalize_before=True,
            concat_after=False,
        )
        self.out = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        tgt = h.unsqueeze(1)
        mem = h.unsqueeze(1)
        y, *_ = self.decoder_block(tgt, None, mem, None)
        return self.out(y.squeeze(1))


class E2EXformer(nn.Module):
    """End-to-end Transformer baseline that attends over encoder states.

    Inputs:
        enc:         [B, max_enc_len, encoder_dim] encoder outputs
        enc_lengths: [B] valid encoder lengths
        b:           [B, max_feat_len, token_dim] per-token feature sequence
        b_lengths:   [B] valid feature lengths
    """

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
        self.out = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, enc, enc_lengths, b, b_lengths):
        B, max_enc_len, _ = enc.size()
        _, max_b_len, _ = b.size()
        device = enc.device

        memory = self.encoder_proj(enc)
        tgt = self.token_proj(b)

        enc_mask = (
            torch.arange(max_enc_len, device=device)[None, :] >= enc_lengths[:, None]
        ).unsqueeze(1)
        b_mask = (
            torch.arange(max_b_len, device=device)[None, :] >= b_lengths[:, None]
        ).unsqueeze(1)

        y, *_ = self.decoder_layer(tgt, b_mask, memory, enc_mask)

        idx = (b_lengths - 1).clamp(min=0)
        y_last = y[torch.arange(B, device=device), idx, :]
        return self.out(y_last)


class NamedTensorDataset(Dataset):
    """Tensor dataset that retains feature names alongside data."""

    def __init__(self, data, targets, feature_names=None):
        self.data = data
        self.targets = targets
        self.feature_names = feature_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "features": self.data[idx],
            "target": self.targets[idx],
            "feature_names": self.feature_names,
        }


class TripleVariableLengthDataset(Dataset):
    """Variable-length dataset of (encoder states, features, target) triples."""

    def __init__(self, enc, features, target):
        self.enc = enc
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        return self.enc[idx], self.features[idx], self.target[idx]


def collate_fn(batch):
    """Pad a batch of (enc, feat, target) triples; used with E2EXformer."""
    enc_batch, feat_batch, tgt_batch = zip(*batch)
    enc_lengths = torch.tensor([e.shape[0] for e in enc_batch])
    feat_lengths = torch.tensor([f.shape[0] for f in feat_batch])
    padded_enc = pad_sequence(enc_batch, batch_first=True)
    padded_feat = pad_sequence(feat_batch, batch_first=True)
    tgt_batch = torch.tensor(tgt_batch)
    return padded_enc, enc_lengths, padded_feat, feat_lengths, tgt_batch

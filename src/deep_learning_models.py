"""
Deep Learning Architectures for Auditory-Evoked Pupillary Response (AEPR) Discrimination.

Implements modular PyTorch architectures for multi-channel pupillometry time series:
1. MultiScaleConv1DNet: Multi-branch 1D temporal CNN with Squeeze-and-Excitation channel attention.
2. BiLSTMAttentionNet: 2-layer Bidirectional LSTM with Bahdanau additive temporal self-attention.
3. DilatedTCNNet: Residual dilated causal 1D temporal convolutional network.
4. CNNTransformerNet: Hybrid 1D-CNN temporal stem with Transformer Encoder layers.
5. Loss Functions: Focal Loss and Weighted Binary Cross-Entropy with Logits.
6. Dataset & Tensor Extractors.
"""

from typing import Dict, Any, List, Optional, Tuple, Sequence
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.preprocessing import TrialEpoch


# -----------------------------------------------------------------------------
# 1. Dataset & Multi-Channel Tensor Extraction
# -----------------------------------------------------------------------------

class PupilTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for multi-channel pupillometry time-series epochs.
    Shape: (C, T) where C=3 channels (subtractive, divisive, velocity), T=201 timepoints.
    """
    def __init__(self, tensors: np.ndarray, labels: np.ndarray, subject_ids: Optional[Sequence[str]] = None):
        """
        Parameters:
            tensors: Float array of shape (N, C, T).
            labels: Int/Float array of shape (N,).
            subject_ids: Optional subject identifiers.
        """
        self.tensors = torch.tensor(tensors, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tensors[idx], self.labels[idx]


def extract_multichannel_tensor_from_epoch(
    epoch: TrialEpoch,
    target_fs: float = 50.0,
    t_start: float = -0.50,
    t_end: float = 3.50
) -> Optional[np.ndarray]:
    """
    Extracts a 3-channel (C=3, T=201) aligned time-series tensor from a TrialEpoch:
      Channel 0: Subtractive baseline-corrected diameter Delta P(t)
      Channel 1: Divisive baseline-corrected signal % Delta P(t)
      Channel 2: Numerical first derivative (dilation velocity) d(Delta P)/dt
    """
    if not epoch.is_valid:
        return None

    t = epoch.time
    y_sub = epoch.pupil_subtractive
    y_div = epoch.pupil_divisive

    valid = np.isfinite(y_sub)
    if np.sum(valid) < 10:
        return None

    time_grid = np.arange(t_start, t_end + 1e-5, 1.0 / target_fs)
    n_pts = len(time_grid)

    # Channel 0: Subtractive
    c0 = np.interp(time_grid, t[valid], y_sub[valid], left=np.nan, right=np.nan)
    # Channel 1: Divisive
    valid_div = np.isfinite(y_div)
    c1 = np.interp(time_grid, t[valid_div], y_div[valid_div], left=np.nan, right=np.nan) if np.sum(valid_div) >= 10 else (c0 / max(epoch.baseline_val, 1e-2) * 100.0)

    # Edge imputation for any boundary extrapolation
    for c in [c0, c1]:
        if np.any(np.isnan(c)):
            fin = np.where(np.isfinite(c))[0]
            if len(fin) == 0:
                return None
            c[:fin[0]] = c[fin[0]]
            c[fin[-1] + 1:] = c[fin[-1]]

    # Channel 2: Velocity
    dt = 1.0 / target_fs
    c2 = np.gradient(c0, dt)

    tensor = np.stack([c0, c1, c2], axis=0).astype(np.float32)  # Shape: (3, T)
    return tensor


def build_tensor_dataset_from_epochs(
    epochs: Sequence[TrialEpoch],
    subject_ids: Sequence[str],
    labels: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds aligned multi-channel tensor array (N, 3, T), label array (N,), and subject array (N,).
    """
    tensors = []
    y_list = []
    subjs_list = []

    for ep, subj, y_lbl in zip(epochs, subject_ids, labels):
        t_arr = extract_multichannel_tensor_from_epoch(ep)
        if t_arr is not None and np.all(np.isfinite(t_arr)):
            tensors.append(t_arr)
            y_list.append(y_lbl)
            subjs_list.append(subj)

    X_mat = np.stack(tensors, axis=0)  # (N, 3, T)
    y_arr = np.array(y_list, dtype=np.int32)
    subjs_arr = np.array(subjs_list, dtype=object)

    return X_mat, y_arr, subjs_arr


# -----------------------------------------------------------------------------
# 2. Loss Functions
# -----------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary time-series classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * torch.pow(1.0 - p_t + 1e-8, self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# -----------------------------------------------------------------------------
# 3. Model Architecture 1: Multi-Scale 1D-CNN
# -----------------------------------------------------------------------------

class SqueezeExcitation1D(nn.Module):
    """1D Squeeze-and-Excitation channel attention block."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.GELU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1)
        return x * w


class MultiScaleConv1DNet(nn.Module):
    """
    Multi-Scale Temporal 1D-CNN for biosignal classification.
    Processes signals through 3 parallel convolutional kernel sizes (3, 7, 15 timepoints).
    """
    def __init__(self, in_channels: int = 3, num_filters: int = 32, dropout: float = 0.30):
        super().__init__()
        # 3 parallel multi-scale branches
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_filters),
            nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(num_filters),
            nn.GELU()
        )

        comb_filters = num_filters * 3
        self.se1 = SqueezeExcitation1D(comb_filters)
        self.pool1 = nn.MaxPool1d(2)  # T: 201 -> 100

        # Layer 2 Conv
        self.conv2 = nn.Sequential(
            nn.Conv1d(comb_filters, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2)  # T: 100 -> 50
        )
        self.se2 = SqueezeExcitation1D(64)

        # Layer 3 Conv
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification Head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.pool1(self.se1(out))
        out = self.se2(self.conv2(out))
        out = self.conv3(out)
        out = self.global_pool(out).flatten(1)
        logits = self.head(out)
        return logits


# -----------------------------------------------------------------------------
# 4. Model Architecture 2: Bi-LSTM with Temporal Self-Attention
# -----------------------------------------------------------------------------

class TemporalAdditiveAttention(nn.Module):
    """
    Bahdanau-style additive self-attention over temporal hidden states.
    Produces context vector c and attention weights alpha_t.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.v = nn.Linear(hidden_dim // 2, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, hidden_dim)
        score = self.v(torch.tanh(self.proj(h)))  # (B, T, 1)
        weights = F.softmax(score, dim=1)         # (B, T, 1)
        context = torch.sum(h * weights, dim=1)   # (B, hidden_dim)
        return context, weights.squeeze(-1)       # context: (B, hidden_dim), weights: (B, T)


class BiLSTMAttentionNet(nn.Module):
    """
    2-layer Bidirectional LSTM with temporal self-attention for single-trial pupillometry.
    Allows extracting temporal attention heatmaps for explainable saliency analysis.
    """
    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.30):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = TemporalAdditiveAttention(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, C, T)
        stem_out = self.stem(x)               # (B, 32, T)
        lstm_in = stem_out.transpose(1, 2)    # (B, T, 32)
        h, _ = self.lstm(lstm_in)             # (B, T, 2*hidden_dim)
        context, attn_weights = self.attention(h) # context: (B, 2*hidden_dim), attn: (B, T)
        logits = self.head(context)           # (B, 1)

        if return_attention:
            return logits, attn_weights
        return logits


# -----------------------------------------------------------------------------
# 5. Model Architecture 3: Dilated Temporal Convolutional Network (TCN)
# -----------------------------------------------------------------------------

class TemporalBlock1D(nn.Module):
    """Residual causal dilated 1D temporal convolutional block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.20):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        # Match lengths if slight padding mismatch
        if out.shape[-1] != res.shape[-1]:
            out = out[:, :, :res.shape[-1]]
        return out + res


class DilatedTCNNet(nn.Module):
    """
    Dilated Temporal Convolutional Network with exponentially increasing receptive fields.
    """
    def __init__(self, in_channels: int = 3, num_channels: Sequence[int] = (32, 64, 128), kernel_size: int = 3, dropout: float = 0.25):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_c = in_channels if i == 0 else num_channels[i - 1]
            out_c = num_channels[i]
            layers.append(TemporalBlock1D(in_c, out_c, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1], 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        out = self.global_pool(out).flatten(1)
        logits = self.head(out)
        return logits


# -----------------------------------------------------------------------------
# 6. Model Architecture 4: CNN-Transformer
# -----------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal Transformer tokens."""
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1)]


class CNNTransformerNet(nn.Module):
    """
    Hybrid 1D-CNN feature stem coupled with a Multi-Head Self-Attention Transformer Encoder.
    """
    def __init__(self, in_channels: int = 3, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.30):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=5, stride=2, padding=2),  # T: 201 -> 101
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        stem_out = self.stem(x)                  # (B, d_model, T_sub)
        tokens = stem_out.transpose(1, 2)        # (B, T_sub, d_model)
        tokens = self.pos_encoder(tokens)
        encoded = self.transformer_encoder(tokens) # (B, T_sub, d_model)
        cls_rep = torch.mean(encoded, dim=1)     # (B, d_model)
        logits = self.head(cls_rep)              # (B, 1)
        return logits

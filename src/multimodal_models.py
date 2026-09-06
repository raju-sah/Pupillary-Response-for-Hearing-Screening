"""
PyTorch Multimodal Architectures for Dataset C (Pupillometry + ECG + EEG).
Implements:
1. PupilOnlyNet: 1D Temporal CNN + Multi-Scale Feature Extractor
2. ECGOnlyNet: Temporal 1D CNN with cardiac wave representation
3. EEGOnlyNet: Spatial-Temporal EEGNet-inspired 4-channel conv net
4. MultimodalCrossAttentionNet: Cross-Modal Attention Fusion of Pupil + ECG + EEG
"""

import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class PupilOnlyNet(nn.Module):
    """Deep 1D-CNN operating strictly on pupillometry channels [ΔP, %ΔP, dΔP/dt]."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, num_classes: int = 1):
        super().__init__()
        self.stem = ConvBlock1D(in_channels, hidden_dim, kernel_size=7)
        self.b1 = ConvBlock1D(hidden_dim, hidden_dim, kernel_size=5)
        self.b2 = ConvBlock1D(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2)
        self.b3 = ConvBlock1D(hidden_dim * 2, hidden_dim * 2, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def extract_features(self, x_pupil: torch.Tensor) -> torch.Tensor:
        h = self.stem(x_pupil)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        return self.pool(h).squeeze(-1)

    def forward(self, x_pupil: torch.Tensor, *args) -> torch.Tensor:
        feat = self.extract_features(x_pupil)
        return self.classifier(feat)


class ECGOnlyNet(nn.Module):
    """Deep 1D-CNN operating strictly on ECG cardiac channels [ECG(t), dECG/dt]."""

    def __init__(self, in_channels: int = 2, hidden_dim: int = 64, num_classes: int = 1):
        super().__init__()
        self.stem = ConvBlock1D(in_channels, hidden_dim, kernel_size=9)
        self.b1 = ConvBlock1D(hidden_dim, hidden_dim, kernel_size=7)
        self.b2 = ConvBlock1D(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2)
        self.b3 = ConvBlock1D(hidden_dim * 2, hidden_dim * 2, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def extract_features(self, x_ecg: torch.Tensor) -> torch.Tensor:
        h = self.stem(x_ecg)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        return self.pool(h).squeeze(-1)

    def forward(self, x_pupil: torch.Tensor, x_ecg: torch.Tensor, *args) -> torch.Tensor:
        feat = self.extract_features(x_ecg)
        return self.classifier(feat)


class EEGOnlyNet(nn.Module):
    """Spatial-temporal 1D-CNN operating strictly on 4 central EEG channels [Cz, Fz, Pz, FCz]."""

    def __init__(self, in_channels: int = 4, hidden_dim: int = 64, num_classes: int = 1):
        super().__init__()
        # Spatial filtering across electrodes
        self.spatial_conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.bn_spatial = nn.BatchNorm1d(hidden_dim)
        # Temporal filtering
        self.stem = ConvBlock1D(hidden_dim, hidden_dim, kernel_size=7)
        self.b1 = ConvBlock1D(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2)
        self.b2 = ConvBlock1D(hidden_dim * 2, hidden_dim * 2, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def extract_features(self, x_eeg: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn_spatial(self.spatial_conv(x_eeg)))
        h = self.stem(h)
        h = self.b1(h)
        h = self.b2(h)
        return self.pool(h).squeeze(-1)

    def forward(self, x_pupil: torch.Tensor, x_ecg: torch.Tensor, x_eeg: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x_eeg)
        return self.classifier(feat)


class MultimodalCrossAttentionNet(nn.Module):
    """
    Cross-Modal Attention Fusion Network.
    Encodes Pupil, ECG, and EEG modalities into aligned hidden spaces,
    then applies bidirectional Cross-Attention and residual gated fusion.
    """

    def __init__(
        self,
        pupil_channels: int = 3,
        ecg_channels: int = 2,
        eeg_channels: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_classes: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Modality encoders
        self.pupil_encoder = nn.Sequential(
            ConvBlock1D(pupil_channels, embed_dim, kernel_size=7),
            ConvBlock1D(embed_dim, embed_dim, kernel_size=5, stride=2),
            ConvBlock1D(embed_dim, embed_dim, kernel_size=3),
        )
        self.ecg_encoder = nn.Sequential(
            ConvBlock1D(ecg_channels, embed_dim, kernel_size=9),
            ConvBlock1D(embed_dim, embed_dim, kernel_size=5, stride=2),
            ConvBlock1D(embed_dim, embed_dim, kernel_size=3),
        )
        self.eeg_encoder = nn.Sequential(
            nn.Conv1d(eeg_channels, embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            ConvBlock1D(embed_dim, embed_dim, kernel_size=7),
            ConvBlock1D(embed_dim, embed_dim, kernel_size=5, stride=2),
            ConvBlock1D(embed_dim, embed_dim, kernel_size=3),
        )

        # Cross-Attention: Pupil queries attend to ECG and EEG key-values
        self.cross_attn_ecg = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.cross_attn_eeg = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True, dropout=0.1)

        self.norm_pupil = nn.LayerNorm(embed_dim)
        self.norm_ecg = nn.LayerNorm(embed_dim)
        self.norm_eeg = nn.LayerNorm(embed_dim)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Multimodal Fusion Classifier
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x_pupil: torch.Tensor, x_ecg: torch.Tensor, x_eeg: torch.Tensor) -> torch.Tensor:
        # Encode modalities: (B, D, L')
        p_feat = self.pupil_encoder(x_pupil)
        c_feat = self.ecg_encoder(x_ecg)
        e_feat = self.eeg_encoder(x_eeg)

        # Transpose to (B, L', D) for MultiheadAttention
        p_seq = p_feat.transpose(1, 2)
        c_seq = c_feat.transpose(1, 2)
        e_seq = e_feat.transpose(1, 2)

        # Cross-Attention: Pupil queries attend to cardiac and EEG contexts
        p_c_attn, _ = self.cross_attn_ecg(query=p_seq, key=c_seq, value=c_seq)
        p_e_attn, _ = self.cross_attn_eeg(query=p_seq, key=e_seq, value=e_seq)

        p_fused = self.norm_pupil(p_seq + p_c_attn + p_e_attn).transpose(1, 2)
        c_fused = self.norm_ecg(c_seq).transpose(1, 2)
        e_fused = self.norm_eeg(e_seq).transpose(1, 2)

        # Global average pooling
        p_pool = self.pool(p_fused).squeeze(-1)
        c_pool = self.pool(c_fused).squeeze(-1)
        e_pool = self.pool(e_fused).squeeze(-1)

        # Concatenate multi-modal representation
        multimodal_repr = torch.cat([p_pool, c_pool, e_pool], dim=-1)  # (B, 3*D)
        return self.fusion_classifier(multimodal_repr)

"""
Unit tests for Deep Learning AEPR architectures and tensor utilities.
"""

import unittest
import numpy as np
import torch

from src.preprocessing import TrialEpoch
from src.deep_learning_models import (
    MultiScaleConv1DNet,
    BiLSTMAttentionNet,
    DilatedTCNNet,
    CNNTransformerNet,
    FocalLoss,
    extract_multichannel_tensor_from_epoch,
    build_tensor_dataset_from_epochs,
    PupilTimeSeriesDataset,
)
from src.deep_learning_trainer import (
    standardize_channels,
    evaluate_dl_model_stratified_group_cv,
)


def create_synthetic_epoch():
    """Generates a realistic synthetic AEPR trial epoch."""
    time_grid = np.linspace(-0.5, 3.5, 201)  # 50 Hz
    baseline = 3.5  # mm
    dilation = 0.4 * np.exp(-((time_grid - 1.2) ** 2) / 0.3)
    dilation[time_grid < 0] = 0.0
    raw_signal = baseline + dilation + np.random.normal(0, 0.01, size=len(time_grid))
    sub_signal = raw_signal - baseline
    div_signal = (sub_signal / baseline) * 100.0

    return TrialEpoch(
        trial_id=1,
        stimulus="oddball_deviant",
        condition="salient",
        time=time_grid,
        pupil_raw=raw_signal,
        pupil_subtractive=sub_signal,
        pupil_divisive=div_signal,
        baseline_val=baseline,
        missing_ratio=0.0,
        is_valid=True,
    )


class TestDeepLearningModels(unittest.TestCase):

    def setUp(self):
        self.synthetic_epoch = create_synthetic_epoch()
        self.dummy_input = torch.randn(4, 3, 201)  # Batch=4, C=3, T=201

    def test_tensor_extraction(self):
        """Verifies 3-channel tensor extraction (Delta P, % Delta P, d(Delta P)/dt)."""
        tensor = extract_multichannel_tensor_from_epoch(self.synthetic_epoch)
        self.assertIsNotNone(tensor)
        self.assertEqual(tensor.shape, (3, 201))
        self.assertTrue(np.all(np.isfinite(tensor)))

    def test_multiscale_cnn_forward(self):
        """Tests MultiScaleConv1DNet forward pass."""
        model = MultiScaleConv1DNet(in_channels=3, num_filters=16)
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (4, 1))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_bilstm_attention_forward(self):
        """Tests BiLSTMAttentionNet forward pass and attention sum to 1.0."""
        model = BiLSTMAttentionNet(in_channels=3, hidden_dim=32, num_layers=2)
        logits, attn = model(self.dummy_input, return_attention=True)
        self.assertEqual(logits.shape, (4, 1))
        self.assertEqual(attn.shape, (4, 201))
        # Attention weights along time must sum to 1.0
        attn_sums = torch.sum(attn, dim=1)
        self.assertTrue(torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5))

    def test_dilated_tcn_forward(self):
        """Tests DilatedTCNNet forward pass."""
        model = DilatedTCNNet(in_channels=3, num_channels=(16, 32, 64))
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (4, 1))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_cnn_transformer_forward(self):
        """Tests CNNTransformerNet forward pass."""
        model = CNNTransformerNet(in_channels=3, d_model=32, nhead=2, num_layers=1)
        out = model(self.dummy_input)
        self.assertEqual(out.shape, (4, 1))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_focal_loss(self):
        """Tests FocalLoss calculation."""
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        logits = torch.tensor([[2.0], [-2.0], [0.5], [-0.5]], requires_grad=True)
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        loss = criterion(logits, targets)
        self.assertTrue(loss.item() > 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_channel_standardization(self):
        """Verifies channel-wise standardization."""
        X_train = np.random.normal(loc=5.0, scale=2.0, size=(10, 3, 201))
        X_val = np.random.normal(loc=5.0, scale=2.0, size=(5, 3, 201))

        X_tr_norm, X_v_norm, mean, std = standardize_channels(X_train, X_val)
        self.assertEqual(X_tr_norm.shape, (10, 3, 201))
        self.assertEqual(X_v_norm.shape, (5, 3, 201))
        self.assertTrue(np.allclose(np.mean(X_tr_norm, axis=(0, 2)), 0.0, atol=1e-4))
        self.assertTrue(np.allclose(np.std(X_tr_norm, axis=(0, 2)), 1.0, atol=1e-4))


if __name__ == "__main__":
    unittest.main()

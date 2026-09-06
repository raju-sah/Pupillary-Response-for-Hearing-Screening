"""
Unit tests for STEP 15 Multimodal Parsing and Models.
"""

from pathlib import Path
import pytest
import torch
import numpy as np

from src.parser_c import parse_dataset_c_subject
from src.multimodal_models import (
    PupilOnlyNet,
    ECGOnlyNet,
    EEGOnlyNet,
    MultimodalCrossAttentionNet,
)


@pytest.fixture
def sample_dataset_c_subject():
    base_dir = Path(__file__).resolve().parent.parent
    sub_dir = base_dir / "data" / "raw" / "dataset_c_openneuro" / "sub-AB4"
    if not sub_dir.exists():
        pytest.skip("Dataset C sub-AB4 not downloaded.")
    return sub_dir


def test_parser_c_extraction(sample_dataset_c_subject):
    epochs = parse_dataset_c_subject(sample_dataset_c_subject, "sub-AB4", group="Young")
    assert len(epochs) > 0
    ep = epochs[0]
    assert ep.subject_id == "sub-AB4"
    assert ep.pupil.ndim == 1
    assert ep.ecg.ndim == 1
    assert ep.eeg.shape[0] == 4  # 4 central EEG channels
    assert len(ep.pupil) == len(ep.ecg) == ep.eeg.shape[1]
    assert ep.label in [0, 1]


def test_multimodal_model_forward_passes():
    batch_size = 4
    seq_len = 150  # 3.0s at 50Hz

    x_pupil = torch.randn(batch_size, 3, seq_len)
    x_ecg = torch.randn(batch_size, 2, seq_len)
    x_eeg = torch.randn(batch_size, 4, seq_len)

    # 1. PupilOnlyNet
    m_p = PupilOnlyNet()
    out_p = m_p(x_pupil)
    assert out_p.shape == (batch_size, 1)

    # 2. ECGOnlyNet
    m_c = ECGOnlyNet()
    out_c = m_c(x_pupil, x_ecg)
    assert out_c.shape == (batch_size, 1)

    # 3. EEGOnlyNet
    m_e = EEGOnlyNet()
    out_e = m_e(x_pupil, x_ecg, x_eeg)
    assert out_e.shape == (batch_size, 1)

    # 4. MultimodalCrossAttentionNet
    m_fused = MultimodalCrossAttentionNet()
    out_fused = m_fused(x_pupil, x_ecg, x_eeg)
    assert out_fused.shape == (batch_size, 1)

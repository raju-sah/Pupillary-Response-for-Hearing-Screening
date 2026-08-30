"""
Unit tests for AEPR physiological baseline characterization and metrics.
"""

import unittest
import numpy as np
import pandas as pd
from src.preprocessing import TrialEpoch, PreprocessingConfig
from src.physiological_baseline import (
    compute_aepr_metrics_from_epoch,
    extract_resting_pseudo_epochs,
    compute_paired_statistics,
    apply_holm_bonferroni_correction,
    AEPRMetrics
)


class TestPhysiologicalBaseline(unittest.TestCase):

    def test_aepr_metric_extraction(self):
        """Verifies peak amplitude, latency, onset, half-recovery, and AUC calculation."""
        time = np.linspace(-0.5, 3.5, 201)  # 50 Hz
        pupil_raw = np.full(201, 4.0)
        dilation = 0.8 * np.exp(-0.5 * ((time - 1.5) / 0.5) ** 2)
        dilation[time < 0] = 0.0
        pupil_raw += dilation

        sub = pupil_raw - 4.0
        div = (sub / 4.0) * 100.0

        epoch = TrialEpoch(
            trial_id=1,
            stimulus="pure_tone",
            condition="audio_stimulation",
            time=time,
            pupil_raw=pupil_raw,
            pupil_subtractive=sub,
            pupil_divisive=div,
            baseline_val=4.0,
            missing_ratio=0.0,
            is_valid=True
        )

        metrics = compute_aepr_metrics_from_epoch(epoch, subject_id="sub-01")
        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics.baseline_diameter, 4.0, places=2)
        self.assertAlmostEqual(metrics.peak_amplitude, 0.8, places=1)
        self.assertAlmostEqual(metrics.peak_percentage, 20.0, delta=1.0)
        self.assertAlmostEqual(metrics.latency_to_peak_s, 1.5, delta=0.1)
        self.assertTrue(0.2 <= metrics.onset_latency_s <= 1.5)
        self.assertGreater(metrics.half_recovery_s, 1.5)
        self.assertGreater(metrics.auc_response, 0.5)

    def test_extract_resting_pseudo_epochs(self):
        """Verifies pseudo-epoch extraction on continuous resting baseline data."""
        cfg = PreprocessingConfig()
        ts = np.arange(0, 40.0, 0.02)  # 40 seconds at 50 Hz
        df = pd.DataFrame({
            "subject_id": "sub-1F",
            "recording_id": "1F_baseline",
            "timestamp": ts,
            "pupil_left": np.full(len(ts), 120.0),
            "pupil_right": np.full(len(ts), 125.0),
            "condition": "resting_baseline"
        })

        pseudo_epochs = extract_resting_pseudo_epochs(df, cfg, pseudo_interval_s=4.0)
        self.assertGreaterEqual(len(pseudo_epochs), 8)
        for ep in pseudo_epochs:
            self.assertTrue(ep.is_valid)
            self.assertEqual(ep.stimulus, "resting_control")
            self.assertEqual(len(ep.time), len(np.arange(-0.5, 3.5 + 1e-6, 0.02)))

    def test_compute_paired_statistics(self):
        """Verifies paired t-test, Wilcoxon, Cohen's d, and normality checks."""
        np.random.seed(42)
        n = 20
        b_vals = np.random.normal(0.25, 0.05, n)
        a_vals = b_vals + np.random.normal(0.10, 0.03, n)

        res = compute_paired_statistics(a_vals, b_vals, name_a="Deviant", name_b="Standard")
        self.assertEqual(res["n_subjects"], 20)
        self.assertGreater(res["mean_diff"], 0.05)
        self.assertLess(res["t_p"], 0.001)
        self.assertLess(res["wilcox_p"], 0.001)
        self.assertGreater(res["cohen_dz"], 1.0)

    def test_holm_bonferroni_correction(self):
        """Verifies step-down Holm-Bonferroni correction properties."""
        raw_p = [0.01, 0.04, 0.03]
        adj_p = apply_holm_bonferroni_correction(raw_p)
        self.assertAlmostEqual(adj_p[0], 0.03, places=3)
        self.assertAlmostEqual(adj_p[1], 0.06, places=3)
        self.assertAlmostEqual(adj_p[2], 0.06, places=3)


if __name__ == "__main__":
    unittest.main()

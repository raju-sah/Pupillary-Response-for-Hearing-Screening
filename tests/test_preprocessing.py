"""
Unit test suite for AEPR physiological signal preprocessing pipeline.
"""

import unittest
import numpy as np
from src.preprocessing import (
    PreprocessingConfig,
    validate_diameter_bounds,
    detect_velocity_outliers,
    pad_blink_margins,
    interpolate_gaps,
    apply_butterworth_lowpass,
    resample_to_canonical_grid,
    baseline_correct_trial,
    preprocess_pupil_series,
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.default_config = PreprocessingConfig()

    def test_diameter_bounds_validation(self):
        """Verifies that non-physiological values are set to NaN."""
        sig_mm = np.array([1.0, 1.5, 3.5, 8.5, 9.0, 9.5, -1.0, 0.0, np.nan])
        cleaned, mask = validate_diameter_bounds(sig_mm, unit="mm", config=self.default_config)
        self.assertTrue(np.isnan(cleaned[0]))  # 1.0 < 1.5
        self.assertEqual(cleaned[1], 1.5)
        self.assertEqual(cleaned[2], 3.5)
        self.assertEqual(cleaned[3], 8.5)
        self.assertEqual(cleaned[4], 9.0)
        self.assertTrue(np.isnan(cleaned[5]))  # 9.5 > 9.0
        self.assertTrue(np.isnan(cleaned[6]))  # -1.0 <= 0
        self.assertTrue(np.isnan(cleaned[7]))  # 0.0 <= 0
        self.assertTrue(np.isnan(cleaned[8]))  # NaN

        # Test pixel bounds [10.0, 300.0]
        sig_px = np.array([5.0, 10.0, 150.0, 300.0, 350.0])
        cleaned_px, mask_px = validate_diameter_bounds(sig_px, unit="pixel", config=self.default_config)
        self.assertTrue(np.isnan(cleaned_px[0]))
        self.assertEqual(cleaned_px[1], 10.0)
        self.assertEqual(cleaned_px[2], 150.0)
        self.assertEqual(cleaned_px[3], 300.0)
        self.assertTrue(np.isnan(cleaned_px[4]))

    def test_velocity_outlier_detection(self):
        """Verifies detection of sudden unphysiological dilation spikes."""
        ts = np.linspace(0, 1.0, 50)  # 50 Hz, dt = 0.02s
        sig = np.full(50, 4.0)
        sig[25] = 8.0
        vel_mask = detect_velocity_outliers(sig, ts, unit="mm", config=self.default_config)
        self.assertTrue(vel_mask[24] or vel_mask[25])

    def test_blink_margin_padding(self):
        """Verifies that blink intervals are padded 50ms before and 100ms after."""
        ts = np.arange(0, 1.0, 0.01)  # 100 Hz, 10 ms steps
        invalid = np.zeros(100, dtype=bool)
        invalid[40:51] = True

        padded = pad_blink_margins(invalid, ts, pre_margin_s=0.050, post_margin_s=0.100)
        self.assertTrue(padded[35])
        self.assertTrue(padded[39])
        self.assertTrue(padded[55])
        self.assertTrue(padded[60])
        self.assertFalse(padded[30])
        self.assertFalse(padded[65])

    def test_gap_interpolation_short_vs_long(self):
        """Verifies that gaps <= 500ms are interpolated, and gaps > 500ms remain NaN."""
        ts = np.arange(0, 2.0, 0.02)  # 50 Hz, 20ms steps, 100 samples
        sig = np.sin(2 * np.pi * 0.5 * ts) + 4.0

        invalid_mask = np.zeros(100, dtype=bool)
        invalid_mask[20:30] = True
        invalid_mask[50:90] = True

        interp_sig, unrec_mask = interpolate_gaps(sig, ts, invalid_mask, max_gap_s=0.500)

        self.assertTrue(np.all(np.isfinite(interp_sig[20:30])))
        self.assertFalse(np.any(unrec_mask[20:30]))
        self.assertTrue(np.all(np.isnan(interp_sig[50:90])))
        self.assertTrue(np.all(unrec_mask[50:90]))

    def test_butterworth_lowpass_filter(self):
        """Verifies low-pass filter suppresses high frequency noise while preserving DC trend."""
        fs = 50.0
        ts = np.arange(0, 4.0, 1.0 / fs)
        clean_wave = 4.0 + np.sin(2 * np.pi * 0.5 * ts)
        hf_noise = 0.5 * np.sin(2 * np.pi * 20.0 * ts)
        noisy = clean_wave + hf_noise

        filtered = apply_butterworth_lowpass(noisy, fs=fs, cutoff_hz=4.0, order=3)
        mse_noisy = np.mean((noisy - clean_wave) ** 2)
        mse_filtered = np.mean((filtered - clean_wave) ** 2)
        self.assertLess(mse_filtered, 0.2 * mse_noisy)

    def test_canonical_resampling_grid(self):
        """Verifies resampling onto a uniform 50 Hz grid."""
        ts_61hz = np.arange(0, 2.0, 1.0 / 61.5)
        sig_61hz = 4.0 + 0.5 * np.sin(2 * np.pi * 1.0 * ts_61hz)

        resampled, grid_ts = resample_to_canonical_grid(sig_61hz, ts_61hz, target_fs=50.0)
        dt_grid = np.diff(grid_ts)
        self.assertTrue(np.allclose(dt_grid, 0.020, atol=1e-5))
        self.assertEqual(len(resampled), len(grid_ts))
        self.assertTrue(np.all(np.isfinite(resampled)))

    def test_baseline_correction_subtractive_and_divisive(self):
        """Verifies subtractive and stabilized divisive baseline correction."""
        ep_time = np.linspace(-0.5, 3.5, 201)
        ep_pupil = np.full(201, 4.0)
        ep_pupil[ep_time > 0] = 4.8

        sub, div, b_val, is_valid, warn = baseline_correct_trial(
            ep_pupil, ep_time, baseline_window=(-0.5, 0.0), unit="mm", config=self.default_config
        )
        self.assertTrue(is_valid)
        self.assertAlmostEqual(b_val, 4.0, places=3)
        self.assertTrue(np.allclose(sub[ep_time <= 0], 0.0, atol=1e-3))
        self.assertTrue(np.allclose(sub[ep_time > 0], 0.8, atol=1e-3))
        self.assertTrue(np.allclose(div[ep_time > 0], 20.0, atol=1e-2))

    def test_baseline_correction_failure_on_missing_baseline(self):
        """Verifies that a trial with NaN baseline is flagged as invalid."""
        ep_time = np.linspace(-0.5, 3.5, 201)
        ep_pupil = np.full(201, 4.0)
        ep_pupil[ep_time <= 0] = np.nan

        sub, div, b_val, is_valid, reason = baseline_correct_trial(
            ep_pupil, ep_time, baseline_window=(-0.5, 0.0), unit="mm", config=self.default_config
        )
        self.assertFalse(is_valid)
        self.assertIn("insufficient_baseline_samples", reason)


if __name__ == "__main__":
    unittest.main()

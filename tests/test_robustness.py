"""
Unit tests for STEP 11 Robustness and Perturbation module (src/robustness.py).
"""

import unittest
import numpy as np
from src.robustness import (
    truncate_epoch_tensors,
    downsample_epoch_tensors,
    inject_artificial_blink_dropout,
    inject_sensor_noise,
)


class TestRobustnessModule(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.N, self.C, self.T = 10, 3, 201
        self.time_grid = np.linspace(-0.5, 3.5, 201)
        self.X = np.sin(np.linspace(0, 10, self.T))[None, None, :] * np.ones((self.N, self.C, self.T))

    def test_truncate_epoch_tensors(self):
        # Truncate at max_time_s = 1.0s
        X_trunc, sub_grid = truncate_epoch_tensors(self.X, self.time_grid, max_time_s=1.0)
        self.assertEqual(X_trunc.shape[0], self.N)
        self.assertEqual(X_trunc.shape[1], self.C)
        self.assertTrue(sub_grid[-1] <= 1.0)
        self.assertTrue(X_trunc.shape[2] < self.T)
        self.assertEqual(len(sub_grid), X_trunc.shape[2])

    def test_downsample_epoch_tensors(self):
        # Downsample 50 Hz -> 10 Hz
        X_resampled, new_grid = downsample_epoch_tensors(self.X, self.time_grid, target_fs=10.0)
        self.assertEqual(X_resampled.shape[0], self.N)
        self.assertEqual(X_resampled.shape[1], self.C)
        # Duration = 4.0s -> ~41 points at 10 Hz
        self.assertEqual(X_resampled.shape[2], 41)
        self.assertEqual(len(new_grid), 41)
        self.assertAlmostEqual(new_grid[0], -0.5, places=3)
        self.assertAlmostEqual(new_grid[-1], 3.5, places=3)

    def test_inject_artificial_blink_dropout(self):
        # Inject 20% dropout
        X_corrupted = inject_artificial_blink_dropout(
            self.X,
            dropout_fraction=0.20,
            burst_duration_samples=(10, 20),
            interpolation="zero",
            rng=np.random.RandomState(42)
        )
        self.assertEqual(X_corrupted.shape, self.X.shape)
        # Verify that zero-interpolation created zeros
        self.assertTrue(np.any(X_corrupted == 0.0))

    def test_inject_sensor_noise(self):
        X_noisy = inject_sensor_noise(self.X, noise_sigma=0.10, rng=np.random.RandomState(42))
        self.assertEqual(X_noisy.shape, self.X.shape)
        self.assertFalse(np.allclose(X_noisy, self.X))
        # Zero noise preserves original
        X_clean = inject_sensor_noise(self.X, noise_sigma=0.0)
        self.assertTrue(np.allclose(X_clean, self.X))


if __name__ == "__main__":
    unittest.main()

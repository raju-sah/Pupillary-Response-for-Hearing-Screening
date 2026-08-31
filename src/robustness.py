"""
Module for STEP 11: Robustness, Early Detection Latency, and Signal Perturbation Experiments.

Provides functions to test:
1. Temporal Window Truncation (Early Detection Latency): How early can an AEPR be detected?
2. Sampling Rate Downsampling (50 Hz -> 25 Hz -> 10 Hz -> 5 Hz).
3. Missing Data & Artificial Blink Burst Dropout (5% to 40%).
4. Additive Sensor Noise Perturbation (Gaussian jitter & amplitude scaling).
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from scipy import interpolate
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, brier_score_loss


def truncate_epoch_tensors(
    X: np.ndarray,
    time_grid: np.ndarray,
    max_time_s: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truncates multi-channel epoch tensors to an observation window [-0.5s, max_time_s].
    
    Args:
        X: Tensor of shape (N, C, T)
        time_grid: 1D array of time stamps of length T (e.g. [-0.5 to 3.5s])
        max_time_s: Upper time bound in seconds (e.g. 1.0, 1.5, 2.0s)
        
    Returns:
        X_trunc: Truncated tensor of shape (N, C, T_sub)
        sub_grid: Truncated time grid
    """
    mask = time_grid <= max_time_s
    if not np.any(mask):
        mask[0] = True
    X_trunc = X[:, :, mask]
    sub_grid = time_grid[mask]
    return X_trunc, sub_grid


def downsample_epoch_tensors(
    X: np.ndarray,
    orig_time_grid: np.ndarray,
    target_fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsamples multi-channel epoch tensors from original frequency to target_fs.
    
    Args:
        X: Tensor of shape (N, C, T)
        orig_time_grid: 1D array of original timestamps (e.g., 50 Hz -> 201 points)
        target_fs: Target sampling frequency in Hz (e.g., 25, 10, 5)
        
    Returns:
        X_resampled: Tensor resampled to target_fs
        new_time_grid: Corresponding new timestamps
    """
    t_start, t_end = orig_time_grid[0], orig_time_grid[-1]
    duration = t_end - t_start
    n_points = int(np.round(duration * target_fs)) + 1
    new_time_grid = np.linspace(t_start, t_end, n_points)
    
    N, C, _ = X.shape
    X_resampled = np.zeros((N, C, n_points), dtype=X.dtype)
    
    # Vectorized / 1D linear interpolation across trials
    for c in range(C):
        f = interpolate.interp1d(orig_time_grid, X[:, c, :], axis=1, kind="linear", fill_value="extrapolate")
        X_resampled[:, c, :] = f(new_time_grid)
        
    return X_resampled, new_time_grid


def inject_artificial_blink_dropout(
    X: np.ndarray,
    dropout_fraction: float,
    burst_duration_samples: Tuple[int, int] = (10, 25),
    interpolation: str = "linear",
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Simulates severe eye-tracking corruption by masking random contiguous segments
    (representing un-interpolated blinks or tracking loss) and applying recovery.
    
    Args:
        X: Tensor of shape (N, C, T)
        dropout_fraction: Proportion of timepoints to corrupt (e.g. 0.10 for 10% loss)
        burst_duration_samples: Range of contiguous burst lengths in samples
        interpolation: 'linear', 'spline', or 'zero' (unrecovered masked dropout)
        rng: RandomState instance
    """
    if dropout_fraction <= 0.0:
        return X.copy()
        
    if rng is None:
        rng = np.random.RandomState(42)
        
    N, C, T = X.shape
    X_corrupted = X.copy()
    target_corrupt_points = int(np.round(T * dropout_fraction))
    
    for i in range(N):
        corrupt_mask = np.zeros(T, dtype=bool)
        total_corrupted = 0
        
        while total_corrupted < target_corrupt_points:
            burst_len = rng.randint(burst_duration_samples[0], burst_duration_samples[1] + 1)
            # Avoid masking tone onset exactly at t=0
            start_idx = rng.randint(0, max(T - burst_len, 1))
            end_idx = min(start_idx + burst_len, T)
            corrupt_mask[start_idx:end_idx] = True
            total_corrupted = np.sum(corrupt_mask)
            
        for c in range(C):
            sig = X_corrupted[i, c, :].copy()
            if interpolation == "zero":
                sig[corrupt_mask] = 0.0
            elif interpolation == "linear":
                valid_idx = np.where(~corrupt_mask)[0]
                if len(valid_idx) >= 2:
                    sig[corrupt_mask] = np.interp(np.where(corrupt_mask)[0], valid_idx, sig[valid_idx])
            X_corrupted[i, c, :] = sig
            
    return X_corrupted


def inject_sensor_noise(
    X: np.ndarray,
    noise_sigma: float,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Injects additive zero-mean Gaussian noise scaled relative to signal standard deviation.
    
    Args:
        X: Tensor of shape (N, C, T)
        noise_sigma: Noise standard deviation multiplier (e.g., 0.05 for 5% SNR degradation)
        rng: RandomState instance
    """
    if noise_sigma <= 0.0:
        return X.copy()
        
    if rng is None:
        rng = np.random.RandomState(42)
        
    sig_std = np.std(X, axis=(0, 2), keepdims=True) + 1e-6
    noise = rng.normal(loc=0.0, scale=noise_sigma * sig_std, size=X.shape)
    return X + noise

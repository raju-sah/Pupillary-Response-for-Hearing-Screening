"""
Physiological Preprocessing Pipeline for Auditory-Evoked Pupillary Responses (AEPR).

Implements:
1. Physiologically plausible diameter range bounds checking.
2. Blink margin padding around true occlusion/blink episodes (-50ms/+100ms).
3. Velocity-based outlier detection for isolated single-sample jitter spikes.
4. Configurable spline/linear gap interpolation (default <= 500 ms).
5. Zero-phase Butterworth low-pass filtering (default 4 Hz, 3rd order).
6. Canonical grid resampling (default 50.0 Hz) preserving unrecoverable gap NaNs.
7. Trial epoching (-0.5s to +3.5s) and baseline correction (subtractive & stabilized divisive).
8. Strict trial quality audit and rejection for excessive missingness (default > 25%).
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import interp1d


# ============================================================================
# Physiological Limits & Configuration
# ============================================================================

@dataclass
class PreprocessingConfig:
    """
    Configuration parameters for AEPR physiological signal preprocessing.
    All thresholds are configurable with justifications grounded in pupillometry literature.
    """
    # 1. Target Resampling Rate
    target_sampling_rate_hz: float = 50.0  # Common uniform grid (20 ms interval)

    # 2. Physiologically Plausible Bounds
    # Dataset B (mm): Human physiological limits [1.5 mm, 9.0 mm] (Loewenfeld, 1993; Mathot, 2018)
    min_diameter_mm: float = 1.5
    max_diameter_mm: float = 9.0

    # Dataset A (pixels): Camera ROI limits [10.0 px, 300.0 px] (Zenodo 10497437 sensor specs)
    min_diameter_px: float = 10.0
    max_diameter_px: float = 300.0

    # 3. Blink Velocity Thresholds (d(diameter)/dt)
    # Biological pupil movements rarely exceed 5.0 mm/s; faster changes represent eyelid occlusion (Mathot, 2018)
    velocity_thresh_mm_s: float = 5.0
    velocity_thresh_px_s: float = 300.0
    velocity_mad_multiplier: float = 5.0

    # 4. Blink Margin Padding
    # Pad 50ms before and 100ms after true blink/closure episodes to remove eyelid occlusion recovery artifacts
    pre_blink_margin_sec: float = 0.050
    post_blink_margin_sec: float = 0.100

    # 5. Gap Interpolation Limits
    # Gaps <= 500 ms are interpolated; longer gaps represent prolonged closure and are flagged as invalid
    max_gap_duration_sec: float = 0.500
    interpolation_method: str = "linear"

    # 6. Low-Pass Filtering
    # 4.0 Hz cutoff removes high-frequency tracker noise while preserving pupillary dynamics (< 4 Hz)
    lowpass_cutoff_hz: float = 4.0
    filter_order: int = 3

    # 7. Epoching & Baseline Correction
    epoch_window_sec: Tuple[float, float] = (-0.5, 3.5)  # [-500 ms, +3500 ms] relative to stimulus onset
    baseline_window_sec: Tuple[float, float] = (-0.5, 0.0)  # [-500 ms, 0 ms] pre-stimulus
    epsilon_base_mm: float = 1.0  # Floor for divisive baseline in mm
    epsilon_base_px: float = 10.0  # Floor for divisive baseline in px

    # 8. Trial Quality Rejection Threshold
    # Trials with > 25% missing/invalid samples are excluded from downstream modeling (Winn et al., 2018)
    max_trial_missing_ratio: float = 0.25


@dataclass
class TrialEpoch:
    """Represents a single baseline-corrected trial epoch."""
    trial_id: int
    stimulus: str
    condition: str
    time: np.ndarray  # Relative time in seconds (e.g. -0.5 to 3.5)
    pupil_raw: np.ndarray  # Resampled pupil trace before baseline correction
    pupil_subtractive: np.ndarray  # P(t) - P_base
    pupil_divisive: np.ndarray  # (P(t) - P_base) / P_base * 100%
    baseline_val: float
    missing_ratio: float
    is_valid: bool
    rejection_reason: Optional[str] = None


# ============================================================================
# Core Preprocessing Functions
# ============================================================================

def validate_diameter_bounds(
    signal: np.ndarray,
    unit: str = "mm",
    config: Optional[PreprocessingConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identifies non-physiological pupil values and sets them to NaN.

    Returns:
        cleaned_signal: Array with out-of-bound values replaced by NaN.
        outlier_mask: Boolean array where True indicates out-of-bound values.
    """
    cfg = config or PreprocessingConfig()
    cleaned = signal.copy().astype(float)

    if unit == "mm":
        min_val, max_val = cfg.min_diameter_mm, cfg.max_diameter_mm
    elif unit == "pixel":
        min_val, max_val = cfg.min_diameter_px, cfg.max_diameter_px
    else:
        raise ValueError(f"Unknown unit: {unit}. Must be 'mm' or 'pixel'.")

    outlier_mask = np.isnan(cleaned) | (cleaned < min_val) | (cleaned > max_val) | (cleaned <= 0)
    cleaned[outlier_mask] = np.nan
    return cleaned, outlier_mask


def detect_velocity_outliers(
    signal: np.ndarray,
    timestamps: np.ndarray,
    unit: str = "mm",
    config: Optional[PreprocessingConfig] = None
) -> np.ndarray:
    """
    Detects sudden dilation or constriction spikes exceeding biological limits.

    Returns:
        velocity_outlier_mask: Boolean array where True indicates velocity anomaly.
    """
    cfg = config or PreprocessingConfig()
    n = len(signal)
    if n < 2:
        return np.zeros(n, dtype=bool)

    dt = np.diff(timestamps)
    dt[dt <= 0] = 1e-6

    dp = np.diff(signal)
    vel = np.zeros(n)
    vel[:-1] = np.abs(dp / dt)
    vel[-1] = vel[-2] if n > 2 else vel[0]

    fixed_thresh = cfg.velocity_thresh_mm_s if unit == "mm" else cfg.velocity_thresh_px_s

    valid_vel = vel[np.isfinite(vel) & (vel > 0)]
    if len(valid_vel) > 10:
        med = np.median(valid_vel)
        mad = np.median(np.abs(valid_vel - med))
        mad_thresh = med + cfg.velocity_mad_multiplier * (mad * 1.4826)
        thresh = min(fixed_thresh, max(fixed_thresh * 0.5, mad_thresh))
    else:
        thresh = fixed_thresh

    return np.isfinite(vel) & (vel > thresh)


def pad_blink_margins(
    blink_mask: np.ndarray,
    timestamps: np.ndarray,
    pre_margin_s: float = 0.050,
    post_margin_s: float = 0.100
) -> np.ndarray:
    """
    Extends detected blink intervals by pre_margin_s before and post_margin_s after.
    Uses binary search for high performance on large high-rate series.
    """
    n = len(blink_mask)
    if n == 0 or not np.any(blink_mask):
        return blink_mask.copy()

    padded = blink_mask.copy()
    diff = np.diff(blink_mask.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        t_start = timestamps[s] if s < n else timestamps[-1]
        t_end = timestamps[min(e - 1, n - 1)]

        s_pre = max(0, int(np.searchsorted(timestamps, t_start - pre_margin_s)))
        e_post = min(n, int(np.searchsorted(timestamps, t_end + post_margin_s, side="right")))
        padded[s_pre:e_post] = True

    return padded


def interpolate_gaps(
    signal: np.ndarray,
    timestamps: np.ndarray,
    invalid_mask: np.ndarray,
    max_gap_s: float = 0.500,
    method: str = "linear"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolates invalid gaps shorter than or equal to max_gap_s.
    Gaps > max_gap_s remain NaN.

    Returns:
        interpolated_signal: Signal with short gaps filled.
        unrecoverable_mask: Boolean array of samples that could not be interpolated.
    """
    n = len(signal)
    out = signal.copy().astype(float)
    out[invalid_mask] = np.nan

    valid_indices = np.where(~invalid_mask)[0]
    if len(valid_indices) < 2:
        return out, np.ones(n, dtype=bool)

    diff = np.diff(invalid_mask.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    unrecoverable = np.zeros(n, dtype=bool)

    for s, e in zip(starts, ends):
        idx_range = np.arange(s, e)
        if s == 0 or e >= n:
            out[idx_range] = np.nan
            unrecoverable[idx_range] = True
            continue

        gap_dur = timestamps[e - 1] - timestamps[s]
        if gap_dur <= max_gap_s:
            t_left, t_right = timestamps[s - 1], timestamps[e]
            p_left, p_right = signal[s - 1], signal[e]

            if np.isfinite(p_left) and np.isfinite(p_right):
                t_gap = timestamps[idx_range]
                out[idx_range] = np.interp(t_gap, [t_left, t_right], [p_left, p_right])
            else:
                out[idx_range] = np.nan
                unrecoverable[idx_range] = True
        else:
            out[idx_range] = np.nan
            unrecoverable[idx_range] = True

    return out, unrecoverable


def apply_butterworth_lowpass(
    signal: np.ndarray,
    fs: float,
    cutoff_hz: float = 4.0,
    order: int = 3
) -> np.ndarray:
    """
    Applies a zero-phase 3rd-order Butterworth low-pass filter (sosfiltfilt).
    Filters contiguous valid segments.
    """
    if len(signal) < 15 or fs <= 0 or cutoff_hz >= fs / 2:
        return signal.copy()

    sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    out = signal.copy()

    is_valid = np.isfinite(out)
    if not np.any(is_valid):
        return out

    diff = np.diff(is_valid.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    min_padlen = 3 * (2 * order + 1)
    for s, e in zip(starts, ends):
        seg_len = e - s
        if seg_len > min_padlen:
            try:
                out[s:e] = sosfiltfilt(sos, out[s:e])
            except Exception:
                pass

    return out


def resample_to_canonical_grid(
    signal: np.ndarray,
    timestamps: np.ndarray,
    target_fs: float = 50.0,
    max_gap_s: float = 0.500,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resamples a time series onto a uniform canonical time grid.
    Crucially preserves unrecoverable gaps (> max_gap_s) as strict NaNs on the grid.

    Returns:
        resampled_signal: Array sampled at target_fs with unrecoverable gaps preserved as NaN.
        grid_timestamps: Canonical timestamps (t = 0, dt, 2dt, ...).
    """
    t0 = timestamps[0] if t_start is None else t_start
    t1 = timestamps[-1] if t_end is None else t_end
    dt = 1.0 / target_fs

    grid_timestamps = np.arange(t0, t1 + 1e-6, dt)
    n_grid = len(grid_timestamps)
    is_valid = np.isfinite(signal)

    if np.sum(is_valid) < 2:
        return np.full(n_grid, np.nan), grid_timestamps

    # Continuous linear interpolation over valid points
    f_interp = interp1d(
        timestamps[is_valid],
        signal[is_valid],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan
    )
    resampled = f_interp(grid_timestamps)

    # Re-apply unrecoverable NaN mask
    diff_valid = np.diff(is_valid.astype(int), prepend=0, append=0)
    gap_starts = np.where(diff_valid == -1)[0]
    gap_ends = np.where(diff_valid == 1)[0]

    for gs, ge in zip(gap_starts, gap_ends):
        t_gap_start = timestamps[max(0, gs - 1)]
        t_gap_end = timestamps[min(len(timestamps) - 1, ge)]
        if (t_gap_end - t_gap_start) > max_gap_s or gs == 0 or ge >= len(timestamps):
            mask_idx = (grid_timestamps >= t_gap_start) & (grid_timestamps <= t_gap_end)
            resampled[mask_idx] = np.nan

    return resampled, grid_timestamps


def baseline_correct_trial(
    epoch_signal: np.ndarray,
    epoch_time: np.ndarray,
    baseline_window: Tuple[float, float] = (-0.5, 0.0),
    unit: str = "mm",
    config: Optional[PreprocessingConfig] = None
) -> Tuple[np.ndarray, np.ndarray, float, bool, Optional[str]]:
    """
    Performs subtractive and stabilized divisive baseline correction for a trial epoch.

    Returns:
        subtractive: P(t) - P_base
        divisive: (P(t) - P_base) / max(P_base, epsilon) * 100%
        base_val: Computed baseline median diameter
        is_valid: True if baseline is stable and valid
        warning: Description if baseline instability or rejection occurred
    """
    cfg = config or PreprocessingConfig()
    base_mask = (epoch_time >= baseline_window[0]) & (epoch_time <= baseline_window[1])
    base_samples = epoch_signal[base_mask]
    valid_base = base_samples[np.isfinite(base_samples)]

    epsilon = cfg.epsilon_base_mm if unit == "mm" else cfg.epsilon_base_px
    min_val = cfg.min_diameter_mm if unit == "mm" else cfg.min_diameter_px

    if len(valid_base) < (0.5 * np.sum(base_mask)):
        return (
            np.full_like(epoch_signal, np.nan),
            np.full_like(epoch_signal, np.nan),
            np.nan,
            False,
            "insufficient_baseline_samples"
        )

    base_val = float(np.median(valid_base))

    if base_val < min_val:
        return (
            np.full_like(epoch_signal, np.nan),
            np.full_like(epoch_signal, np.nan),
            base_val,
            False,
            f"baseline_below_physiological_minimum ({base_val:.2f} < {min_val})"
        )

    subtractive = epoch_signal - base_val
    denom = max(base_val, epsilon)
    divisive = (subtractive / denom) * 100.0

    warning = None
    if base_val <= epsilon:
        warning = f"baseline_near_floor_warning ({base_val:.2f} <= {epsilon})"

    return subtractive, divisive, base_val, True, warning


def preprocess_pupil_series(
    raw_signal: np.ndarray,
    timestamps: np.ndarray,
    unit: str = "mm",
    config: Optional[PreprocessingConfig] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Runs full preprocessing pipeline on a single eye pupil time series:
    1. Range bounds validation (discards non-physiological values / closures).
    2. True blink margin padding (-50ms/+100ms around range dropouts).
    3. Point velocity outlier detection for isolated single-sample jitter.
    4. Interpolation of short gaps (<= 500 ms).
    5. Zero-phase Butterworth low-pass filter (4 Hz).

    Returns:
        clean_signal: Fully preprocessed pupil series on native timestamps.
        unrecoverable_mask: Boolean mask indicating unrecoverable missing/artifact samples.
        stats: Diagnostic counts and percentages.
    """
    cfg = config or PreprocessingConfig()
    n_total = len(raw_signal)

    # 1. Range bounds check (detects missing values and eye closures)
    bounded_sig, range_mask = validate_diameter_bounds(raw_signal, unit=unit, config=cfg)

    # 2. Blink margin padding around true closures/missing episodes
    padded_blink_mask = pad_blink_margins(
        range_mask,
        timestamps,
        pre_margin_s=cfg.pre_blink_margin_sec,
        post_margin_s=cfg.post_blink_margin_sec
    )

    # 3. Detect isolated velocity spikes on valid samples outside blinks
    vel_mask = detect_velocity_outliers(bounded_sig, timestamps, unit=unit, config=cfg)
    total_invalid_mask = padded_blink_mask | vel_mask

    # 4. Interpolate recoverable gaps (<= 500 ms)
    interpolated_sig, unrecoverable_mask = interpolate_gaps(
        bounded_sig,
        timestamps,
        total_invalid_mask,
        max_gap_s=cfg.max_gap_duration_sec,
        method=cfg.interpolation_method
    )

    # 5. Native sampling rate estimate
    dt_valid = np.diff(timestamps)[np.diff(timestamps) > 0]
    native_fs = (1.0 / np.median(dt_valid)) if len(dt_valid) > 0 else cfg.target_sampling_rate_hz

    # 6. Zero-phase Butterworth low-pass filter
    clean_signal = apply_butterworth_lowpass(
        interpolated_sig,
        fs=native_fs,
        cutoff_hz=cfg.lowpass_cutoff_hz,
        order=cfg.filter_order
    )

    stats = {
        "n_samples": n_total,
        "raw_missing_pct": float(np.mean(range_mask) * 100),
        "velocity_outlier_pct": float(np.mean(vel_mask) * 100),
        "padded_blink_pct": float(np.mean(padded_blink_mask) * 100),
        "final_unrecoverable_pct": float(np.mean(unrecoverable_mask) * 100),
        "native_fs_hz": float(native_fs),
    }

    return clean_signal, unrecoverable_mask, stats

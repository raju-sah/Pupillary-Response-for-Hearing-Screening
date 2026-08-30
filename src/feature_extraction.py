"""
Feature Extraction Module for Classical ML AEPR Baselines.

Extracts a comprehensive 25-dimensional domain-informed feature set from
preprocessed single-trial pupillometry epochs:
1. Morphological & Amplitude Features (8)
2. Temporal & Latency Dynamics (7)
3. Curve Shape & Statistical Distribution (6)
4. Spectral & Frequency-Domain Features (4)

Also provides unit-invariant feature subsets for cross-dataset zero-shot transfer
(handling pixel vs physical mm scale differences) and raw downsampled time-series representations.
"""

from typing import Dict, Any, List, Optional, Tuple, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy import stats
from scipy.signal import periodogram

from src.preprocessing import TrialEpoch, PreprocessingConfig


# Complete 25-feature registry
FEATURE_NAMES_25: List[str] = [
    # A. Morphological & Amplitude Features (8)
    "baseline_diameter_mean",
    "baseline_diameter_std",
    "peak_dilation_amplitude",
    "peak_dilation_percent",
    "mean_response_amplitude",
    "auc_response_trapezoid",
    "initial_constriction_depth",
    "end_recovery_amplitude",
    # B. Temporal & Latency Dynamics (7)
    "latency_to_peak_s",
    "onset_latency_10pct_s",
    "time_to_half_recovery_s",
    "dilation_duration_s",
    "latency_to_constriction_s",
    "max_dilation_velocity",
    "time_to_max_velocity_s",
    # C. Curve Shape & Distribution (6)
    "response_slope_onset_to_peak",
    "response_variance",
    "response_skewness",
    "response_kurtosis",
    "half_rise_time_s",
    "rebound_slope",
    # D. Spectral & Frequency-Domain (4)
    "spectral_power_low",
    "spectral_power_mid",
    "spectral_power_high",
    "spectral_centroid",
]

MORPHOLOGICAL_FEATURES: List[str] = [
    "baseline_diameter_mean",
    "baseline_diameter_std",
    "peak_dilation_amplitude",
    "peak_dilation_percent",
    "mean_response_amplitude",
    "auc_response_trapezoid",
    "initial_constriction_depth",
    "end_recovery_amplitude",
]

DYNAMICS_FEATURES: List[str] = [
    "latency_to_peak_s",
    "onset_latency_10pct_s",
    "time_to_half_recovery_s",
    "dilation_duration_s",
    "latency_to_constriction_s",
    "max_dilation_velocity",
    "time_to_max_velocity_s",
]

SHAPE_SPECTRAL_FEATURES: List[str] = [
    "response_slope_onset_to_peak",
    "response_variance",
    "response_skewness",
    "response_kurtosis",
    "half_rise_time_s",
    "rebound_slope",
    "spectral_power_low",
    "spectral_power_mid",
    "spectral_power_high",
    "spectral_centroid",
]

# Strictly unit-invariant features for cross-dataset transfer (Dataset B mm -> Dataset A px)
# Excludes absolute scale amplitude metrics in raw mm or px
UNIT_INVARIANT_FEATURES: List[str] = [
    "peak_dilation_percent",
    "latency_to_peak_s",
    "onset_latency_10pct_s",
    "time_to_half_recovery_s",
    "dilation_duration_s",
    "latency_to_constriction_s",
    "time_to_max_velocity_s",
    "response_slope_onset_to_peak",
    "response_skewness",
    "response_kurtosis",
    "half_rise_time_s",
    "spectral_power_low",
    "spectral_power_mid",
    "spectral_power_high",
    "spectral_centroid",
]


def extract_features_from_epoch(
    epoch: TrialEpoch,
    min_latency_s: float = 0.20,
    max_latency_s: float = 3.50,
    fs: float = 50.0
) -> Optional[Dict[str, float]]:
    """
    Extracts all 25 domain-informed AEPR features from a single TrialEpoch.

    Parameters:
        epoch: Preprocessed and baseline-corrected TrialEpoch.
        min_latency_s: Start of active physiological AEPR window (default: 0.20s).
        max_latency_s: End of evaluation window (default: 3.50s).
        fs: Target sampling rate in Hz (default: 50.0 Hz).

    Returns:
        Dictionary of 25 feature name-value pairs, or None if epoch is invalid or empty.
    """
    if not epoch.is_valid:
        return None

    t = epoch.time
    y_sub = epoch.pupil_subtractive
    y_div = epoch.pupil_divisive
    b_val = float(epoch.baseline_val)
    raw = epoch.pupil_raw

    # Pre-stimulus window [-0.5s, 0.0s]
    pre_mask = (t >= -0.50) & (t <= 0.0)
    pre_raw = raw[pre_mask & np.isfinite(raw)]
    b_mean = float(np.nanmean(pre_raw)) if len(pre_raw) > 0 else b_val
    b_std = float(np.nanstd(pre_raw, ddof=1)) if len(pre_raw) > 1 else 0.0

    # Post-stimulus active evaluation window [min_latency_s, max_latency_s]
    post_mask = (t >= min_latency_s) & (t <= max_latency_s)
    if np.sum(post_mask) < 5 or np.sum(np.isfinite(y_sub[post_mask])) < 5:
        return None

    t_post = t[post_mask]
    y_post_sub = y_sub[post_mask]
    y_post_div = y_div[post_mask]

    # Clean finite arrays for post-stimulus window
    valid_post = np.isfinite(y_post_sub)
    if not np.any(valid_post):
        return None

    # 1. Peak Dilation & Latency
    peak_amp = float(np.nanmax(y_post_sub))
    peak_idx = int(np.nanargmax(y_post_sub))
    t_peak = float(t_post[peak_idx])

    if np.isfinite(y_post_div[peak_idx]):
        peak_pct = float(y_post_div[peak_idx])
    else:
        denom = max(abs(b_val), 1e-2)
        peak_pct = float((peak_amp / denom) * 100.0)

    # 2. Onset Latency (10% of peak dilation)
    onset_thresh = 0.10 * peak_amp if peak_amp > 0 else 0.0
    pre_peak_mask = (t_post <= t_peak) & (y_post_sub >= onset_thresh)
    if np.any(pre_peak_mask):
        t_onset = float(t_post[pre_peak_mask][0])
    else:
        t_onset = float(min_latency_s)

    # 3. Half-Rise Time (time to 50% of peak dilation)
    half_rise_thresh = 0.50 * peak_amp if peak_amp > 0 else 0.0
    half_rise_mask = (t_post <= t_peak) & (y_post_sub >= half_rise_thresh)
    if np.any(half_rise_mask):
        t_half_rise = float(t_post[half_rise_mask][0])
    else:
        t_half_rise = float(min_latency_s)

    # 4. Time to Half-Recovery (post-peak decay to 50%)
    half_rec_thresh = 0.50 * peak_amp if peak_amp > 0 else 0.0
    post_peak_mask = (t_post >= t_peak) & (y_post_sub <= half_rec_thresh)
    if np.any(post_peak_mask):
        t_half_rec = float(t_post[post_peak_mask][0])
    else:
        t_half_rec = float(max_latency_s)

    dilation_duration = max(0.0, t_half_rec - t_onset)

    # 5. Initial Constriction Reflex Dip ([0.1s, 0.8s])
    constrict_mask = (t >= 0.10) & (t <= 0.80)
    t_constrict = t[constrict_mask]
    y_constrict = y_sub[constrict_mask]
    if np.any(np.isfinite(y_constrict)):
        min_c_idx = int(np.nanargmin(y_constrict))
        initial_constriction_depth = float(y_constrict[min_c_idx])
        latency_to_constriction = float(t_constrict[min_c_idx])
    else:
        initial_constriction_depth = 0.0
        latency_to_constriction = 0.10

    # 6. Response Window [0.5s, 2.5s] Mean Amplitude
    mid_mask = (t >= 0.50) & (t <= 2.50)
    y_mid = y_sub[mid_mask & np.isfinite(y_sub)]
    mean_response_amp = float(np.mean(y_mid)) if len(y_mid) > 0 else 0.0

    # 7. AUC Trapezoid & End Amplitude ([0.0s, 3.5s])
    resp_mask = (t >= 0.0) & (t <= max_latency_s)
    t_resp = t[resp_mask]
    y_resp = y_sub[resp_mask]
    valid_resp = np.isfinite(y_resp)

    if np.sum(valid_resp) >= 2:
        t_clean = t_resp[valid_resp]
        y_clean = y_resp[valid_resp]
        auc_val = float(trapezoid(np.maximum(0.0, y_clean), t_clean))
        end_recovery_amp = float(y_clean[-1])
        resp_var = float(np.var(y_clean, ddof=1)) if len(y_clean) > 1 else 0.0
        resp_skew = float(stats.skew(y_clean, nan_policy="omit")) if len(y_clean) > 2 else 0.0
        resp_kurt = float(stats.kurtosis(y_clean, nan_policy="omit")) if len(y_clean) > 3 else 0.0
    else:
        auc_val = 0.0
        end_recovery_amp = 0.0
        resp_var = 0.0
        resp_skew = 0.0
        resp_kurt = 0.0

    # 8. Velocity & Acceleration Dynamics ([0.2s, 2.0s])
    vel_mask = (t >= 0.20) & (t <= 2.00)
    t_vel = t[vel_mask]
    y_vel = y_sub[vel_mask]
    if len(t_vel) >= 3 and np.sum(np.isfinite(y_vel)) >= 3:
        dt = float(np.median(np.diff(t_vel)))
        if dt > 0:
            dy = np.gradient(y_vel, dt)
            max_vel_idx = int(np.nanargmax(dy))
            max_dilation_vel = float(dy[max_vel_idx])
            t_max_vel = float(t_vel[max_vel_idx])
        else:
            max_dilation_vel = 0.0
            t_max_vel = 0.20
    else:
        max_dilation_vel = 0.0
        t_max_vel = 0.20

    # 9. Slopes
    rise_dt = max(t_peak - t_onset, 1e-3)
    response_slope = float(peak_pct / rise_dt)

    rebound_dt = max(max_latency_s - t_peak, 1e-3)
    rebound_slope = float((end_recovery_amp - peak_amp) / rebound_dt)

    # 10. Spectral & Frequency-Domain Features ([0.0s, 3.5s])
    # Frequency resolution note: For T=3.5s window, bin width Delta f = 1/T ~ 0.286 Hz
    time_regular = np.arange(0.0, max_latency_s + 1e-5, 1.0 / fs)
    if np.sum(valid_resp) >= 5:
        sig_interp = np.interp(time_regular, t_resp[valid_resp], y_resp[valid_resp])
        # Demean response before spectral decomposition
        sig_interp = sig_interp - np.mean(sig_interp)
        freqs, psd = periodogram(sig_interp, fs=fs, window="hamming", scaling="density")

        total_band = (freqs >= 0.10) & (freqs <= 4.0)
        tot_power = float(np.sum(psd[total_band])) if np.any(total_band) else 1e-6
        tot_power = max(tot_power, 1e-6)

        low_band = (freqs >= 0.10) & (freqs < 0.50)
        mid_band = (freqs >= 0.50) & (freqs < 1.50)
        high_band = (freqs >= 1.50) & (freqs <= 4.00)

        p_low = float(np.sum(psd[low_band])) / tot_power
        p_mid = float(np.sum(psd[mid_band])) / tot_power
        p_high = float(np.sum(psd[high_band])) / tot_power

        # Spectral Centroid (Hz)
        if np.any(total_band) and np.sum(psd[total_band]) > 0:
            spec_centroid = float(np.sum(freqs[total_band] * psd[total_band]) / np.sum(psd[total_band]))
        else:
            spec_centroid = 0.50
    else:
        p_low = 0.33
        p_mid = 0.33
        p_high = 0.34
        spec_centroid = 0.50

    feats = {
        "baseline_diameter_mean": b_mean,
        "baseline_diameter_std": b_std,
        "peak_dilation_amplitude": peak_amp,
        "peak_dilation_percent": peak_pct,
        "mean_response_amplitude": mean_response_amp,
        "auc_response_trapezoid": auc_val,
        "initial_constriction_depth": initial_constriction_depth,
        "end_recovery_amplitude": end_recovery_amp,
        "latency_to_peak_s": t_peak,
        "onset_latency_10pct_s": t_onset,
        "time_to_half_recovery_s": t_half_rec,
        "dilation_duration_s": dilation_duration,
        "latency_to_constriction_s": latency_to_constriction,
        "max_dilation_velocity": max_dilation_vel,
        "time_to_max_velocity_s": t_max_vel,
        "response_slope_onset_to_peak": response_slope,
        "response_variance": resp_var,
        "response_skewness": resp_skew,
        "response_kurtosis": resp_kurt,
        "half_rise_time_s": t_half_rise,
        "rebound_slope": rebound_slope,
        "spectral_power_low": p_low,
        "spectral_power_mid": p_mid,
        "spectral_power_high": p_high,
        "spectral_centroid": spec_centroid,
    }

    # Ensure all values are finite
    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0

    return feats


def extract_downsampled_timeseries(
    epoch: TrialEpoch,
    t_start: float = 0.0,
    t_end: float = 3.50,
    target_fs: float = 10.0
) -> Optional[np.ndarray]:
    """
    Extracts a regularly downsampled 10 Hz time-series vector over [0.0s, 3.5s] (36 features).
    Used as an ablation baseline comparing raw time-series representation against handcrafted features.
    """
    if not epoch.is_valid:
        return None

    t = epoch.time
    y_sub = epoch.pupil_subtractive
    mask = (t >= t_start - 0.1) & (t <= t_end + 0.1)
    t_ep = t[mask]
    y_ep = y_sub[mask]

    valid = np.isfinite(y_ep)
    if np.sum(valid) < 5:
        return None

    t_grid = np.arange(t_start, t_end + 1e-5, 1.0 / target_fs)
    interp_vals = np.interp(t_grid, t_ep[valid], y_ep[valid], left=np.nan, right=np.nan)

    if np.any(np.isnan(interp_vals)):
        finite_idx = np.where(np.isfinite(interp_vals))[0]
        if len(finite_idx) == 0:
            return None
        interp_vals[:finite_idx[0]] = interp_vals[finite_idx[0]]
        interp_vals[finite_idx[-1] + 1:] = interp_vals[finite_idx[-1]]

    return interp_vals.astype(np.float64)


def extract_feature_matrix_from_epochs(
    epochs: Sequence[TrialEpoch],
    subject_ids: Sequence[str],
    labels: Sequence[int],
    feature_names: Optional[Sequence[str]] = None
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts tabular feature DataFrame X, label array y, subject_id array, and trial_id array
    from a list of TrialEpoch objects.

    Parameters:
        epochs: Collection of preprocessed TrialEpoch objects.
        subject_ids: Corresponding subject identifier for each epoch.
        labels: Binary target label (0 or 1) for each epoch.
        feature_names: Optional subset of feature names to select. Defaults to FEATURE_NAMES_25.

    Returns:
        Tuple of (df_features, y, subjects, trial_ids)
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES_25

    rows = []
    y_list = []
    subjs_list = []
    trial_ids_list = []

    for ep, subj, y_lbl in zip(epochs, subject_ids, labels):
        feats = extract_features_from_epoch(ep)
        if feats is not None:
            filtered_feats = {k: feats[k] for k in feature_names if k in feats}
            rows.append(filtered_feats)
            y_list.append(int(y_lbl))
            subjs_list.append(str(subj))
            trial_ids_list.append(int(ep.trial_id))

    df_feats = pd.DataFrame(rows)
    # Ensure column order matches feature_names
    cols = [c for c in feature_names if c in df_feats.columns]
    df_feats = df_feats[cols]

    y_arr = np.array(y_list, dtype=np.int32)
    subjs_arr = np.array(subjs_list, dtype=object)
    trials_arr = np.array(trial_ids_list, dtype=np.int32)

    return df_feats, y_arr, subjs_arr, trials_arr

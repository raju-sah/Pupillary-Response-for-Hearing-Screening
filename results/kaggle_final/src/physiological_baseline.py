"""
Physiological Baseline Characterization and AEPR Feature Extraction Module.

Computes standard Auditory-Evoked Pupillary Response (AEPR) metrics:
1. Baseline diameter (P_base) in [-500ms, 0ms].
2. Peak dilation amplitude (Delta P_peak in physical units and %Delta P_peak).
3. Latency to peak (t_peak in seconds).
4. Dilation onset latency (t_onset: time to 10% peak dilation).
5. Time to half-recovery (t_half_rec: time post-peak to 50% amplitude decay).
6. Area Under Curve (AUC: integral of response curve over [0, 3.5s]).
7. Mean response amplitude.

Includes subject-level aggregation, pseudo-epoching for resting baselines,
paired statistical hypothesis tests, effect sizes, and Holm-Bonferroni correction.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy import stats

from src.preprocessing import PreprocessingConfig, TrialEpoch


@dataclass
class AEPRMetrics:
    """Standard physiological metrics for a single AEPR trial epoch."""
    trial_id: int
    subject_id: str
    stimulus: str
    condition: str
    baseline_diameter: float
    peak_amplitude: float  # max Delta P in physical units (mm or px)
    peak_percentage: float  # % Delta P relative to baseline
    latency_to_peak_s: float  # time of maximum dilation in seconds
    onset_latency_s: float  # time to 10% peak dilation in seconds
    half_recovery_s: float  # time post-peak where response drops to 50% peak amplitude
    auc_response: float  # integral of positive response over [0, 3.5s]
    mean_amplitude: float  # mean Delta P over [0, 3.5s]
    is_valid: bool


def compute_aepr_metrics_from_epoch(
    epoch: TrialEpoch,
    subject_id: str,
    min_latency_s: float = 0.20,
    max_latency_s: float = 3.50
) -> Optional[AEPRMetrics]:
    """
    Extracts standard physiological AEPR metrics from a single baseline-corrected TrialEpoch.
    """
    if not epoch.is_valid:
        return None

    t = epoch.time
    y_sub = epoch.pupil_subtractive
    y_div = epoch.pupil_divisive
    b_val = epoch.baseline_val

    # Post-stimulus evaluation window [0.20s, 3.50s]
    post_mask = (t >= min_latency_s) & (t <= max_latency_s)
    if np.sum(post_mask) < 5 or np.sum(np.isfinite(y_sub[post_mask])) < 5:
        return None

    t_post = t[post_mask]
    y_post_sub = y_sub[post_mask]
    y_post_div = y_div[post_mask]

    # 1. Peak Amplitude & Latency
    valid_idx = np.where(np.isfinite(y_post_sub))[0]
    if len(valid_idx) == 0:
        return None

    peak_sub = float(np.nanmax(y_post_sub))
    peak_idx_local = int(np.nanargmax(y_post_sub))
    t_peak = float(t_post[peak_idx_local])

    # Corresponding divisive peak
    peak_div = float(y_post_div[peak_idx_local]) if np.isfinite(y_post_div[peak_idx_local]) else (peak_sub / max(b_val, 1e-3) * 100.0)

    # 2. Dilation Onset Latency (time to 10% of peak dilation)
    onset_thresh = 0.10 * peak_sub if peak_sub > 0 else 0.0
    pre_peak_mask = (t_post <= t_peak) & (y_post_sub >= onset_thresh)
    if np.any(pre_peak_mask):
        t_onset = float(t_post[pre_peak_mask][0])
    else:
        t_onset = min_latency_s

    # 3. Time to Half-Recovery (time post-peak where response decays to 50% peak)
    half_thresh = 0.50 * peak_sub if peak_sub > 0 else 0.0
    post_peak_mask = (t_post >= t_peak) & (y_post_sub <= half_thresh)
    if np.any(post_peak_mask):
        t_half_rec = float(t_post[post_peak_mask][0])
    else:
        t_half_rec = max_latency_s  # Capped at epoch window end if unrecovered

    # 4. Area Under Curve (AUC) over [0s, 3.5s]
    auc_mask = (t >= 0.0) & (t <= max_latency_s)
    t_auc = t[auc_mask]
    y_auc = y_sub[auc_mask]
    finite_auc = np.isfinite(y_auc)
    if np.sum(finite_auc) >= 2:
        auc_val = float(trapezoid(np.maximum(0.0, y_auc[finite_auc]), t_auc[finite_auc]))
        mean_amp = float(np.nanmean(y_auc[finite_auc]))
    else:
        auc_val = 0.0
        mean_amp = 0.0

    return AEPRMetrics(
        trial_id=epoch.trial_id,
        subject_id=subject_id,
        stimulus=epoch.stimulus,
        condition=epoch.condition,
        baseline_diameter=b_val,
        peak_amplitude=peak_sub,
        peak_percentage=peak_div,
        latency_to_peak_s=t_peak,
        onset_latency_s=t_onset,
        half_recovery_s=t_half_rec,
        auc_response=auc_val,
        mean_amplitude=mean_amp,
        is_valid=True
    )


def extract_resting_pseudo_epochs(
    df_processed: pd.DataFrame,
    config: PreprocessingConfig,
    pseudo_interval_s: float = 4.0
) -> List[TrialEpoch]:
    """
    Extracts non-overlapping 4-second pseudo-epochs from a resting baseline recording
    to construct empirical control trials for statistical comparison.
    """
    grid_ts = df_processed["timestamp"].values
    t_min, t_max = grid_ts[0], grid_ts[-1]
    target_fs = config.target_sampling_rate_hz
    n_grid = len(grid_ts)

    # Combine left and right eyes
    sig_pair = np.stack([df_processed["pupil_left"].values, df_processed["pupil_right"].values], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        combined_pupil = np.nanmean(sig_pair, axis=0)

    pseudo_onsets = np.arange(t_min + 1.0, t_max - 4.0, pseudo_interval_s)
    pseudo_epochs = []

    for t_id, t_onset in enumerate(pseudo_onsets, 1):
        ep_start = t_onset + config.epoch_window_sec[0]
        ep_end = t_onset + config.epoch_window_sec[1]

        i_start = max(0, int(round((ep_start - t_min) * target_fs)))
        i_end = min(n_grid, int(round((ep_end - t_min) * target_fs)) + 1)

        if (i_end - i_start) < 10:
            continue

        ep_pupil = combined_pupil[i_start:i_end]
        ep_time = grid_ts[i_start:i_end] - t_onset

        miss_ratio = float(np.mean(np.isnan(ep_pupil)))
        base_mask = (ep_time >= config.baseline_window_sec[0]) & (ep_time <= config.baseline_window_sec[1])
        valid_base = ep_pupil[base_mask & np.isfinite(ep_pupil)]

        if len(valid_base) < (0.5 * np.sum(base_mask)) or miss_ratio > config.max_trial_missing_ratio:
            continue

        b_val = float(np.median(valid_base))
        sub = ep_pupil - b_val
        div = (sub / max(b_val, 10.0)) * 100.0

        pseudo_epochs.append(TrialEpoch(
            trial_id=t_id,
            stimulus="resting_control",
            condition="resting_baseline",
            time=ep_time,
            pupil_raw=ep_pupil,
            pupil_subtractive=sub,
            pupil_divisive=div,
            baseline_val=b_val,
            missing_ratio=miss_ratio,
            is_valid=True
        ))

    return pseudo_epochs


def apply_holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Applies Holm-Bonferroni step-down family-wise error rate correction.
    
    Returns:
        adjusted_p_values in the original order.
    """
    n = len(p_values)
    if n == 0:
        return []
    
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    p_adj = [0.0] * n
    running_max = 0.0
    
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(1.0, p * (n - rank))
        running_max = max(running_max, adj_p)
        p_adj[orig_idx] = running_max
        
    return p_adj


def compute_paired_statistics(
    condition_a_values: np.ndarray,
    condition_b_values: np.ndarray,
    name_a: str = "Condition A",
    name_b: str = "Condition B"
) -> Dict[str, Any]:
    """
    Runs comprehensive paired statistical tests (paired t-test, Wilcoxon signed-rank,
    Shapiro-Wilk normality test, Cohen's d_z, Hedge's g, and Rank-Biserial r).
    """
    a = np.array(condition_a_values, dtype=float)
    b = np.array(condition_b_values, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    n = len(a)

    if n < 5:
        return {"n": n, "error": "insufficient_paired_samples"}

    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    sem_diff = std_diff / np.sqrt(n) if n > 0 else 0.0

    # 1. Normality of differences (Shapiro-Wilk)
    if n >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(diff)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan

    # 2. Paired Student's t-test
    t_stat, t_p = stats.ttest_rel(a, b)

    # 3. Wilcoxon signed-rank test
    try:
        wilcox_res = stats.wilcoxon(a, b, alternative="two-sided")
        wilcox_stat, wilcox_p = float(wilcox_res.statistic), float(wilcox_res.pvalue)
    except Exception:
        wilcox_stat, wilcox_p = np.nan, np.nan

    # 4. Effect size: Cohen's d_z for paired samples
    cohen_dz = (mean_diff / std_diff) if std_diff > 0 else 0.0
    # Hedge's g correction
    hedges_g = cohen_dz * (1 - (3 / (4 * n - 1))) if n > 1 else cohen_dz

    # 5. Rank-Biserial correlation for Wilcoxon
    if not np.isnan(wilcox_stat):
        total_rank_sum = n * (n + 1) / 2.0
        rank_biserial = 1.0 - (2.0 * wilcox_stat / total_rank_sum) if total_rank_sum > 0 else 0.0
    else:
        rank_biserial = np.nan

    # 6. 95% Confidence Interval for mean difference
    ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean_diff, scale=sem_diff) if n > 1 else (mean_diff, mean_diff)

    return {
        "n_subjects": n,
        "name_a": name_a,
        "name_b": name_b,
        "mean_a": float(np.mean(a)),
        "std_a": float(np.std(a, ddof=1)),
        "mean_b": float(np.mean(b)),
        "std_b": float(np.std(b, ddof=1)),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "sem_diff": sem_diff,
        "ci_95": (float(ci_low), float(ci_high)),
        "shapiro_p": float(shapiro_p),
        "is_normal": bool(shapiro_p >= 0.05),
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "wilcox_stat": float(wilcox_stat),
        "wilcox_p": float(wilcox_p),
        "cohen_dz": float(cohen_dz),
        "hedges_g": float(hedges_g),
        "rank_biserial_r": float(rank_biserial)
    }

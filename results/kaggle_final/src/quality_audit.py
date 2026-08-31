"""
Quality Audit and Preprocessing Execution Engine for Dataset A and Dataset B.

Generates processed datasets in data/processed/ and comprehensive DATA_QUALITY_REPORT.md.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from src.preprocessing import (
    PreprocessingConfig,
    preprocess_pupil_series,
    resample_to_canonical_grid,
    baseline_correct_trial,
    TrialEpoch,
)


def process_dataset_a_recording(
    file_path: Path,
    config: PreprocessingConfig
) -> Tuple[pd.DataFrame, List[TrialEpoch], Dict[str, Any]]:
    """
    Processes a single Dataset A recording file (parquet):
    Preprocesses left and right eyes, resamples to canonical 50 Hz, extracts baseline-corrected epochs.
    """
    df = pd.read_parquet(file_path)
    subject_id = df["subject_id"].iloc[0]
    recording_id = df["recording_id"].iloc[0]
    condition = df["condition"].iloc[0]

    # Preprocess left eye on its native valid timestamps
    df_l = df[df["pupil_left"].notna()].sort_values("timestamp")
    if len(df_l) > 10:
        sig_l_raw = df_l["pupil_left"].values
        ts_l = df_l["timestamp"].values
        sig_l_clean, unrec_l, stats_l = preprocess_pupil_series(sig_l_raw, ts_l, unit="pixel", config=config)
    else:
        sig_l_clean, ts_l, unrec_l, stats_l = np.array([]), np.array([]), np.array([]), {"raw_missing_pct": 100.0, "native_fs_hz": 0.0, "final_unrecoverable_pct": 100.0}

    # Preprocess right eye on its native valid timestamps
    df_r = df[df["pupil_right"].notna()].sort_values("timestamp")
    if len(df_r) > 10:
        sig_r_raw = df_r["pupil_right"].values
        ts_r = df_r["timestamp"].values
        sig_r_clean, unrec_r, stats_r = preprocess_pupil_series(sig_r_raw, ts_r, unit="pixel", config=config)
    else:
        sig_r_clean, ts_r, unrec_r, stats_r = np.array([]), np.array([]), np.array([]), {"raw_missing_pct": 100.0, "native_fs_hz": 0.0, "final_unrecoverable_pct": 100.0}

    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()

    # Resample both eyes onto uniform 50.0 Hz canonical grid (preserving unrecoverable gap NaNs)
    resampled_l, grid_ts = resample_to_canonical_grid(
        sig_l_clean, ts_l, target_fs=config.target_sampling_rate_hz, max_gap_s=config.max_gap_duration_sec, t_start=t_min, t_end=t_max
    )
    resampled_r, _ = resample_to_canonical_grid(
        sig_r_clean, ts_r, target_fs=config.target_sampling_rate_hz, max_gap_s=config.max_gap_duration_sec, t_start=t_min, t_end=t_max
    )

    target_fs = config.target_sampling_rate_hz
    n_grid = len(grid_ts)

    # Combine left and right eye signals
    sig_pair = np.stack([resampled_l, resampled_r], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        combined_pupil = np.nanmean(sig_pair, axis=0)

    unique_trials = [t for t in df["trial_id"].unique() if t > 0]
    epochs = []

    if condition == "audio_stimulation" and len(unique_trials) > 0:
        for t_id in unique_trials:
            t_df = df[df["trial_id"] == t_id]
            t_onset = t_df[t_df["stimulus"] == "pure_tone"]["timestamp"].min() if not t_df[t_df["stimulus"] == "pure_tone"].empty else t_df["timestamp"].min()

            ep_start = t_onset + config.epoch_window_sec[0]
            ep_end = t_onset + config.epoch_window_sec[1]

            i_start = max(0, int(round((ep_start - t_min) * target_fs)))
            i_end = min(n_grid, int(round((ep_end - t_min) * target_fs)) + 1)

            if (i_end - i_start) < 10:
                continue

            ep_pupil = combined_pupil[i_start:i_end]
            ep_time = grid_ts[i_start:i_end] - t_onset

            miss_ratio = float(np.mean(np.isnan(ep_pupil)))

            sub, div, b_val, b_valid, b_warn = baseline_correct_trial(
                ep_pupil, ep_time, baseline_window=config.baseline_window_sec, unit="pixel", config=config
            )

            is_valid = b_valid and (miss_ratio <= config.max_trial_missing_ratio)
            rejection_reason = None
            if not b_valid:
                rejection_reason = b_warn
            elif miss_ratio > config.max_trial_missing_ratio:
                rejection_reason = f"excessive_missingness ({miss_ratio*100:.1f}% > {config.max_trial_missing_ratio*100:.0f}%)"

            epoch_obj = TrialEpoch(
                trial_id=int(t_id),
                stimulus="pure_tone",
                condition=condition,
                time=ep_time,
                pupil_raw=ep_pupil,
                pupil_subtractive=sub,
                pupil_divisive=div,
                baseline_val=b_val,
                missing_ratio=miss_ratio,
                is_valid=is_valid,
                rejection_reason=rejection_reason
            )
            epochs.append(epoch_obj)

    df_processed = pd.DataFrame({
        "subject_id": subject_id,
        "recording_id": recording_id,
        "timestamp": grid_ts,
        "pupil_left": resampled_l,
        "pupil_right": resampled_r,
        "condition": condition
    })

    audit_summary = {
        "dataset": "Dataset A (APURE)",
        "subject_id": subject_id,
        "recording_id": recording_id,
        "condition": condition,
        "native_fs_left_hz": stats_l.get("native_fs_hz", 0.0),
        "native_fs_right_hz": stats_r.get("native_fs_hz", 0.0),
        "raw_missing_left_pct": stats_l.get("raw_missing_pct", 100.0),
        "raw_missing_right_pct": stats_r.get("raw_missing_pct", 100.0),
        "post_interp_unrec_left_pct": stats_l.get("final_unrecoverable_pct", 100.0),
        "post_interp_unrec_right_pct": stats_r.get("final_unrecoverable_pct", 100.0),
        "total_trials": len(epochs),
        "retained_trials": sum(1 for e in epochs if e.is_valid),
        "rejected_trials": sum(1 for e in epochs if not e.is_valid),
        "epochs": epochs
    }

    return df_processed, epochs, audit_summary


def process_dataset_b_recording(
    file_path: Path,
    config: PreprocessingConfig
) -> Tuple[pd.DataFrame, List[TrialEpoch], Dict[str, Any]]:
    """
    Processes a single Dataset B (PsPM-AOB) recording file (parquet):
    Preprocesses left and right eyes, resamples to 50 Hz, extracts baseline-corrected oddball epochs.
    """
    df = pd.read_parquet(file_path)
    subject_id = df["subject_id"].iloc[0]
    recording_id = df["recording_id"].iloc[0]
    condition = df["condition"].iloc[0]

    ts = df["timestamp"].values
    dt = np.diff(ts)[np.diff(ts) > 0]
    native_fs = (1.0 / np.median(dt)) if len(dt) > 0 else 500.0

    # Preprocess left eye
    sig_l_raw = df["pupil_left"].values
    sig_l_clean, unrec_l, stats_l = preprocess_pupil_series(sig_l_raw, ts, unit="mm", config=config)

    # Preprocess right eye
    sig_r_raw = df["pupil_right"].values
    sig_r_clean, unrec_r, stats_r = preprocess_pupil_series(sig_r_raw, ts, unit="mm", config=config)

    # Resample to canonical 50.0 Hz (preserving unrecoverable gap NaNs)
    t_min, t_max = ts[0], ts[-1]
    target_fs = config.target_sampling_rate_hz
    resampled_l, grid_ts = resample_to_canonical_grid(
        sig_l_clean, ts, target_fs=target_fs, max_gap_s=config.max_gap_duration_sec, t_start=t_min, t_end=t_max
    )
    resampled_r, _ = resample_to_canonical_grid(
        sig_r_clean, ts, target_fs=target_fs, max_gap_s=config.max_gap_duration_sec, t_start=t_min, t_end=t_max
    )
    n_grid = len(grid_ts)

    # Combine left and right eyes
    sig_pair = np.stack([resampled_l, resampled_r], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        combined_pupil = np.nanmean(sig_pair, axis=0)

    # Find trial onset indices
    event_indices = df[df["stimulus"].isin(["standard_tone", "oddball_deviant"])].index.values

    epochs = []
    if len(event_indices) > 0:
        diff_idx = np.diff(event_indices, prepend=-999)
        onset_indices = event_indices[diff_idx > 1]

        for onset_idx in onset_indices:
            t_id = int(df["trial_id"].iloc[onset_idx])
            stim_type = df["stimulus"].iloc[onset_idx]
            t_onset = df["timestamp"].iloc[onset_idx]

            ep_start = t_onset + config.epoch_window_sec[0]
            ep_end = t_onset + config.epoch_window_sec[1]

            i_start = max(0, int(round((ep_start - t_min) * target_fs)))
            i_end = min(n_grid, int(round((ep_end - t_min) * target_fs)))

            if (i_end - i_start) < 10:
                continue

            ep_pupil = combined_pupil[i_start:i_end]
            ep_time = grid_ts[i_start:i_end] - t_onset

            miss_ratio = float(np.mean(np.isnan(ep_pupil)))

            # Baseline correction
            sub, div, b_val, b_valid, b_warn = baseline_correct_trial(
                ep_pupil, ep_time, baseline_window=config.baseline_window_sec, unit="mm", config=config
            )

            is_valid = b_valid and (miss_ratio <= config.max_trial_missing_ratio)
            rejection_reason = None
            if not b_valid:
                rejection_reason = b_warn
            elif miss_ratio > config.max_trial_missing_ratio:
                rejection_reason = f"excessive_missingness ({miss_ratio*100:.1f}% > {config.max_trial_missing_ratio*100:.0f}%)"

            epoch_obj = TrialEpoch(
                trial_id=t_id,
                stimulus=stim_type,
                condition=condition,
                time=ep_time,
                pupil_raw=ep_pupil,
                pupil_subtractive=sub,
                pupil_divisive=div,
                baseline_val=b_val,
                missing_ratio=miss_ratio,
                is_valid=is_valid,
                rejection_reason=rejection_reason
            )
            epochs.append(epoch_obj)

    df_processed = pd.DataFrame({
        "subject_id": subject_id,
        "recording_id": recording_id,
        "timestamp": grid_ts,
        "pupil_left": resampled_l,
        "pupil_right": resampled_r,
        "condition": condition
    })

    audit_summary = {
        "dataset": "Dataset B (PsPM-AOB)",
        "subject_id": subject_id,
        "recording_id": recording_id,
        "condition": condition,
        "native_fs_left_hz": float(native_fs),
        "native_fs_right_hz": float(native_fs),
        "raw_missing_left_pct": stats_l.get("raw_missing_pct", 0.0),
        "raw_missing_right_pct": stats_r.get("raw_missing_pct", 0.0),
        "post_interp_unrec_left_pct": stats_l.get("final_unrecoverable_pct", 0.0),
        "post_interp_unrec_right_pct": stats_r.get("final_unrecoverable_pct", 0.0),
        "total_trials": len(epochs),
        "retained_trials": sum(1 for e in epochs if e.is_valid),
        "rejected_trials": sum(1 for e in epochs if not e.is_valid),
        "epochs": epochs
    }

    return df_processed, epochs, audit_summary

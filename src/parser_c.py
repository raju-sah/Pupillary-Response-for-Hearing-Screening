"""
Parser and Preprocessor for Dataset C (OpenNeuro ds003690).
Official Title: EEG, ECG and pupil data from young and older adults: rest and auditory cued reaction time tasks
DOI: 10.18112/openneuro.ds003690.v1.0.0
Synchronized Modalities:
- Pupillometry: R-Dia-X-(mm), R-Dia-Y-(mm) at 500 Hz
- Cardiac: EKG at 500 Hz
- Central Auditory EEG: Cz, Fz, Pz, FCz at 500 Hz
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample_poly, butter, filtfilt


@dataclass
class MultimodalEpoch:
    subject_id: str
    group: str  # 'Young' or 'Older'
    trial_idx: int
    event_type: str  # 'cue', 'go', 'resting_control'
    label: int  # 1 for auditory stimulus present, 0 for resting control
    pupil: np.ndarray  # (L,) normalized pupil diameter (%ΔP)
    pupil_raw: np.ndarray  # (L,) raw pupil diameter (mm)
    ecg: np.ndarray  # (L,) standardized ECG lead
    eeg: np.ndarray  # (4, L) standardized central EEG channels [Cz, Fz, Pz, FCz]
    time_vector: np.ndarray  # (L,) in seconds relative to onset


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=-1)


def parse_dataset_c_subject(
    subject_dir: Path,
    subject_id: str,
    group: str = "Young",
    task: str = "task-simpleRT",
    run: str = "run-1",
    epoch_window: Tuple[float, float] = (-0.5, 2.5),
    target_fs: int = 50,
) -> List[MultimodalEpoch]:
    """
    Parses a single subject's continuous EEG/ECG/Pupil recording from Dataset C.
    Extracts time-locked epochs for auditory tones (cue, go) and resting baseline intervals.
    """
    eeg_dir = subject_dir / "eeg"
    set_file = eeg_dir / f"{subject_id}_{task}_{run}_eeg.set"
    events_file = eeg_dir / f"{subject_id}_{task}_{run}_events.tsv"
    channels_file = eeg_dir / f"{subject_id}_{task}_{run}_channels.tsv"

    if not set_file.exists() or not events_file.exists():
        raise FileNotFoundError(f"Missing files for {subject_id}: {set_file}")

    # 1. Load MATLAB EEGLAB struct
    mat = scipy.io.loadmat(str(set_file), simplify_cells=True)
    raw_data = mat["data"]  # (nbchan, pnts)
    srate = float(mat["srate"])  # 500 Hz
    chanlocs = mat["chanlocs"]
    labels = [c["labels"] for c in chanlocs]

    # 2. Identify channel indices
    def find_idx(name: str) -> Optional[int]:
        for idx, lbl in enumerate(labels):
            if lbl.lower() == name.lower():
                return idx
        return None

    cz_idx = find_idx("Cz")
    fz_idx = find_idx("Fz")
    pz_idx = find_idx("Pz")
    fcz_idx = find_idx("FCz") or find_idx("FC1") or cz_idx

    eeg_indices = [
        cz_idx if cz_idx is not None else 0,
        fz_idx if fz_idx is not None else 1,
        pz_idx if pz_idx is not None else 2,
        fcz_idx if fcz_idx is not None else 3,
    ]

    ekg_idx = find_idx("EKG") or find_idx("ECG") or -3
    dia_x_idx = find_idx("R-Dia-X-(mm)") or find_idx("Dia-X") or -2
    dia_y_idx = find_idx("R-Dia-Y-(mm)") or find_idx("Dia-Y") or -1

    # Extract continuous signals
    pupil_x = raw_data[dia_x_idx].astype(np.float64)
    pupil_y = raw_data[dia_y_idx].astype(np.float64)
    # Combine X and Y diameter, handling any zero values
    pupil_continuous = np.where((pupil_x > 0) & (pupil_y > 0), 0.5 * (pupil_x + pupil_y), np.maximum(pupil_x, pupil_y))
    # Replace zeros / missing with linear interpolation
    zero_mask = pupil_continuous <= 0.5
    if np.any(zero_mask) and not np.all(zero_mask):
        valid_idx = np.where(~zero_mask)[0]
        pupil_continuous[zero_mask] = np.interp(np.where(zero_mask)[0], valid_idx, pupil_continuous[valid_idx])

    # Pre-filter continuous pupil with 0.05-4 Hz bandpass
    try:
        pupil_filtered = butter_bandpass_filter(pupil_continuous, 0.05, 4.0, srate)
    except Exception:
        pupil_filtered = pupil_continuous - np.median(pupil_continuous)

    # Continuous ECG
    ecg_continuous = raw_data[ekg_idx].astype(np.float64)
    # Bandpass filter ECG (0.5 - 40 Hz)
    try:
        ecg_filtered = butter_bandpass_filter(ecg_continuous, 0.5, 40.0, srate)
    except Exception:
        ecg_filtered = ecg_continuous

    # Continuous EEG (4 channels)
    eeg_continuous = raw_data[eeg_indices].astype(np.float64)
    # Bandpass filter EEG (0.5 - 30 Hz)
    try:
        eeg_filtered = butter_bandpass_filter(eeg_continuous, 0.5, 30.0, srate)
    except Exception:
        eeg_filtered = eeg_continuous

    # 3. Read events
    events_df = pd.read_csv(events_file, sep="\t")
    n_samples_total = raw_data.shape[1]

    # Desired epoch length at target_fs
    t_start, t_end = epoch_window
    n_target_samples = int(round((t_end - t_start) * target_fs))
    time_vec = np.linspace(t_start, t_end, n_target_samples)

    downsample_factor = int(round(srate / target_fs))  # 500 / 50 = 10

    epochs: List[MultimodalEpoch] = []
    trial_idx = 0

    # Extract auditory stimulus epochs (cue, go)
    stim_events = events_df[events_df["trial_type"].isin(["cue", "go"])].copy()

    for _, row in stim_events.iterrows():
        onset_sec = float(row["onset"])
        event_type = str(row["trial_type"])

        start_samp = int(round((onset_sec + t_start) * srate))
        end_samp = int(round((onset_sec + t_end) * srate))

        if start_samp < 0 or end_samp >= n_samples_total:
            continue

        # Extract segments
        p_raw_seg = pupil_continuous[start_samp:end_samp]
        p_filt_seg = pupil_filtered[start_samp:end_samp]
        ecg_seg = ecg_filtered[start_samp:end_samp]
        eeg_seg = eeg_filtered[:, start_samp:end_samp]

        # Baseline normalization for pupil
        base_samples = int(abs(t_start) * srate)
        base_val = np.median(p_raw_seg[:base_samples]) if base_samples > 0 else np.median(p_raw_seg)
        if base_val <= 0.1:
            base_val = 1.0

        p_norm_seg = ((p_raw_seg - base_val) / base_val) * 100.0

        # Downsample to target_fs (50 Hz)
        p_norm_ds = resample_poly(p_norm_seg, 1, downsample_factor)[:n_target_samples]
        p_raw_ds = resample_poly(p_raw_seg, 1, downsample_factor)[:n_target_samples]
        ecg_ds = resample_poly(ecg_seg, 1, downsample_factor)[:n_target_samples]
        eeg_ds = np.zeros((4, n_target_samples), dtype=np.float32)
        for ch in range(4):
            eeg_ds[ch] = resample_poly(eeg_seg[ch], 1, downsample_factor)[:n_target_samples]

        # Standardize ECG and EEG per epoch (zero mean, unit variance)
        ecg_std = np.std(ecg_ds)
        ecg_ds = (ecg_ds - np.mean(ecg_ds)) / (ecg_std if ecg_std > 1e-6 else 1.0)

        for ch in range(4):
            ch_std = np.std(eeg_ds[ch])
            eeg_ds[ch] = (eeg_ds[ch] - np.mean(eeg_ds[ch])) / (ch_std if ch_std > 1e-6 else 1.0)

        epochs.append(
            MultimodalEpoch(
                subject_id=subject_id,
                group=group,
                trial_idx=trial_idx,
                event_type=event_type,
                label=1,  # Stimulus Present
                pupil=p_norm_ds.astype(np.float32),
                pupil_raw=p_raw_ds.astype(np.float32),
                ecg=ecg_ds.astype(np.float32),
                eeg=eeg_ds.astype(np.float32),
                time_vector=time_vec.astype(np.float32),
            )
        )
        trial_idx += 1

    # Extract resting baseline control pseudo-epochs (from pre-trial or long inter-trial gaps)
    # Between trials, take quiescent segments
    for i in range(len(stim_events) - 1):
        curr_onset = float(stim_events.iloc[i]["onset"])
        next_onset = float(stim_events.iloc[i + 1]["onset"])
        gap = next_onset - (curr_onset + t_end)

        if gap >= (t_end - t_start) + 1.0:  # Gap is wide enough for a clean resting control epoch
            control_onset = curr_onset + t_end + 0.5
            start_samp = int(round((control_onset + t_start) * srate))
            end_samp = int(round((control_onset + t_end) * srate))

            if start_samp < 0 or end_samp >= n_samples_total:
                continue

            p_raw_seg = pupil_continuous[start_samp:end_samp]
            ecg_seg = ecg_filtered[start_samp:end_samp]
            eeg_seg = eeg_filtered[:, start_samp:end_samp]

            base_samples = int(abs(t_start) * srate)
            base_val = np.median(p_raw_seg[:base_samples]) if base_samples > 0 else np.median(p_raw_seg)
            if base_val <= 0.1:
                base_val = 1.0
            p_norm_seg = ((p_raw_seg - base_val) / base_val) * 100.0

            p_norm_ds = resample_poly(p_norm_seg, 1, downsample_factor)[:n_target_samples]
            p_raw_ds = resample_poly(p_raw_seg, 1, downsample_factor)[:n_target_samples]
            ecg_ds = resample_poly(ecg_seg, 1, downsample_factor)[:n_target_samples]
            eeg_ds = np.zeros((4, n_target_samples), dtype=np.float32)
            for ch in range(4):
                eeg_ds[ch] = resample_poly(eeg_seg[ch], 1, downsample_factor)[:n_target_samples]

            ecg_std = np.std(ecg_ds)
            ecg_ds = (ecg_ds - np.mean(ecg_ds)) / (ecg_std if ecg_std > 1e-6 else 1.0)
            for ch in range(4):
                ch_std = np.std(eeg_ds[ch])
                eeg_ds[ch] = (eeg_ds[ch] - np.mean(eeg_ds[ch])) / (ch_std if ch_std > 1e-6 else 1.0)

            epochs.append(
                MultimodalEpoch(
                    subject_id=subject_id,
                    group=group,
                    trial_idx=trial_idx,
                    event_type="resting_control",
                    label=0,  # Resting Baseline Control
                    pupil=p_norm_ds.astype(np.float32),
                    pupil_raw=p_raw_ds.astype(np.float32),
                    ecg=ecg_ds.astype(np.float32),
                    eeg=eeg_ds.astype(np.float32),
                    time_vector=time_vec.astype(np.float32),
                )
            )
            trial_idx += 1

    return epochs

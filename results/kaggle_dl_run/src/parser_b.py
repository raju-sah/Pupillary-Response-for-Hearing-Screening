"""
Parser for Dataset B: PsPM-AOB (Auditory Oddball Pupillometry).
DOI: 10.5281/zenodo.3608706
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import scipy.io as sio
from src.schema import validate_pupil_dataframe, REQUIRED_COLUMNS


def parse_pspm_mat_file(pupil_mat_path: Path, cogent_mat_path: Path = None) -> pd.DataFrame:
    """
    Parses a single PsPM-AOB pupil recording MAT file into a standardized DataFrame.
    """
    mat_data = sio.loadmat(str(pupil_mat_path), squeeze_me=True, struct_as_record=False)
    filename = pupil_mat_path.stem
    parts = filename.split("_")
    subj_num = parts[2] if len(parts) >= 3 else "unknown"
    subject_id = f"sub-{subj_num}"
    recording_id = filename

    d = mat_data.get("data")
    if d is None or len(d) < 2:
        raise ValueError(f"Unexpected data format in {pupil_mat_path}")

    # Channel 0: pupil_l, Channel 1: pupil_r
    pupil_left_raw = d[0].data
    pupil_right_raw = d[1].data
    sr = getattr(d[0].header, "sr", 500.0)

    n_samples = len(pupil_left_raw)
    timestamps = np.arange(n_samples) / float(sr)

    # Event markers (Channel 2)
    marker_times = []
    marker_values = []
    if len(d) > 2 and hasattr(d[2], "data") and hasattr(d[2], "markerinfo"):
        marker_times = np.atleast_1d(d[2].data)
        marker_values = np.atleast_1d(d[2].markerinfo.value)

    # Assign trial_id and stimulus
    stimulus_col = np.array(["none"] * n_samples, dtype=object)
    trial_id_col = np.zeros(n_samples, dtype=int)

    # Convert marker timestamps (seconds) to sample indices
    if len(marker_times) > 0 and len(marker_values) > 0:
        for trial_idx, (t_sec, val) in enumerate(zip(marker_times, marker_values), start=1):
            s_idx = int(round(t_sec * sr))
            if 0 <= s_idx < n_samples:
                # Mark stimulus window (e.g. 50ms tone duration = 25 samples at 500 Hz)
                duration_samples = max(1, int(round(0.05 * sr)))
                end_idx = min(n_samples, s_idx + duration_samples)
                stim_label = "standard_tone" if val == 1 else ("oddball_deviant" if val == 2 else f"marker_{val}")
                stimulus_col[s_idx:end_idx] = stim_label

            # Assign trial id from this stimulus until next stimulus (or end)
            next_s_idx = int(round(marker_times[trial_idx] * sr)) if trial_idx < len(marker_times) else n_samples
            trial_id_col[s_idx:next_s_idx] = trial_idx

    df = pd.DataFrame({
        "subject_id": subject_id,
        "recording_id": recording_id,
        "trial_id": trial_id_col,
        "timestamp": timestamps,
        "pupil_left": pupil_left_raw,
        "pupil_right": pupil_right_raw,
        "stimulus": stimulus_col,
        "condition": "auditory_oddball",
    })

    return df


def parse_all_dataset_b(raw_extracted_dir: Path, output_dir: Path) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Parses all PsPM-AOB recordings in raw_extracted_dir and saves intermediate Parquet/CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pupil_files = sorted(list(raw_extracted_dir.glob("**/*pupil*.mat")))
    
    parsed_files = []
    failed_files = []

    for p_file in pupil_files:
        cogent_name = p_file.name.replace("pupil", "cogent").split("_sn")[0] + ".mat"
        cogent_file = p_file.parent / cogent_name
        if not cogent_file.exists():
            cogent_file = None

        try:
            df = parse_pspm_mat_file(p_file, cogent_file)
            val = validate_pupil_dataframe(df)
            if not val.is_valid:
                failed_files.append(f"{p_file.name}: {val.errors}")
                continue

            out_file = output_dir / f"{p_file.stem}.parquet"
            df.to_parquet(out_file, index=False)
            parsed_files.append(str(out_file))
        except Exception as e:
            failed_files.append(f"{p_file.name}: {str(e)}")

    summary = {
        "num_files_found": len(pupil_files),
        "num_files_parsed": len(parsed_files),
        "num_files_failed": len(failed_files),
        "failed_details": failed_files,
    }

    return parsed_files, failed_files, summary


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw" / "dataset_b_pspm_aob" / "extracted" / "Data"
    out_dir = base_dir / "data" / "intermediate" / "dataset_b"
    parsed, failed, summ = parse_all_dataset_b(raw_dir, out_dir)
    print(f"Dataset B parsing complete: {len(parsed)} files parsed, {len(failed)} failed.")

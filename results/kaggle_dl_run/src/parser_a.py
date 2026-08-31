"""
Parser for Dataset A: APURE - Pupil Response After Audio Stimulation.
Zenodo DOI: 10.5281/zenodo.10497437
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from src.schema import validate_pupil_dataframe, REQUIRED_COLUMNS


def standardize_excel_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes English and Italian column headers from APURE Excel sheets."""
    new_cols = {}
    for c in df.columns:
        cl = str(c).lower()
        if "millisecond" in cl or "tempo" in cl or "ms" in cl:
            new_cols[c] = "milliseconds"
        elif "diameter" in cl or "diametro" in cl:
            new_cols[c] = "diameter"
        elif "audio" in cl or "stimol" in cl:
            new_cols[c] = "audio"
        elif "blink" in cl:
            new_cols[c] = "blink"
        elif "artifact" in cl or "artefatto" in cl:
            new_cols[c] = "artifact"
        elif "confidence" in cl:
            new_cols[c] = "confidence"
    return df.rename(columns=new_cols)


def parse_subject_dataset_a(subject_dir: Path, audio_stimuli_df: Optional[pd.DataFrame] = None) -> List[pd.DataFrame]:
    """
    Parses dx (right eye) and sx (left eye) Excel files for a subject across all sheets (audio, baseline).
    Synchronizes left and right eye streams on the physical millisecond timeline.
    """
    subj_code = subject_dir.name
    subject_id = f"sub-{subj_code}"

    # Stimulus metadata
    stim_freq = 2000
    stim_db = 70
    if audio_stimuli_df is not None and "Subject ID" in audio_stimuli_df.columns:
        match = audio_stimuli_df[audio_stimuli_df["Subject ID"] == subj_code]
        if not match.empty:
            stim_freq = int(match["Stimuli Frequency (Hz)"].iloc[0])
            stim_db = int(match["Stimuli Level (dB)"].iloc[0])

    dx_file = subject_dir / f"{subj_code}_dx.xlsx"
    sx_file = subject_dir / f"{subj_code}_sx.xlsx"

    if not dx_file.exists() and not sx_file.exists():
        raise FileNotFoundError(f"No dx/sx Excel files found in {subject_dir}")

    sheets = []
    if dx_file.exists():
        xl_dx = pd.ExcelFile(dx_file)
        sheets = xl_dx.sheet_names
    elif sx_file.exists():
        xl_sx = pd.ExcelFile(sx_file)
        sheets = xl_sx.sheet_names

    parsed_dfs = []

    for sheet in sheets:
        df_dx = pd.read_excel(dx_file, sheet_name=sheet) if dx_file.exists() else None
        df_sx = pd.read_excel(sx_file, sheet_name=sheet) if sx_file.exists() else None

        if df_dx is not None:
            df_dx = standardize_excel_cols(df_dx)
            df_dx.columns = [f"{c}_dx" if c != "milliseconds" else c for c in df_dx.columns]

        if df_sx is not None:
            df_sx = standardize_excel_cols(df_sx)
            df_sx.columns = [f"{c}_sx" if c != "milliseconds" else c for c in df_sx.columns]

        if df_dx is not None and df_sx is not None:
            merged = pd.merge(
                df_dx, df_sx, on="milliseconds", how="outer"
            ).sort_values("milliseconds").reset_index(drop=True)
        elif df_dx is not None:
            merged = df_dx.copy()
        else:
            merged = df_sx.copy()

        n_samples = len(merged)
        timestamps = merged["milliseconds"].astype(float).values / 1000.0

        # Extract pupil diameters
        pupil_right = merged["diameter_dx"].astype(float).values if "diameter_dx" in merged.columns else np.full(n_samples, np.nan)
        pupil_left = merged["diameter_sx"].astype(float).values if "diameter_sx" in merged.columns else np.full(n_samples, np.nan)

        # Audio triggers (combine triggers from both channels)
        aud_r = merged["audio_dx"].fillna(0).values if "audio_dx" in merged.columns else np.zeros(n_samples)
        aud_l = merged["audio_sx"].fillna(0).values if "audio_sx" in merged.columns else np.zeros(n_samples)
        audio_trigger = np.maximum(aud_r, aud_l).astype(int)

        # Build stimulus labels and trial IDs
        stimulus_labels = []
        trial_ids = np.zeros(n_samples, dtype=int)
        current_trial = 0
        in_sound = False

        for i, aud in enumerate(audio_trigger):
            if aud == 1:
                if not in_sound:
                    current_trial += 1
                    in_sound = True
                stimulus_labels.append("pure_tone")
            else:
                in_sound = False
                stimulus_labels.append("none")
            trial_ids[i] = current_trial

        recording_id = f"{subj_code}_{sheet}"
        condition_name = "audio_stimulation" if sheet == "audio" else "resting_baseline"

        df_out = pd.DataFrame({
            "subject_id": subject_id,
            "recording_id": recording_id,
            "trial_id": trial_ids,
            "timestamp": timestamps,
            "pupil_left": pupil_left,
            "pupil_right": pupil_right,
            "stimulus": stimulus_labels,
            "condition": condition_name,
        })

        parsed_dfs.append(df_out)

    return parsed_dfs


def parse_all_dataset_a(raw_extracted_dir: Path, output_dir: Path) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Parses all subject folders in extracted Dataset A and saves intermediate Parquet files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stim_file = raw_extracted_dir / "audio_stimuli.xlsx"
    stim_df = pd.read_excel(stim_file) if stim_file.exists() else None

    subject_dirs = [
        d for d in sorted(list(raw_extracted_dir.iterdir()))
        if d.is_dir() and ((d / f"{d.name}_dx.xlsx").exists() or (d / f"{d.name}_sx.xlsx").exists())
    ]

    parsed_files = []
    failed_files = []

    print(f"Found {len(subject_dirs)} subject directories in Dataset A...")
    for s_dir in subject_dirs:
        try:
            print(f"Parsing subject {s_dir.name}...")
            dfs = parse_subject_dataset_a(s_dir, stim_df)
            for df in dfs:
                val = validate_pupil_dataframe(df)
                if not val.is_valid:
                    failed_files.append(f"{df['recording_id'].iloc[0]}: {val.errors}")
                    continue

                rec_id = df["recording_id"].iloc[0]
                out_file = output_dir / f"{rec_id}.parquet"
                df.to_parquet(out_file, index=False)
                parsed_files.append(str(out_file))
        except Exception as e:
            failed_files.append(f"{s_dir.name}: {str(e)}")

    summary = {
        "num_subjects_found": len(subject_dirs),
        "num_recordings_parsed": len(parsed_files),
        "num_recordings_failed": len(failed_files),
        "failed_details": failed_files,
    }

    return parsed_files, failed_files, summary


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw" / "dataset_a_zenodo" / "extracted"
    out_dir = base_dir / "data" / "intermediate" / "dataset_a"
    parsed, failed, summ = parse_all_dataset_a(raw_dir, out_dir)
    print(f"\nDataset A parsing complete: {len(parsed)} recordings parsed, {len(failed)} failed.")
    if failed:
        print(f"Failed details: {failed}")

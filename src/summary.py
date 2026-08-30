"""
Summary generator for dataset parsing and data quality analysis.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def compute_dataset_summary(intermediate_dir: Path, dataset_name: str) -> Dict[str, Any]:
    """
    Computes rigorous data summary statistics across all parsed files in intermediate_dir.
    """
    parquet_files = sorted(list(intermediate_dir.glob("*.parquet")))
    if not parquet_files:
        return {
            "dataset_name": dataset_name,
            "status": "No parsed files found",
            "num_files": 0,
        }

    dfs = [pd.read_parquet(f) for f in parquet_files]
    full_df = pd.concat(dfs, ignore_index=True)

    # 1. Subjects & Recordings & Trials
    num_subjects = int(full_df["subject_id"].nunique())
    subjects_list = sorted(list(full_df["subject_id"].unique()))
    num_recordings = int(full_df["recording_id"].nunique())
    num_trials = int(full_df["trial_id"].nunique())
    total_samples = int(len(full_df))

    # 2. Sampling rate calculation per recording
    sampling_rates = []
    for rec_id, group in full_df.groupby("recording_id"):
        t = group["timestamp"].values
        if len(t) > 1:
            diffs = np.diff(t)
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                median_dt = np.median(diffs)
                sr = 1.0 / median_dt
                sampling_rates.append(sr)

    sr_summary = {
        "median_hz": float(np.median(sampling_rates)) if sampling_rates else np.nan,
        "min_hz": float(np.min(sampling_rates)) if sampling_rates else np.nan,
        "max_hz": float(np.max(sampling_rates)) if sampling_rates else np.nan,
    }

    # 3. Missingness
    missingness = {
        col: float(full_df[col].isna().mean() * 100.0)
        for col in full_df.columns
    }

    # 4. Available labels
    labels_stimulus = {str(k): int(v) for k, v in full_df["stimulus"].value_counts().items()}
    labels_condition = {str(k): int(v) for k, v in full_df["condition"].value_counts().items()}

    summary = {
        "dataset_name": dataset_name,
        "total_samples": total_samples,
        "num_subjects": num_subjects,
        "subject_ids": subjects_list,
        "num_recordings": num_recordings,
        "num_trials": num_trials,
        "sampling_rate": sr_summary,
        "available_columns": list(full_df.columns),
        "missingness_percentage": missingness,
        "stimulus_distribution": labels_stimulus,
        "condition_distribution": labels_condition,
        "num_files_parsed": len(parquet_files),
    }

    return summary


def format_summary_markdown(summary: Dict[str, Any]) -> str:
    """Formats the summary dict into a clean Markdown table / report."""
    md = []
    md.append(f"### Data Summary: {summary.get('dataset_name', 'Unknown')}\n")
    md.append(f"- **Total Samples:** {summary.get('total_samples', 0):,}")
    md.append(f"- **Number of Subjects:** {summary.get('num_subjects', 0)}")
    md.append(f"- **Number of Recordings:** {summary.get('num_recordings', 0)}")
    md.append(f"- **Number of Unique Trials:** {summary.get('num_trials', 0)}")
    
    sr = summary.get("sampling_rate", {})
    md.append(f"- **Sampling Rate (Median):** {sr.get('median_hz', 'N/A'):.1f} Hz (Range: {sr.get('min_hz', 'N/A'):.1f} - {sr.get('max_hz', 'N/A'):.1f} Hz)")
    md.append(f"- **Available Columns:** `{', '.join(summary.get('available_columns', []))}`\n")
    
    md.append("#### Missing Value Analysis:")
    md.append("| Column | Missing Percentage |")
    md.append("| :--- | :--- |")
    for col, miss in summary.get("missingness_percentage", {}).items():
        md.append(f"| `{col}` | {miss:.2f}% |")
    md.append("")

    md.append("#### Stimulus / Event Distribution:")
    md.append("| Event / Stimulus | Sample Count |")
    md.append("| :--- | :--- |")
    for stim, cnt in summary.get("stimulus_distribution", {}).items():
        md.append(f"| `{stim}` | {cnt:,} |")
    md.append("")

    md.append("#### Condition Distribution:")
    md.append("| Condition | Sample Count |")
    md.append("| :--- | :--- |")
    for cond, cnt in summary.get("condition_distribution", {}).items():
        md.append(f"| `{cond}` | {cnt:,} |")
    md.append("")

    return "\n".join(md)

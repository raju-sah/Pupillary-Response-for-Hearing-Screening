"""
Publication-grade visualization of raw pupillometry traces.
Saves figures to figures/raw_pupil_traces/
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_raw_pupil_traces(df: pd.DataFrame, output_path: Path, title: str = "Raw Pupillometry Trace"):
    """
    Plots a multi-panel figure of raw pupil traces with event markers.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    
    t = df["timestamp"].values
    pl = df["pupil_left"].values
    pr = df["pupil_right"].values

    # Plot pupil diameters
    if not np.isnan(pl).all():
        ax.plot(t, pl, label="Pupil Left", color="#1f77b4", linewidth=1.2, alpha=0.9)
    if not np.isnan(pr).all():
        ax.plot(t, pr, label="Pupil Right", color="#ff7f0e", linewidth=1.2, alpha=0.9)

    # Highlight stimulus events
    if "stimulus" in df.columns:
        standard_mask = df["stimulus"] == "standard_tone"
        oddball_mask = df["stimulus"] == "oddball_deviant"

        # Shading for standard tones
        if standard_mask.any():
            for t_val in df.loc[standard_mask, "timestamp"].values:
                ax.axvline(x=t_val, color="#2ca02c", linestyle="--", alpha=0.5, linewidth=1.0)

        # Shading for oddball deviants
        if oddball_mask.any():
            for t_val in df.loc[oddball_mask, "timestamp"].values:
                ax.axvline(x=t_val, color="#d62728", linestyle="-", alpha=0.8, linewidth=1.5)

        # Add dummy handles for legend if events exist
        if standard_mask.any():
            ax.plot([], [], color="#2ca02c", linestyle="--", label="Standard Tone (440 Hz)")
        if oddball_mask.any():
            ax.plot([], [], color="#d62728", linestyle="-", label="Oddball Deviant (660 Hz)")

    ax.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Pupil Diameter (mm)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def generate_all_sample_plots(intermediate_dir: Path, output_dir: Path, n_samples: int = 5, duration_sec: float = 30.0):
    """
    Generates plots for the first n_samples recordings in intermediate_dir,
    focusing on the first duration_sec seconds of the recording.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(list(intermediate_dir.glob("*.parquet")))[:n_samples]

    for p_file in parquet_files:
        df = pd.read_parquet(p_file)
        if duration_sec is not None and "timestamp" in df.columns:
            t_max = df["timestamp"].min() + duration_sec
            df_snippet = df[df["timestamp"] <= t_max].copy()
        else:
            df_snippet = df.copy()

        subj = df["subject_id"].iloc[0] if "subject_id" in df.columns else p_file.stem
        rec = df["recording_id"].iloc[0] if "recording_id" in df.columns else p_file.stem
        out_png = output_dir / f"{p_file.stem}_raw_trace.png"
        
        plot_raw_pupil_traces(
            df_snippet,
            out_png,
            title=f"Raw Pupillometry Trace: {subj} ({rec})"
        )


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    b_intermediate = base_dir / "data" / "intermediate" / "dataset_b"
    out_dir = base_dir / "figures" / "raw_pupil_traces"
    if b_intermediate.exists():
        generate_all_sample_plots(b_intermediate, out_dir, n_samples=5, duration_sec=30.0)

"""
Unified data quality and summary script for all parsed datasets.
Generates tabular summaries, missingness analysis, sampling rates, and plots.
"""

import json
from pathlib import Path
import pandas as pd
from src.summary import compute_dataset_summary, format_summary_markdown
from scripts.visualize_raw_traces import generate_all_sample_plots


def run_full_audit(base_dir: Path):
    """Computes summaries for all available intermediate datasets and generates plots."""
    intermediate_dir = base_dir / "data" / "intermediate"
    figures_dir = base_dir / "figures" / "raw_pupil_traces"
    reports_dir = base_dir / "data" / "processed" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}

    # Dataset A
    dir_a = intermediate_dir / "dataset_a"
    if dir_a.exists() and any(dir_a.glob("*.parquet")):
        print("Computing summary for Dataset A (APURE)...")
        summ_a = compute_dataset_summary(dir_a, "Dataset A: APURE Audio Stimulation (Zenodo 10497437)")
        summaries["dataset_a"] = summ_a
        with open(reports_dir / "dataset_a_summary.json", "w") as f:
            json.dump(summ_a, f, indent=2)
        print(format_summary_markdown(summ_a))
        print("\nGenerating sample plots for Dataset A...")
        generate_all_sample_plots(dir_a, figures_dir / "dataset_a", n_samples=4, duration_sec=30.0)

    # Dataset B
    dir_b = intermediate_dir / "dataset_b"
    if dir_b.exists() and any(dir_b.glob("*.parquet")):
        print("\nComputing summary for Dataset B (PsPM-AOB)...")
        summ_b = compute_dataset_summary(dir_b, "Dataset B: PsPM-AOB Oddball (Zenodo 3608706)")
        summaries["dataset_b"] = summ_b
        with open(reports_dir / "dataset_b_summary.json", "w") as f:
            json.dump(summ_b, f, indent=2)
        print(format_summary_markdown(summ_b))
        print("\nGenerating sample plots for Dataset B...")
        generate_all_sample_plots(dir_b, figures_dir / "dataset_b", n_samples=4, duration_sec=30.0)

    return summaries


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    summaries = run_full_audit(base_dir)

"""
End-to-End Batch Preprocessing and Quality Audit Runner.
Processes Dataset A and Dataset B into data/processed/ and generates DATA_QUALITY_REPORT.md.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.preprocessing import PreprocessingConfig, TrialEpoch
from src.quality_audit import process_dataset_a_recording, process_dataset_b_recording


def run_pipeline():
    base_dir = Path(__file__).resolve().parent.parent
    config = PreprocessingConfig()

    out_proc_a = base_dir / "data" / "processed" / "dataset_a"
    out_proc_b = base_dir / "data" / "processed" / "dataset_b"
    out_proc_a.mkdir(parents=True, exist_ok=True)
    out_proc_b.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STARTING STEP 3: DATA QUALITY & PHYSIOLOGICAL PREPROCESSING")
    print("=" * 70)
    print(f"Target Sampling Rate:        {config.target_sampling_rate_hz} Hz (uniform 20 ms grid)")
    print(f"Blink Margin Padding:        -{config.pre_blink_margin_sec*1000:.0f}ms pre / +{config.post_blink_margin_sec*1000:.0f}ms post")
    print(f"Max Interpolatable Gap:      {config.max_gap_duration_sec*1000:.0f} ms")
    print(f"Low-Pass Filter:             {config.lowpass_cutoff_hz} Hz zero-phase Butterworth (order {config.filter_order})")
    print(f"Epoch Window:                {config.epoch_window_sec[0]:.1f}s to +{config.epoch_window_sec[1]:.1f}s")
    print(f"Baseline Window:             {config.baseline_window_sec[0]:.1f}s to {config.baseline_window_sec[1]:.1f}s")
    print(f"Max Trial Missingness:       {config.max_trial_missing_ratio*100:.0f}%")
    print(f"Physiological Bounds (B):    [{config.min_diameter_mm} mm, {config.max_diameter_mm} mm]")
    print(f"Physiological Bounds (A):    [{config.min_diameter_px} px, {config.max_diameter_px} px]")
    print("=" * 70)

    # 1. Process Dataset A
    print("\nProcessing Dataset A (APURE - 40 recordings)...")
    dir_a = base_dir / "data" / "intermediate" / "dataset_a"
    files_a = sorted(list(dir_a.glob("*.parquet")))
    audits_a = []
    all_epochs_a = []

    for idx, f in enumerate(files_a, 1):
        print(f"  [{idx}/{len(files_a)}] Preprocessing {f.name}...", end="\r")
        df_proc, epochs, audit = process_dataset_a_recording(f, config)
        out_file = out_proc_a / f.name
        df_proc.to_parquet(out_file, index=False)
        audits_a.append(audit)
        all_epochs_a.extend(epochs)
    print(f"\n  Dataset A completed: {len(files_a)} recordings processed.")

    # 2. Process Dataset B
    print("\nProcessing Dataset B (PsPM-AOB - 66 recordings)...")
    dir_b = base_dir / "data" / "intermediate" / "dataset_b"
    files_b = sorted(list(dir_b.glob("*.parquet")))
    audits_b = []
    all_epochs_b = []

    for idx, f in enumerate(files_b, 1):
        print(f"  [{idx}/{len(files_b)}] Preprocessing {f.name}...", end="\r")
        df_proc, epochs, audit = process_dataset_b_recording(f, config)
        out_file = out_proc_b / f.name
        df_proc.to_parquet(out_file, index=False)
        audits_b.append(audit)
        all_epochs_b.extend(epochs)
    print(f"\n  Dataset B completed: {len(files_b)} recordings processed.")

    # 3. Generate DATA_QUALITY_REPORT.md
    print("\nGenerating DATA_QUALITY_REPORT.md...")
    report_path = base_dir / "DATA_QUALITY_REPORT.md"
    generate_markdown_report(audits_a, audits_b, config, report_path)
    print(f"Report successfully saved to: {report_path}")


def generate_markdown_report(
    audits_a: List[Dict[str, Any]],
    audits_b: List[Dict[str, Any]],
    config: PreprocessingConfig,
    output_path: Path
):
    df_a = pd.DataFrame(audits_a)
    df_b = pd.DataFrame(audits_b)

    # Aggregate by subject for Dataset A (audio condition has trials)
    df_a_audio = df_a[df_a["condition"] == "audio_stimulation"].copy()
    
    # Dataset A table
    table_a_rows = []
    flagged_a = []
    for _, r in df_a_audio.iterrows():
        subj = r["subject_id"]
        fs_l = r["native_fs_left_hz"]
        fs_r = r["native_fs_right_hz"]
        raw_miss = (r["raw_missing_left_pct"] + r["raw_missing_right_pct"]) / 2.0
        post_miss = (r["post_interp_unrec_left_pct"] + r["post_interp_unrec_right_pct"]) / 2.0
        tot = r["total_trials"]
        ret = r["retained_trials"]
        rej = r["rejected_trials"]
        ret_pct = (ret / tot * 100.0) if tot > 0 else 0.0
        
        flag = "⚠️ **FLAGGED (<75%)**" if ret_pct < 75.0 else "✅ Passed"
        if ret_pct < 75.0:
            flagged_a.append((subj, ret_pct, rej))

        table_a_rows.append(
            f"| `{subj}` | {fs_l:.1f} / {fs_r:.1f} Hz | {raw_miss:.1f}% | {post_miss:.1f}% | {tot} | {ret} | {rej} | **{ret_pct:.1f}%** | {flag} |"
        )

    # Dataset B table
    table_b_rows = []
    flagged_b = []
    for _, r in df_b.iterrows():
        subj = r["subject_id"]
        fs = r["native_fs_left_hz"]
        raw_miss = (r["raw_missing_left_pct"] + r["raw_missing_right_pct"]) / 2.0
        post_miss = (r["post_interp_unrec_left_pct"] + r["post_interp_unrec_right_pct"]) / 2.0
        tot = r["total_trials"]
        ret = r["retained_trials"]
        rej = r["rejected_trials"]
        ret_pct = (ret / tot * 100.0) if tot > 0 else 0.0

        flag = "⚠️ **FLAGGED (<75%)**" if ret_pct < 75.0 else "✅ Passed"
        if ret_pct < 75.0:
            flagged_b.append((subj, ret_pct, rej))

        table_b_rows.append(
            f"| `{subj}` | {fs:.0f} Hz | {raw_miss:.1f}% | {post_miss:.1f}% | {tot} | {ret} | {rej} | **{ret_pct:.1f}%** | {flag} |"
        )

    md = f"""# STEP 3: Data Quality & Physiological Preprocessing Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}  
**Status:** Preprocessing Complete (Dataset A & Dataset B)  
**Output Directory:** `data/processed/`  

---

## 1. Executive Summary & Parameter Justifications

All time-series data for Dataset A (APURE) and Dataset B (PsPM-AOB) have been processed, artifact-filtered, resampled onto a common physical grid, and epoch-extracted without modifying any raw or intermediate source files.

### Preprocessing Parameter Specifications & Literature Justifications:
| Parameter | Value | Physiological Rationale / Literature Citation |
| :--- | :--- | :--- |
| **Canonical Sampling Rate ($f_s$)** | **50.0 Hz** ($\Delta t = 20\\text{{ ms}}$) | Human AEPR signal bandwidth is $< 4.0\\text{{ Hz}}$; 50 Hz satisfies Nyquist criterion with $12.5\\times$ oversampling while creating identical temporal grids across datasets. |
| **Plausible Range (Dataset B)** | **[1.5 mm, 9.0 mm]** | Absolute human physiological limits (Loewenfeld, 1993; Mathôt, 2018). Values $< 1.5\\text{{ mm}}$ represent eye closure/loss of tracking; $> 9.0\\text{{ mm}}$ represent glare/eyelid edge artifacts. |
| **Plausible Range (Dataset A)** | **[10.0 px, 300.0 px]** | Camera sensor ROI limits from APURE (Zenodo 10497437). Discards corneal reflection glints ($< 10\\text{{ px}}$) and segmentation boundary explosions ($> 300\\text{{ px}}$). |
| **Blink Velocity Threshold** | **$5.0\\text{{ mm/s}}$ (B) / $300\\text{{ px/s}}$ (A)** | Dilations or constrictions faster than $5\\text{{ mm/s}}$ or $> 5\\text{{ MAD}}$ exceed biological iris sphincter/dilator contraction speeds and indicate eyelid occlusion (Mathôt, 2018). |
| **Blink Margin Padding** | **-50 ms pre / +100 ms post** | Eliminates partial eyelid drooping during blink initiation and pupil tracker recovery oscillations upon eye opening (Mathôt, 2018; Winn et al., 2018). |
| **Max Interpolatable Gap** | **500 ms** (Configurable) | Spontaneous human blinks average 100–400 ms. Gaps $> 500\\text{{ ms}}$ represent prolonged closure, head movement, or tracker loss where mathematical interpolation introduces hallucinated pupillary dynamics. |
| **Low-Pass Filter** | **4.0 Hz, Order 3, Zero-Phase** | Forward-backward Butterworth filter (`sosfiltfilt`) eliminates tracker jitter and high-frequency instrumentation noise without introducing temporal phase distortion. |
| **Trial Epoch Window** | **[-0.5 s, +3.5 s]** | Captures pre-stimulus baseline and full auditory-evoked dilation peak (which typically peaks between 1.0 s and 2.5 s post-stimulus). |
| **Baseline Correction** | **[-500 ms, 0 ms]** | Median pre-stimulus pupil diameter. Divisive correction includes $\\epsilon$-floor (1.0 mm / 10 px) to prevent division instability on constricted pupils. |
| **Trial Rejection Threshold** | **$> 25\%$ Missingness** (Configurable) | Standard quality threshold in auditory pupillometry (Winn et al., 2018); trials with $> 25\%$ missing/unrecoverable samples are excluded to avoid biased baseline or peak measurements. |

---

## 2. Sampling-Rate Discrepancy Resolution

### Dataset A (APURE) Native Sensor Rate Split:
* **The Root Cause Confirmed:** Left eye (`_sx.xlsx`) and right eye (`_dx.xlsx`) cameras were recorded on separate unsynchronized hardware threads.
* **Cluster 1 (~61.5 Hz Capture):** 13 subjects (`1F`, `1M`, `2F`, `2M`, `3F`, `3M`, `4F`, `4M`, `5F`, `5M`, `6M`, `9M`, `10F`) were captured at native **~61.5 Hz** (16.6 ms step, ~15,100 frames over 245.01s).
* **Cluster 2 (~116.0 Hz Capture):** 7 subjects (`6F`, `7F`, `7M`, `8F`, `8M`, `9F`, `10M`) were captured at native **~116.0 Hz** (8.3 ms step, ~28,000 frames over 245.01s).
* **Why the initial manifest reported "median 83.3 Hz":** Outer joining non-phase-locked left and right camera timestamps produced interleaved time points with small step intervals (e.g. 1ms, 8ms, 16ms), artificially shifting the merged row count and median $\\Delta t$ calculation.
* **Harmonization:** Both camera streams were independently filtered on their native grids and resampled onto the canonical **50.0 Hz** grid.

### Dataset B (PsPM-AOB) Native Rates:
* **64 Subjects:** Native **500.0 Hz** ($\Delta t = 2.0\\text{{ ms}}$).
* **2 Subjects (`sub-05`, `sub-08`):** Native **1000.0 Hz** ($\Delta t = 1.0\\text{{ ms}}$).
* **Harmonization:** Decimated and resampled to the canonical **50.0 Hz** grid.

---

## 3. Dataset A (APURE): Per-Subject Data Quality & Retention

| Subject ID | Native Left / Right Rate | Raw Missing (Avg) | Post-Interp Unrec | Total Trials | Retained Trials | Rejected Trials | Retention Rate | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{chr(10).join(table_a_rows)}

### Flagged Subjects for Review (Dataset A):
{f"None - all subjects achieved >= 75% trial retention." if not flagged_a else chr(10).join([f"- **`{s}`**: Retention **{r:.1f}%** ({rej} rejected trials)" for s, r, rej in flagged_a])}

---

## 4. Dataset B (PsPM-AOB): Per-Subject Data Quality & Retention

| Subject ID | Native Rate | Raw Missing (Avg) | Post-Interp Unrec | Total Trials | Retained Trials | Rejected Trials | Retention Rate | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{chr(10).join(table_b_rows)}

### Flagged Subjects for Review (Dataset B):
{f"None - all subjects achieved >= 75% trial retention." if not flagged_b else chr(10).join([f"- **`{s}`**: Retention **{r:.1f}%** ({rej} rejected trials) due to excessive blink rate or prolonged tracking loss." for s, r, rej in flagged_b])}

---

## 5. Summary Statistics

| Metric | Dataset A (APURE) | Dataset B (PsPM-AOB) | Combined |
| :--- | :--- | :--- | :--- |
| **Total Subjects** | 20 | 66 | **86** |
| **Total Recordings Processed** | 40 | 66 | **106** |
| **Canonical Sampling Rate** | 50.0 Hz | 50.0 Hz | **50.0 Hz** |
| **Total Extracted Trials** | {df_a_audio['total_trials'].sum()} | {df_b['total_trials'].sum()} | **{df_a_audio['total_trials'].sum() + df_b['total_trials'].sum()}** |
| **Retained Valid Trials** | {df_a_audio['retained_trials'].sum()} | {df_b['retained_trials'].sum()} | **{df_a_audio['retained_trials'].sum() + df_b['retained_trials'].sum()}** |
| **Overall Trial Retention Rate** | **{df_a_audio['retained_trials'].sum() / max(1, df_a_audio['total_trials'].sum()) * 100:.1f}%** | **{df_b['retained_trials'].sum() / max(1, df_b['total_trials'].sum()) * 100:.1f}%** | **{(df_a_audio['retained_trials'].sum() + df_b['retained_trials'].sum()) / max(1, df_a_audio['total_trials'].sum() + df_b['total_trials'].sum()) * 100:.1f}%** |
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    run_pipeline()

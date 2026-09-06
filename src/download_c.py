"""
Downloader and verifier for Dataset C (OpenNeuro ds003690).
Official Title: EEG, ECG and pupil data from young and older adults: rest and auditory cued reaction time tasks
DOI: 10.18112/openneuro.ds003690.v1.0.0
License: CC0 Public Domain Dedication
"""

import os
from pathlib import Path
from typing import List, Optional
import requests
from tqdm import tqdm
from src.download import download_file_parallel

OPENNEURO_S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds003690"
DEFAULT_TARGET_SUBJECTS = [
    "sub-AB4",
    "sub-AB9",
    "sub-AB10",
    "sub-AB7",
    "sub-AB8",
    "sub-AB11",
]


def download_file_small(url: str, dest_path: Path, timeout: int = 30) -> bool:
    """Downloads a small file (tsv, json) directly."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)
    return True


def download_dataset_c_cohort(
    data_dir: Optional[Path] = None,
    subjects: Optional[List[str]] = None,
    task: str = "task-simpleRT",
    run: str = "run-1",
    num_threads: int = 16,
) -> Path:
    """
    Downloads metadata and task runs for selected Dataset C cohort.
    Uses parallel chunk downloading for .set files.
    """
    if data_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        data_dir = base_dir / "data" / "raw" / "dataset_c_openneuro"
    else:
        data_dir = Path(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    subjects = subjects or DEFAULT_TARGET_SUBJECTS

    print(f"=== Syncing Dataset C (ds003690) Cohort: {len(subjects)} subjects ===")

    # 1. Download global metadata
    meta_files = ["dataset_description.json", "participants.tsv"]
    for mf in meta_files:
        url = f"{OPENNEURO_S3_BASE}/{mf}"
        dest = data_dir / mf
        download_file_small(url, dest)

    # 2. Download subject files
    for sub in subjects:
        sub_dir = data_dir / sub / "eeg"
        sub_dir.mkdir(parents=True, exist_ok=True)

        for ft in ["channels.tsv", "events.tsv", "eeg.json"]:
            filename = f"{sub}_{task}_{run}_{ft}"
            url = f"{OPENNEURO_S3_BASE}/{sub}/eeg/{filename}"
            dest = sub_dir / filename
            download_file_small(url, dest)

        # Large .set file via parallel multi-threading
        set_filename = f"{sub}_{task}_{run}_eeg.set"
        set_url = f"{OPENNEURO_S3_BASE}/{sub}/eeg/{set_filename}"
        set_dest = sub_dir / set_filename
        if not set_dest.exists() or set_dest.stat().st_size == 0:
            download_file_parallel(set_url, set_dest, num_threads=num_threads)

    print(f"Dataset C cohort download complete at: {data_dir}")
    return data_dir


if __name__ == "__main__":
    download_dataset_c_cohort()

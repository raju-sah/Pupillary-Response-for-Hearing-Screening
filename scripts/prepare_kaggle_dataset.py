"""
Prepares and packages intermediate pupillometry data into the Kaggle dataset staging directory.
"""

import shutil
import json
from pathlib import Path

def prepare_dataset(base_dir: Path):
    staging_dir = base_dir / "kaggle" / "dataset"
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy metadata
    metadata = {
        "title": "Auditory-Evoked Pupillary Responses (AEPR)",
        "id": "rajucode/aepr-pupillometry-dataset",
        "licenses": [
            {
                "name": "CC-BY-4.0"
            }
        ],
        "description": "Standardized and parsed pupillometry time-series datasets for Auditory-Evoked Pupillary Responses (AEPR) research.\n\n### Sources & Attributions:\n1. Dataset A: APURE - Pupil Response After Audio Stimulation (Zenodo DOI: 10.5281/zenodo.10497437) under CC-BY-4.0.\n2. Dataset B: PsPM-AOB - Eye tracker measurements from auditory oddball tasks (Zenodo DOI: 10.5281/zenodo.3608706) by Korn & Bach (2016) under CC-BY-4.0.\n\nAll recordings are standardized to canonical schemas preserving subject_id, recording_id, trial_id, timestamp, pupil_left, pupil_right, stimulus, and condition."
    }

    with open(staging_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 2. Copy intermediate data
    inter_dir = base_dir / "data" / "intermediate"
    for ds_folder in ["dataset_a", "dataset_b"]:
        src_path = inter_dir / ds_folder
        dst_path = staging_dir / ds_folder
        if src_path.exists():
            if dst_path.exists():
                shutil.rmtree(dst_path)
            print(f"Copying {src_path} -> {dst_path}...")
            shutil.copytree(src_path, dst_path)

    # 3. Copy manifests and schemas
    manifest_src = base_dir / "DATASET_MANIFEST.md"
    if manifest_src.exists():
        shutil.copy(manifest_src, staging_dir / "DATASET_MANIFEST.md")

    print(f"Kaggle dataset staging ready at: {staging_dir}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    prepare_dataset(base_dir)

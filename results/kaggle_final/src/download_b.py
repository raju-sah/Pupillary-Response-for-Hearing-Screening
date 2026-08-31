"""
Downloader and verifier for Dataset B (PsPM-AOB).
Zenodo DOI: 10.5281/zenodo.3608706
"""

import os
import json
import zipfile
from pathlib import Path
import requests
from src.download import download_file_parallel, calculate_checksums

ZENODO_RECORD_ID = "3608706"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def download_and_verify_dataset_b(raw_dir: Path) -> dict:
    """Downloads Dataset B (PsPM-AOB) and verifies checksums."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching metadata from Zenodo record {ZENODO_RECORD_ID} (PsPM-AOB)...")
    res = requests.get(ZENODO_API_URL, timeout=30)
    res.raise_for_status()
    record = res.json()

    metadata_path = raw_dir / "zenodo_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    files_info = record.get("files", [])
    downloaded_files = []

    for f_info in files_info:
        file_key = f_info.get("key")
        download_url = f_info.get("links", {}).get("self")
        expected_size = f_info.get("size")
        expected_checksum = f_info.get("checksum", "")

        dest_file = raw_dir / file_key
        print(f"\nDownloading {file_key}...")
        download_file_parallel(download_url, dest_file, expected_size=expected_size, num_threads=8)

        print(f"Verifying checksum for {file_key}...")
        computed = calculate_checksums(dest_file)
        expected_md5 = expected_checksum.replace("md5:", "") if "md5:" in expected_checksum else None
        if expected_md5 and computed["md5"] != expected_md5:
            raise ValueError(f"MD5 mismatch for {file_key}! Expected {expected_md5}, got {computed['md5']}")
        print(f"Checksum verified for {file_key}: MD5={computed['md5']}")

        downloaded_files.append({
            "filename": file_key,
            "path": str(dest_file),
            "size": dest_file.stat().st_size,
            "md5": computed["md5"],
            "sha256": computed["sha256"],
        })

        extract_dir = raw_dir / "extracted"
        if file_key.endswith(".zip"):
            if not extract_dir.exists() or not any(extract_dir.iterdir()):
                print(f"Extracting {file_key} into {extract_dir}...")
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(dest_file, "r") as zf:
                    zf.extractall(extract_dir)
                print("Extraction complete!")

    return {
        "record_id": ZENODO_RECORD_ID,
        "title": record.get("metadata", {}).get("title"),
        "license": record.get("metadata", {}).get("license", {}).get("id"),
        "doi": record.get("doi"),
        "files": downloaded_files,
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw" / "dataset_b_pspm_aob"
    summary = download_and_verify_dataset_b(raw_dir)
    print("\nDataset B Summary:")
    print(json.dumps(summary, indent=2))

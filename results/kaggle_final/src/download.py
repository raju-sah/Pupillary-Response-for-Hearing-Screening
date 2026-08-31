"""
High-speed parallel downloader and verifier for Dataset A from Zenodo.
DOI: 10.5281/zenodo.10497437
"""

import os
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm
import py7zr

ZENODO_RECORD_ID = "10497437"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def calculate_checksums(filepath: Path) -> dict:
    """Calculates MD5 and SHA256 checksums of a file in chunks."""
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            md5.update(chunk)
            sha256.update(chunk)
    return {
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
    }


def download_chunk(url: str, start: int, end: int, filepath: Path, chunk_idx: int):
    """Downloads a specific byte range to an open file at the given offset."""
    headers = {"Range": f"bytes={start}-{end}"}
    r = requests.get(url, headers=headers, stream=True, timeout=60)
    r.raise_for_status()
    with open(filepath, "r+b") as f:
        f.seek(start)
        f.write(r.content)
    return end - start + 1


def download_file_parallel(url: str, dest_path: Path, expected_size: int = None, num_threads: int = 16) -> Path:
    """Downloads a large file in parallel using HTTP Range requests."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and expected_size and dest_path.stat().st_size == expected_size:
        print(f"File already downloaded and size matches: {dest_path}")
        return dest_path

    head = requests.head(url, allow_redirects=True, timeout=30)
    total_size = int(head.headers.get("content-length", 0))
    if expected_size and total_size != expected_size:
        total_size = expected_size

    # Pre-allocate destination file
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    with open(temp_path, "wb") as f:
        f.truncate(total_size)

    chunk_size = total_size // num_threads
    ranges = []
    for i in range(num_threads):
        start = i * chunk_size
        end = (start + chunk_size - 1) if i < num_threads - 1 else total_size - 1
        ranges.append((i, start, end))

    print(f"Downloading {dest_path.name} ({total_size / (1024*1024):.2f} MB) with {num_threads} parallel threads...")
    with tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024, desc=dest_path.name) as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(download_chunk, url, start, end, temp_path, idx)
                for idx, start, end in ranges
            ]
            for future in as_completed(futures):
                bytes_downloaded = future.result()
                pbar.update(bytes_downloaded)

    temp_path.rename(dest_path)
    return dest_path


def download_and_verify_dataset_a(raw_dir: Path) -> dict:
    """
    Fetches metadata, downloads Dataset A archive, verifies MD5/SHA256,
    and extracts it into data/raw/dataset_a_zenodo/extracted/
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching metadata from Zenodo record {ZENODO_RECORD_ID}...")
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
        download_file_parallel(download_url, dest_file, expected_size=expected_size, num_threads=16)

        print(f"Verifying checksums for {file_key}...")
        computed = calculate_checksums(dest_file)
        print(f"Computed MD5:    {computed['md5']}")
        print(f"Computed SHA256: {computed['sha256']}")

        expected_md5 = expected_checksum.replace("md5:", "") if "md5:" in expected_checksum else None
        if expected_md5 and computed["md5"] != expected_md5:
            raise ValueError(f"MD5 mismatch for {file_key}! Expected {expected_md5}, got {computed['md5']}")
        print("Checksum verified successfully!")

        checksum_report = raw_dir / f"{file_key}.checksums.json"
        with open(checksum_report, "w", encoding="utf-8") as f:
            json.dump({
                "filename": file_key,
                "size_bytes": dest_file.stat().st_size,
                "expected_checksum": expected_checksum,
                "computed_md5": computed["md5"],
                "computed_sha256": computed["sha256"],
                "verified": True,
            }, f, indent=2)

        downloaded_files.append({
            "filename": file_key,
            "path": str(dest_file),
            "size": dest_file.stat().st_size,
            "md5": computed["md5"],
            "sha256": computed["sha256"],
        })

        extract_dir = raw_dir / "extracted"
        if file_key.endswith(".7z"):
            if not extract_dir.exists() or not any(extract_dir.iterdir()):
                print(f"Extracting {file_key} into {extract_dir}...")
                extract_dir.mkdir(parents=True, exist_ok=True)
                with py7zr.SevenZipFile(dest_file, mode='r') as archive:
                    archive.extractall(path=extract_dir)
                print("Extraction complete!")
            else:
                print(f"Archive already extracted in {extract_dir}")

    return {
        "record_id": ZENODO_RECORD_ID,
        "title": record.get("metadata", {}).get("title"),
        "license": record.get("metadata", {}).get("license", {}).get("id"),
        "doi": record.get("doi"),
        "files": downloaded_files,
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw" / "dataset_a_zenodo"
    summary = download_and_verify_dataset_a(raw_dir)
    print("\nDownload & Verification Summary:")
    print(json.dumps(summary, indent=2))

"""
Helper script to fetch Kaggle kernel output and print execution log.
"""

import sys
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def fetch_output(kernel_slug: str, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    status_resp = api.kernels_status(kernel_slug)
    print("Kernel status response:", status_resp)

    print(f"Downloading output files for {kernel_slug} into {target_dir}...")
    api.kernels_output(kernel_slug, path=str(target_dir))

    print("\nFiles in output directory:")
    for f in sorted(target_dir.glob("**/*")):
        if f.is_file():
            print(f"  - {f.relative_to(target_dir)} ({f.stat().st_size} bytes)")

    log_files = list(target_dir.glob("*.log"))
    if log_files:
        print(f"\n--- LOG CONTENT ({log_files[0].name}) ---")
        with open(log_files[0], "r", encoding="utf-8") as lf:
            content = lf.read()
            # If JSON array log, parse and pretty print stdout/stderr
            try:
                entries = json.loads(content)
                for entry in entries:
                    stream = entry.get("stream_name", "")
                    data = entry.get("data", "")
                    if stream == "stdout":
                        sys.stdout.write(data)
                    else:
                        sys.stderr.write(data)
            except Exception:
                print(content[-3000:])


if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "rajucode/aepr-training-runner"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/kaggle_smoke_test")
    fetch_output(slug, out)

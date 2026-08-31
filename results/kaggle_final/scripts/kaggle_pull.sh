#!/usr/bin/env bash
set -e

# Change to repo root
cd "$(dirname "$0")/.."

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="results/kaggle_runs/$RUN_ID"
KERNEL_SLUG="rajucode/aepr-training-runner"

mkdir -p "$OUT_DIR"

echo "=================================================="
echo "Fetching Kaggle outputs for $KERNEL_SLUG..."
echo "Target directory: $OUT_DIR"
echo "=================================================="

kaggle kernels output "$KERNEL_SLUG" -p "$OUT_DIR"

echo ""
echo "Files downloaded to $OUT_DIR:"
ls -la "$OUT_DIR"

if [ -f "$OUT_DIR/metrics.json" ]; then
    echo ""
    echo "--- METRICS SUMMARY ---"
    cat "$OUT_DIR/metrics.json"
fi

if [ -f "$OUT_DIR/run_metadata.json" ]; then
    echo ""
    echo "--- RUN METADATA ---"
    cat "$OUT_DIR/run_metadata.json"
fi

echo ""
echo "=================================================="
echo "Outputs successfully fetched to: $OUT_DIR"
echo "=================================================="

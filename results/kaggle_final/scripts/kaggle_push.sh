#!/usr/bin/env bash
set -e

# Change to repo root
cd "$(dirname "$0")/.."

ACCELERATOR="${1:-nvidiaTeslaT4}"

echo "=================================================="
echo "Building Kaggle runner notebook with local src/..."
echo "=================================================="
python3 scripts/build_kaggle_notebook.py

echo ""
echo "Pushing notebook to Kaggle ($ACCELERATOR)..."
kaggle kernels push -p kaggle/ --accelerator "$ACCELERATOR"

echo ""
echo "=================================================="
echo "Kaggle Kernel Pushed Successfully!"
echo "URL: https://www.kaggle.com/code/rajucode/aepr-training-runner"
echo "Check status with: ./scripts/kaggle_status.sh"
echo "=================================================="

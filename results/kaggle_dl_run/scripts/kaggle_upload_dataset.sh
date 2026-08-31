#!/usr/bin/env bash
set -e

# Change to repo root
cd "$(dirname "$0")/.."

echo "=================================================="
echo "Preparing Kaggle Dataset staging directory..."
echo "=================================================="
PYTHONPATH=. .venv/bin/python scripts/prepare_kaggle_dataset.py

DATASET_DIR="kaggle/dataset"
DATASET_SLUG="rajucode/aepr-pupillometry-dataset"

echo ""
echo "Checking if dataset $DATASET_SLUG exists on Kaggle..."
if kaggle datasets status "$DATASET_SLUG" > /dev/null 2>&1; then
    echo "Dataset exists. Creating a new version..."
    kaggle datasets version -p "$DATASET_DIR" -m "Update parsed pupillometry dataset" -r zip
else
    echo "Dataset does not exist. Creating new Kaggle dataset..."
    kaggle datasets create -p "$DATASET_DIR" -r zip
fi

echo ""
echo "=================================================="
echo "Kaggle Dataset Upload Completed Successfully!"
echo "URL: https://www.kaggle.com/datasets/$DATASET_SLUG"
echo "=================================================="

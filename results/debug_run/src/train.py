"""
Universal training entrypoint and smoke-test runner.
Used identically for both local development and Kaggle GPU execution.
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


class DummyTemporalCNN(nn.Module):
    """Minimal 1D temporal CNN for smoke testing GPU execution."""
    def __init__(self, in_channels: int = 2, seq_len: int = 128, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        h = self.relu(self.conv1(x))
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash if in a git repo."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
        return commit
    except Exception:
        return os.environ.get("GIT_COMMIT_HASH", "unknown_commit")


def collect_environment_metadata(config: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Logs system, package, and GPU metadata for full reproducibility."""
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    cuda_version = torch.version.cuda if cuda_available else None

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit_hash": get_git_commit_hash(),
        "random_seed": seed,
        "config": config,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "cuda_available": cuda_available,
            "cuda_device_count": device_count,
            "cuda_device_name": device_name,
            "cuda_version": cuda_version,
        }
    }


def run_smoke_test(data_dir: Path, output_dir: Path, seed: int = 42) -> Dict[str, Any]:
    """
    Executes a 1-epoch dummy smoke-test model on a small sample of pupillometry data
    to verify that data loading, GPU tensors, forward/backward pass, and artifact saving work end-to-end.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("STARTING AEPR INFRASTRUCTURE SMOKE TEST")
    print("=" * 60)

    # 1. Hardware Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")
    if torch.cuda.is_available():
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("  WARNING: CUDA is NOT available. Running on CPU.")

    # 2. Data Loading Check
    print(f"\nChecking data directory: {data_dir}")
    intermediate_files = list(data_dir.glob("**/*.parquet"))
    print(f"Found {len(intermediate_files)} parquet files in {data_dir}")

    if intermediate_files:
        sample_file = intermediate_files[0]
        print(f"Loading sample file: {sample_file.name}")
        df = pd.read_parquet(sample_file)
        print(f"  Loaded shape: {df.shape}, Columns: {list(df.columns)}")
        # Extract small signal chunk for test
        pl = df["pupil_left"].fillna(0).values[:128]
        pr = df["pupil_right"].fillna(0).values[:128]
        if len(pl) < 128:
            pl = np.pad(pl, (0, 128 - len(pl)))
            pr = np.pad(pr, (0, 128 - len(pr)))
        dummy_input = torch.tensor(np.stack([pl, pr], axis=0), dtype=torch.float32).unsqueeze(0)
    else:
        print("  Notice: No parquet files found in data_dir. Using synthetic tensor for unit verification.")
        dummy_input = torch.randn(2, 2, 128, dtype=torch.float32)

    # 3. Model Forward and Backward Pass
    dummy_input = dummy_input.to(device)
    dummy_target = torch.tensor([1] * dummy_input.shape[0], dtype=torch.long).to(device)

    model = DummyTemporalCNN(in_channels=2, seq_len=128, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\nExecuting forward pass on device...")
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    print(f"  Forward loss: {loss.item():.4f}")

    print("Executing backward pass (gradient computation)...")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("  Backward pass completed successfully!")

    # 4. Save Outputs & Reproducibility Metadata
    config = {
        "mode": "smoke_test",
        "in_channels": 2,
        "seq_len": 128,
        "num_classes": 2,
        "batch_size": dummy_input.shape[0],
        "learning_rate": 1e-3,
    }
    metadata = collect_environment_metadata(config, seed)
    metrics = {
        "smoke_test_loss": float(loss.item()),
        "cuda_used": bool(torch.cuda.is_available()),
        "status": "SUCCESS",
    }

    # Save artifacts to output_dir
    metadata_path = output_dir / "run_metadata.json"
    metrics_path = output_dir / "metrics.json"
    checkpoint_path = output_dir / "smoke_test_model.pt"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), checkpoint_path)

    print("\nArtifacts saved:")
    print(f"  Metadata:   {metadata_path}")
    print(f"  Metrics:    {metrics_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 60)
    print("SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="AEPR Model Training & Runner")
    parser.add_argument("--data_dir", type=str, default="data/intermediate", help="Path to input data directory")
    parser.add_argument("--output_dir", type=str, default="results/smoke_test", help="Path to save outputs/checkpoints")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON/YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1-epoch smoke test to verify infrastructure")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if args.smoke_test:
        run_smoke_test(data_dir=data_dir, output_dir=output_dir, seed=args.seed)
    else:
        print("Production training loop not yet initialized. Use --smoke_test for infrastructure verification.")


if __name__ == "__main__":
    main()

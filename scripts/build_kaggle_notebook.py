"""
Builds the Kaggle training runner notebook (kaggle/runner_notebook.ipynb)
by bundling the local src/ package for identical execution on Kaggle GPU.
"""

import json
import base64
import tarfile
import io
from pathlib import Path


def create_src_tar_base64(src_dir: Path) -> str:
    """Creates a base64-encoded tar.gz archive of the local src/ directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(src_dir, arcname="src")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def build_notebook(base_dir: Path, output_ipynb: Path):
    src_dir = base_dir / "src"
    src_b64 = create_src_tar_base64(src_dir)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Auditory-Evoked Pupillary Responses (AEPR) - Kaggle Training Runner\n",
                "\n",
                "Automated execution environment for reproducible physiological deep learning training on Kaggle GPU."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ====================================================================\n",
                "# Cell 1: Environment, Pinned Dependency Checks & GPU Verification\n",
                "# ====================================================================\n",
                "import sys\n",
                "import os\n",
                "import json\n",
                "import torch\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import scipy\n",
                "import sklearn\n",
                "\n",
                "print('=' * 60)\n",
                "print('ENVIRONMENT & HARDWARE DIAGNOSTICS')\n",
                "print('=' * 60)\n",
                "print(f'Python Version:       {sys.version.split()[0]}')\n",
                "print(f'PyTorch Version:      {torch.__version__}')\n",
                "print(f'NumPy Version:        {np.__version__}')\n",
                "print(f'Pandas Version:       {pd.__version__}')\n",
                "print(f'SciPy Version:        {scipy.__version__}')\n",
                "print(f'Scikit-Learn Version: {sklearn.__version__}')\n",
                "\n",
                "cuda_available = torch.cuda.is_available()\n",
                "print(f'\\nCUDA Available:       {cuda_available}')\n",
                "if cuda_available:\n",
                "    gpu_count = torch.cuda.device_count()\n",
                "    gpu_name = torch.cuda.get_device_name(0)\n",
                "    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
                "    print(f'GPU Device Name:      {gpu_name}')\n",
                "    print(f'GPU Device Count:     {gpu_count}')\n",
                "    print(f'GPU Total Memory:     {gpu_mem:.2f} GB')\n",
                "    print(f'CUDA Version:         {torch.version.cuda}')\n",
                "else:\n",
                "    print('WARNING: Running in CPU mode. GPU was not detected!')\n",
                "print('=' * 60)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ====================================================================\n",
                "# Cell 2: Verify Mounted Kaggle Dataset\n",
                "# ====================================================================\n",
                "from pathlib import Path\n",
                "\n",
                "input_dir = Path('/kaggle/input')\n",
                "print(f'Mounted datasets in {input_dir}:')\n",
                "for p in input_dir.glob('*'):\n",
                "    print(f'  - {p.name} ({len(list(p.glob(\"**/*\")))} files)')\n",
                "\n",
                "aepr_dataset_dir = Path('/kaggle/input/aepr-pupillometry-dataset')\n",
                "if not aepr_dataset_dir.exists():\n",
                "    # Fallback search if dataset slug differs slightly\n",
                "    candidates = list(input_dir.glob('*pupil*')) + list(input_dir.glob('*aepr*'))\n",
                "    if candidates:\n",
                "        aepr_dataset_dir = candidates[0]\n",
                "        print(f'Using detected dataset path: {aepr_dataset_dir}')\n",
                "    else:\n",
                "        print(f'Notice: {aepr_dataset_dir} not directly found, scanning /kaggle/input')\n",
                "        aepr_dataset_dir = input_dir\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ====================================================================\n",
                "# Cell 3: Unpack Local Codebase (src/)\n",
                "# ====================================================================\n",
                "import base64\n",
                "import tarfile\n",
                "import io\n",
                "\n",
                "src_payload = '''" + src_b64 + "'''\n",
                "\n",
                "tar_bytes = base64.b64decode(src_payload.encode('ascii'))\n",
                "with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:gz') as tar:\n",
                "    tar.extractall(path='/kaggle/working')\n",
                "\n",
                "if '/kaggle/working' not in sys.path:\n",
                "    sys.path.insert(0, '/kaggle/working')\n",
                "\n",
                "import src\n",
                "print(f'Successfully loaded src package (version {src.__version__}) into Kaggle environment.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ====================================================================\n",
                "# Cell 4: Execute Universal Training / Smoke Test Runner\n",
                "# ====================================================================\n",
                "from src.train import run_smoke_test\n",
                "\n",
                "data_path = aepr_dataset_dir\n",
                "working_dir = Path('/kaggle/working')\n",
                "\n",
                "metrics = run_smoke_test(data_dir=data_path, output_dir=working_dir, seed=42)\n",
                "print('Execution returned metrics:', metrics)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ====================================================================\n",
                "# Cell 5: Validate Saved Outputs and Reproducibility Logs\n",
                "# ====================================================================\n",
                "print('\\nGenerated Outputs in /kaggle/working:')\n",
                "for f in sorted(list(working_dir.glob('*'))):\n",
                "    if f.is_file():\n",
                "        print(f'  - {f.name} ({f.stat().st_size} bytes)')\n",
                "\n",
                "meta_file = working_dir / 'run_metadata.json'\n",
                "if meta_file.exists():\n",
                "    print('\\n--- RUN METADATA ---')\n",
                "    with open(meta_file) as f:\n",
                "        print(f.read())\n",
                "\n",
                "metric_file = working_dir / 'metrics.json'\n",
                "if metric_file.exists():\n",
                "    print('\\n--- METRICS ---')\n",
                "    with open(metric_file) as f:\n",
                "        print(f.read())\n"
            ]
        }
    ]

    notebook_json = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    output_ipynb.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ipynb, "w", encoding="utf-8") as f:
        json.dump(notebook_json, f, indent=2)

    print(f"Generated Kaggle runner notebook at: {output_ipynb}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    output_ipynb = base_dir / "kaggle" / "runner_notebook.ipynb"
    build_notebook(base_dir, output_ipynb)

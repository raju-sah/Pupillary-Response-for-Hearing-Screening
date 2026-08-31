"""
Builds the Kaggle training runner notebook (kaggle/runner_notebook.ipynb)
by bundling the local src/ and scripts/ packages for accelerated execution on Kaggle GPU.
"""

import json
import base64
import tarfile
import io
from pathlib import Path


def create_payload_tar_base64(base_dir: Path) -> str:
    """Creates a base64-encoded tar.gz archive of local src/ and scripts/ directories."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(base_dir / "src", arcname="src")
        tar.add(base_dir / "scripts", arcname="scripts")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def build_notebook(base_dir: Path, output_ipynb: Path):
    payload_b64 = create_payload_tar_base64(base_dir)

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Auditory-Evoked Pupillary Responses (AEPR) - Kaggle Deep Learning Runner\n",
                "\n",
                "Automated GPU execution environment for training and benchmarking Deep Learning architectures on single-trial pupillometry time series."
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
                "if not (aepr_dataset_dir.exists() and (aepr_dataset_dir / 'dataset_b').exists()):\n",
                "    candidates = [p for p in input_dir.glob('*') if (p / 'dataset_b').exists()]\n",
                "    if candidates:\n",
                "        aepr_dataset_dir = candidates[0]\n",
                "        print(f'Using detected dataset path: {aepr_dataset_dir}')\n",
                "    else:\n",
                "        print(f'Notice: standard path not found, falling back to: {input_dir}')\n",
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
                "# Cell 3: Unpack Local Codebase (src/ and scripts/)\n",
                "# ====================================================================\n",
                "import base64\n",
                "import tarfile\n",
                "import io\n",
                "\n",
                "payload = '''" + payload_b64 + "'''\n",
                "\n",
                "tar_bytes = base64.b64decode(payload.encode('ascii'))\n",
                "with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:gz') as tar:\n",
                "    tar.extractall(path='/kaggle/working')\n",
                "\n",
                "if '/kaggle/working' not in sys.path:\n",
                "    sys.path.insert(0, '/kaggle/working')\n",
                "\n",
                "print('Successfully loaded local modules into /kaggle/working.')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ====================================================================\n",
                "# Cell 4: Execute Full Deep Learning Benchmarks on GPU\n",
                "# ====================================================================\n",
                "from scripts.run_deep_learning_benchmarks import run_all_deep_learning_benchmarks\n",
                "\n",
                "data_path = aepr_dataset_dir\n",
                "working_dir = Path('/kaggle/working')\n",
                "\n",
                "run_all_deep_learning_benchmarks(data_dir=data_path, output_dir=working_dir)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ====================================================================\n",
                "# Cell 5: Inspect Generated Reports & Figures\n",
                "# ====================================================================\n",
                "print('\\nGenerated Outputs in /kaggle/working:')\n",
                "for f in sorted(list(working_dir.rglob('*'))):\n",
                "    if f.is_file():\n",
                "        print(f'  - {f.relative_to(working_dir)} ({f.stat().st_size:,} bytes)')\n",
                "\n",
                "report_file = working_dir / 'DEEP_LEARNING_REPORT.md'\n",
                "if report_file.exists():\n",
                "    print('\\n' + '=' * 60)\n",
                "    print('DEEP_LEARNING_REPORT.md HEAD:')\n",
                "    print('=' * 60)\n",
                "    with open(report_file) as f:\n",
                "        print('\\n'.join(f.readlines()[:50]))\n"
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

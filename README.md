# Objective Hearing Screening via Auditory-Evoked Pupillary Responses (AEPR)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Target Venue](https://img.shields.io/badge/Target_Venue-IEEE_TBME_/_Nature_SciRep-success.svg?style=flat-square)]()
[![SOTA ROC-AUC](https://img.shields.io/badge/SOTA_ROC--AUC-0.844-brightgreen.svg?style=flat-square)]()
[![Leakage-Free](https://img.shields.io/badge/Validation-Group_K--Fold_(Zero_Leakage)-orange.svg?style=flat-square)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg?style=flat-square)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-success.svg?style=flat-square&logo=github)](https://raju-sah.github.io/Pupillary-Response-for-Hearing-Screening/)

An end-to-end deep representation learning and computer vision framework for **objective, non-invasive hearing screening and acoustic salience detection** using Auditory-Evoked Pupillary Responses (AEPR).

---

## 📌 Executive Summary

Traditional neonatal and pediatric objective hearing screening relies on **Auditory Brainstem Responses (ABR)** or **Transient-Evoked Otoacoustic Emissions (TEOAE)**. Despite their clinical utility, both modalities exhibit significant real-world constraints:
* **ABR:** Requires skin abrasion, gel electrodes, and deep sedation/immobility in pediatric patients.
* **OAE:** Highly vulnerable to ear canal vernix, cerumen obstruction, and ambient acoustic seal loss.
* **Cost:** Specialized audiometric hardware restricts universal deployment in low-resource community health centers.

This repository implements a leak-free, reproducible machine learning and computer vision pipeline that transforms commodity eye-tracking cameras into **objective optical hearing screening devices**. By decoding the autonomic subcortical **Locus Coeruleus–Norepinephrine (LC-NE)** pupillary dilation pathway, our deep neural models classify single-trial auditory responses with high statistical precision.

```
                  ┌────────────────────────────────────────────────────────┐
                  │              Auditory Stimulus Presentation            │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │    Subcortical Locus Coeruleus (LC-NE) Activation     │
                  │   Sympathetic Pupillary Dilation (Latency: 200-400ms)  │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
         ┌──────────────────────────────┐            ┌──────────────────────────────┐
         │ Commercial Eye-Tracker Data  │            │ Raw 30 fps Eye Video Stream  │
         │   (Calibrated Diameter mm)   │            │ (Least-Squares Ellipse Fit)  │
         └──────────────┬───────────────┘            └──────────────┬───────────────┘
                        │                                           │
                        └─────────────────────┬─────────────────────┘
                                              │ Pearson r = 0.991
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │ Artifact Decontamination & Baseline Normalization      │
                  │ - Monotonic Cubic Spline Blink Interpolation           │
                  │ - Zero-Phase 4th-Order Butterworth Bandpass (0.05-4Hz) │
                  │ - Pre-stimulus Baseline Normalization (ΔP(t), %ΔP(t))  │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
         ┌──────────────────────────────┐            ┌──────────────────────────────┐
         │ 25 Handcrafted Domain Features│            │ Raw Multi-Channel Tensors   │
         │ (Peak Amp, Latency, Slopes)  │            │ [ΔP(t), %ΔP(t), dΔP/dt]      │
         └──────────────┬───────────────┘            └──────────────┬───────────────┘
                        │                                           │
                        ▼                                           ▼
         ┌──────────────────────────────┐            ┌──────────────────────────────┐
         │ Classical Baselines          │            │ Deep Neural Architectures    │
         │ - HistGradientBoosting (0.810│            │ - Multi-Scale 1D-CNN (0.744) │
         │ - Random Forest (0.808)      │            │ - Dilated TCN (0.731)        │
         │ - Logistic Regression (0.763)│            │ - Bi-LSTM Attention (0.755)  │
         └──────────────────────────────┘            │ - CNN-Transformer (★ 0.844) │
                                                     └──────────────┬───────────────┘
                                                                    │
                                                                    ▼
                                                     ┌──────────────────────────────┐
                                                     │ SOTA ROC-AUC: 0.844          │
                                                     │ Temporal Saliency: 0.8-2.2s  │
                                                     │ Truncated Latency: 1.5s      │
                                                     └──────────────────────────────┘
```

---

## 🏆 Key Scientific Findings

1. **State-of-the-Art Deep Learning Benchmark:**
   - Evaluated under strictly subject-independent 5-fold cross-validation (`StratifiedGroupKFold`) across $N=66$ subjects ($18,066$ single-trial epochs).
   - Our **CNN-Transformer** achieves an **ROC-AUC of 0.844** (95% CI: [0.835, 0.853]) and **PR-AUC of 0.512**, demonstrating statistically significant superiority over top gradient-boosted ensembles ($\Delta\text{ROC-AUC} = +0.034, p < 10^{-15}$; DeLong paired test $Z = 9.48, p < 10^{-15}$).
2. **Autonomous Physiological Saliency:**
   - Learned temporal attention weights independently localize the primary discriminative window to **$t \in [0.8\text{s}, 2.2\text{s}]$ post-stimulus** (peaking at $1.35\text{s}$), mirroring human LC-NE autonomic pupillary dilation kinetics without manual feature engineering.
3. **50% Examination Duration Reduction:**
   - Early detection latency truncation experiments reveal that trial observation windows can be cut to **$t = 1.5\text{s}$ post-stimulus** while preserving **$>98\%$ of maximal screening accuracy**, cutting patient test duration in half.
4. **Hardware Downsampling Resilience:**
   - Downsampling sweeps confirm that frame rates can be lowered from 50 Hz down to 10 Hz with negligible performance degradation ($\Delta\text{AUC} < 0.005$).
5. **Direct Video Extraction Concordance:**
   - Direct least-squares ellipse fitting on raw 30 fps eye video achieves **Pearson $r = 0.991$** and **Spearman $\rho = 0.989$ ($p < 10^{-15}$)** against dedicated commercial infrared eye-tracking hardware.

---

## 📊 Comprehensive Benchmark Results

### Task 1: Single-Trial Acoustic Salience (PsPM-AOB, $N=66$, 18,066 Trials)

| Model Architecture | Model Class | ROC-AUC [95% CI] | PR-AUC [95% CI] | Bal. Acc | Sens | Spec | Brier Score | DeLong $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN-Transformer** | **Deep Learning** | **0.844** [0.835, 0.853] | **0.512** [0.490, 0.538] | **0.771** | 0.748 | 0.794 | **0.131** | **Reference** |
| **HistGradientBoosting** | Classical Tree | 0.810 [0.801, 0.820] | 0.462 [0.442, 0.484] | 0.740 | 0.706 | 0.775 | 0.148 | $Z = 9.48, p < 10^{-15}$ |
| **Random Forest** | Classical Tree | 0.808 [0.798, 0.818] | 0.450 [0.431, 0.470] | 0.741 | 0.677 | 0.805 | 0.117 | $Z = 9.87, p < 10^{-15}$ |
| **Logistic Regression** | Classical Linear | 0.763 [0.752, 0.773] | 0.326 [0.307, 0.346] | 0.701 | 0.710 | 0.692 | 0.199 | $Z = 18.2, p < 10^{-15}$ |
| **Bi-LSTM + Attention** | Deep Learning | 0.755 [0.744, 0.766] | 0.359 [0.340, 0.382] | 0.701 | 0.665 | 0.736 | 0.166 | $Z = 17.5, p < 10^{-15}$ |
| **Multi-Scale 1D-CNN** | Deep Learning | 0.744 [0.733, 0.755] | 0.325 [0.305, 0.349] | 0.685 | 0.632 | 0.738 | 0.167 | $Z = 19.8, p < 10^{-15}$ |
| **Dilated TCN** | Deep Learning | 0.731 [0.720, 0.743] | 0.291 [0.274, 0.311] | 0.675 | 0.721 | 0.629 | 0.172 | $Z = 22.4, p < 10^{-15}$ |
| **Single-Feature Heuristic** | Physiological | 0.646 [0.635, 0.658] | 0.177 [0.168, 0.188] | 0.618 | 0.715 | 0.520 | 0.237 | $Z = 33.6, p < 10^{-15}$ |
| **Chance Baseline** | Null Model | 0.493 [0.482, 0.505] | 0.120 [0.114, 0.126] | 0.500 | 0.000 | 1.000 | 0.107 | — |

*All evaluations utilize 5-Fold `StratifiedGroupKFold` grouped strictly by subject ID to prevent inter-subject data leakage.*

---

## 📈 Visual Gallery

| ROC Curves: Deep Learning vs Classical | Temporal Attention Weights (LC-NE Window) |
| :---: | :---: |
| ![ROC Curves](results/figures/dl_vs_classical_roc_task1.png) | ![Temporal Attention](results/figures/temporal_attention_weights.png) |
| **Precision-Recall Curves** | **Early Latency Truncation (1.5s Window)** |
| ![PR Curves](results/figures/dl_vs_classical_pr_task1.png) | ![Early Latency](results/figures/robustness_early_detection_latency.png) |
| **CV Video Extraction vs Hardware** | **Subject Cohort Scaling Curve** |
| ![CV vs Hardware](results/figures/cv_vs_provided_trace_overlay.png) | ![Cohort Scaling](results/figures/subject_cohort_scaling_curve.png) |

---

## 🗂️ Repository Architecture

```
Pupillary-Response-for-Hearing-Screening/
├── data/
│   ├── raw/                           # Raw downloaded archives
│   │   ├── dataset_a_zenodo/          # APURE dataset (Zenodo DOI: 10.5281/zenodo.10497437)
│   │   └── dataset_b_pspm_aob/        # PsPM-AOB dataset (Zenodo DOI: 10.5281/zenodo.7738240)
│   ├── intermediate/                  # Standardized parquet recordings
│   └── processed/                     # Preprocessed epoch tensors and reports
├── figures/                           # High-resolution publication figures
├── results/
│   └── figures/                       # 27 generated empirical figures
├── src/                               # Core Python library
│   ├── classical_models.py            # Feature extractors and scikit-learn pipelines
│   ├── cv_pupil_extraction.py         # OpenCV least-squares ellipse fitting
│   ├── deep_learning_models.py        # 1D-CNN, TCN, Bi-LSTM Attention, CNN-Transformer
│   ├── deep_learning_trainer.py       # GroupKFold trainer with DeLong tests
│   ├── download.py / download_b.py    # Multi-threaded parallel dataset downloaders
│   ├── feature_extraction.py          # 25 domain-informed physiological features
│   ├── parser_a.py / parser_b.py      # BIDS/MAT/Excel parser implementations
│   ├── physiological_baseline.py      # Heuristic baselines & statistical testing
│   ├── preprocessing.py               # Monotonic spline & Butterworth filters
│   ├── quality_audit.py               # Missingness, blink duration, SNR audit
│   ├── robustness.py                  # Noise, dropout, latency, downsampling sweeps
│   └── schema.py                      # Data validation contracts
├── scripts/                           # Reproducible experiment execution runners
│   ├── run_preprocessing.py           # Epoch segmentation & filtering
│   ├── run_classical_ml_baselines.py  # Step 6 evaluation
│   ├── run_deep_learning_benchmarks.py# Steps 7-9 DL architectures
│   ├── run_robustness_experiments.py  # Step 11 stress tests
│   ├── run_external_dataset_validation.py # Step 12 cross-dataset transfer
│   ├── run_cv_vs_provided_comparison.py  # Steps 13-14 video ellipse extraction
│   └── run_ablation_and_statistical_analysis.py # Step 16 significance
├── manuscript/                        # IEEE TBME LaTeX paper & BibTeX database
├── webapp/                            # Interactive Clinical Screening Web Dashboard
├── tests/                             # Pytest test suite
├── Steps.md                           # 17-step master roadmap
├── PAPER_MANUSCRIPT.md                # Full markdown scientific paper
└── README.md                          # Repository documentation
```

---

## ⚡ Quickstart & Reproducibility

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/raju-sah/Pupillary-Response-for-Hearing-Screening.git
cd Pupillary-Response-for-Hearing-Screening

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Preprocessing Pipeline
```bash
python scripts/run_preprocessing.py
```

### 3. Run Classical ML Baselines (Step 6)
```bash
python scripts/run_classical_ml_baselines.py
```

### 4. Run Deep Learning Benchmarks (Steps 7–9)
```bash
python scripts/run_deep_learning_benchmarks.py
```

### 5. Run Robustness & Latency Stress Tests (Step 11)
```bash
python scripts/run_robustness_experiments.py
```

### 6. Run Computer Vision Video Ellipse Extraction (Steps 13–14)
```bash
python scripts/run_cv_vs_provided_comparison.py
```

### 7. Run Full Unit Test Suite
```bash
pytest tests/ -v
```

---

## 💻 Interactive Screening Web Dashboard

* **Live Cloud Deployment (GitHub Pages):** [https://raju-sah.github.io/Pupillary-Response-for-Hearing-Screening/](https://raju-sah.github.io/Pupillary-Response-for-Hearing-Screening/)
* **Local Runner:**
```bash
python webapp/server.py
```
Open your browser at `http://localhost:8088` to inspect raw pupil traces, toggle filtering stages, test observation window truncation sliders, and view live CNN-Transformer inference with LC-NE attention saliency.

---

## 📖 Citation

If you find this work or codebase useful in your research, please cite:

```bibtex
@article{sah2026aepr,
  title={End-to-End Deep Representation Learning and Computer Vision for Objective Hearing Screening via Auditory-Evoked Pupillary Responses},
  author={Sah, Raju and Research Collaborators},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2026},
  note={Under Review}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Datasets PsPM-AOB and APURE are distributed under their respective Creative Commons licenses (CC-BY 4.0 and CC0).

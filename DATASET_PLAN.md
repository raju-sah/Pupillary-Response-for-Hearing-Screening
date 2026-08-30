# Dataset Plan

This project will rely entirely on publicly available datasets, avoiding the significant cost, time, and infrastructure required to collect high-quality physiological data. 

## 1. The Core Datasets

### Dataset A: "Pupil Data Upon Stimulation by Auditory Stimuli" (Zenodo)
*   **Participants:** 16 subjects.
*   **Details:** Auditory tones under different lighting conditions.
*   **Crucial Feature:** This dataset provides both pre-extracted 1D pupil measurements AND the original ~60 FPS IR eye-camera recordings.
*   **Role:** This is the foundational dataset. It allows for dual-track research: 
    *   *Track A:* Physiological time-series research on the 1D signal.
    *   *Track B:* Computer vision research benchmarking CV pipelines against the provided ground-truth IR video.

### Dataset B: PsPM-AOB
*   **Participants:** 66 subjects.
*   **Details:** Pupillometry collected during auditory oddball tasks (440/660 Hz tones).
*   **Role:** The external validation dataset. Models trained on Dataset A will be evaluated on Dataset B to test cross-dataset generalization.

### Dataset C: OpenNeuro ds003690
*   **Participants:** 75 subjects (36 young adults, 39 older adults).
*   **Details:** Synchronized EEG, ECG, and pupillography (240 Hz) during auditory reaction-time tasks and passive listening.
*   **Role:** Allows for age-generalization experiments (testing if models trained on young adults work on older adults) and multimodal extensions (combining Pupil + EEG).

## 2. The Computer Vision Strategy
Because Dataset A includes raw IR video, we do not need to abandon the Computer Vision aspect of the project. However, the CV aspect is reframed as a **Reproducibility/Benchmarking experiment**.

We will build a CV pipeline (Pupil detection → Diameter estimation → Temporal smoothing) and compare our extracted signal against the provided ground-truth pupil signal. 
*   **Metrics for CV Validation:** Mean Absolute Error (MAE), RMSE, Pearson correlation, and Temporal alignment.

# Experiment Plan

To progress from a "Kaggle project" to a publishable research paper, you must structure the work incrementally. Here are 4 proposed experiments.

## Experiment 1: The Signal Processing Baseline (1D)
*   **Goal:** Establish baseline performance using classical methods on a public 1D dataset (e.g., the Zenodo pure-tone dataset).
*   **Method:** 
    *   Preprocess the 1D pupil diameter data (blink interpolation, low-pass filter, baseline correction).
    *   Extract handcrafted features: Peak Pupil Dilation (PPD), Latency to Peak, Mean Dilation in a 2-second window post-stimulus.
    *   **Evaluation:** Statistical testing (e.g., paired t-tests or Linear Mixed Effects models) to prove a significant difference in pupil size between pre-stimulus and post-stimulus windows.
*   **Why:** You must prove you understand the physiology before throwing Deep Learning at it.

## Experiment 2: The ML Sequence Modeling (1D)
*   **Goal:** Determine if Deep Learning outperforms handcrafted features on 1D pupillary time-series.
*   **Method:**
    *   Frame it as a classification task (e.g., Auditory Stimulus Present vs. Absent, or High Noise vs. Low Noise).
    *   **Baselines:** SVM, Random Forest (using PCA features).
    *   **Temporal Models:** 1D-CNN, LSTM, and a Time-Series Transformer. 
    *   **Evaluation:** Leave-One-Subject-Out Cross-Validation (LOSO-CV). Report AUROC and F1-score.

## Experiment 3: CV-Extraction vs. Ground Truth (If collecting data)
*   **Goal:** Validate whether standard webcams can measure the micro-fluctuations required for AEPR.
*   **Method:**
    *   Record participants with a webcam during an auditory oddball task.
    *   Run MediaPipe Iris or OpenFace to extract the 1D pupil signal from the video.
    *   Compare the SNR (Signal-to-Noise Ratio) of the CV-extracted signal against the expected physiological response. Can you statistically detect the pupil dilation in the CV data?

## Experiment 4: Spatiotemporal End-to-End Prediction (The Novel Contribution)
*   **Goal:** Bypass 1D extraction entirely. Predict auditory cognitive load directly from video frames of the eye.
*   **Method:**
    *   Crop the eye region from the video stream.
    *   Train a spatiotemporal model (e.g., 3D-ResNet or Video Vision Transformer) to classify the cognitive load / listening condition directly from the raw pixel data over time.
    *   **Why this is huge:** This allows the model to implicitly learn robust representations that ignore lighting, blinks, and eye color, whereas standard 1D extraction pipelines fail under these conditions.

## Statistical Evaluation Framework
*   **For Physiology:** Use cluster-based permutation testing for time-series data to find the exact time windows where responses differ significantly, correcting for multiple comparisons.
*   **For ML:** Use DeLong's test to compare the AUROC of different models. Ensure all confidence intervals are calculated across *subjects*, not samples.

# Experiment Plan

This plan outlines a sequence of rigorous experiments, starting from basic physiological validation and scaling up to cross-dataset deep learning and multimodal AI.

## Experiment 1: Physiological Baseline
*   **Dataset:** Dataset A (16 subjects).
*   **Goal:** Prove understanding of the physiological signal before applying machine learning.
*   **Method:** Extract classical physiological features: baseline pupil diameter, peak dilation, dilation amplitude, latency to peak, recovery time, area under the response curve, and normalized pupil response.
*   **Analysis:** Statistically investigate whether the auditory stimulation produces a measurable Pupillary Dilation Response (PDR).

## Experiment 2: Machine Learning (Classical vs. Deep Learning)
*   **Dataset:** Dataset A.
*   **Target Label:** Auditory stimulus condition or intensity (NOT hearing loss).
*   **Method:** Compare performance across a spectrum of complexity:
    *   *Classical ML:* Logistic Regression, SVM, Random Forest, XGBoost.
    *   *Deep Learning:* 1D CNN, LSTM, GRU, Temporal Transformer.
*   **Evaluation:** Strict Subject-Independent Cross-Validation. Report Macro-F1, AUROC, PR-AUC, Sensitivity, and Specificity.

## Experiment 3: Cross-Dataset Generalization (Domain Shift)
*   **Datasets:** Train on Dataset A (16-subject), Test on Dataset B (66-subject PsPM-AOB).
*   **Goal:** Answer the research question: *How well do pupillary-response models generalize across datasets, experimental protocols, and populations?*
*   **Analysis:** Measure the performance drop when applying the Dataset A model to Dataset B. Analyze the domain shift.

## Experiment 4: Age Generalization
*   **Dataset:** Dataset C (OpenNeuro ds003690 - 75 subjects).
*   **Goal:** Answer the research question: *Does pupil-based auditory-response modeling generalize across age groups?*
*   **Method:** Train models on the young adult cohort (N=36) and test on the older adult cohort (N=39) using only the pupil signal.

## Experiment 5: Multimodal Extension (Optional)
*   **Dataset:** Dataset C.
*   **Goal:** Investigate if combining physiological signals improves cognitive state estimation.
*   **Method:** Fuse the Pupil, EEG, and ECG signals into a multimodal deep learning architecture.

## Experiment 6: Computer Vision Validation (Optional)
*   **Dataset:** Dataset A (IR Video).
*   **Goal:** Benchmark an open-source or custom CV pupil extraction pipeline against the dataset's provided ground-truth signal.
*   **Metrics:** Pearson correlation, SNR, and MAE between the CV-extracted signal and the provided signal.

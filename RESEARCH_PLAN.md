# Research Plan

## 1. Central Research Question
Based on a rigorous scientific audit of available public datasets, the research question is strictly formulated as:
**"Can deep temporal models learn robust representations of auditory-evoked pupillary responses (AEPR) that generalize across subjects, experimental conditions, and datasets?"**

## 2. Scientific Constraints and Pivot
*   **No Hearing Loss Diagnosis:** None of the public datasets provide clinical audiological ground truth. Claiming diagnostic capabilities is scientifically invalid.
*   **No "Listening Effort" Estimation:** Listening effort is clinically defined by speech-in-noise or dual-task paradigms. The datasets available provide passive tones, oddball tasks, and Go-NoGo tasks. Thus, the project investigates fundamental **auditory-evoked cognitive responses** (orienting reflex, surprise, motor inhibition), not explicit listening effort.
*   **No Participant Data Collection:** The project relies entirely on three distinct, publicly available datasets. The scientific contribution is methodological rigor, cross-dataset validation, and reproducibility, not data ownership.

## 3. Project Objectives
1.  Establish a standardized, reproducible preprocessing pipeline for physiological pupil signals.
2.  Demonstrate the performance gap between classical statistical feature extraction (e.g., peak pupil dilation) and deep temporal representation learning (e.g., LSTMs, Transformers).
3.  Expose the limitations of subject-specific modeling by enforcing strict Leave-One-Subject-Out (LOSO) cross-validation.
4.  Investigate domain-shift degradation by testing models across entirely different experimental datasets.
5.  Determine if pupil-based cognitive models generalize across age demographics.

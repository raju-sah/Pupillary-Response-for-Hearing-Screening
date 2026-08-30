# Experiment Plan

This sequence of experiments is strictly bound by the actual targets available in the audited public datasets.

## Experiment 1: Physiological Characterization
*   **Research Question:** Does auditory stimulation produce a statistically significant pupillary dilation response (PDR) independent of lighting conditions?
*   **Dataset:** Dataset A.
*   **Variables:** IV: Auditory tone presence. DV: Pupil diameter.
*   **Target:** N/A (Statistical extraction).
*   **Statistical Tests:** Paired t-tests on Peak Pupil Dilation (PPD); cluster-based permutation testing on the time-series.
*   **Limitations:** High variance across subjects; luminance changes may mask the cognitive response.

## Experiment 2: Classical ML vs Deep Temporal Models
*   **Research Question:** Do deep temporal models outperform classical ML in detecting auditory-evoked responses?
*   **Dataset:** Dataset A.
*   **Target:** Stimulus Present vs. Baseline Window.
*   **Validation Strategy:** Leave-One-Subject-Out (LOSO) CV.
*   **Models:** 
    *   *Baseline:* Logistic Regression, SVM (using handcrafted PPD features).
    *   *Deep:* 1D-CNN, LSTM, Temporal Transformer.
*   **Metrics:** Macro-F1, AUROC, PR-AUC.
*   **Leakage Prevention:** Strict LOSO CV; baseline correction applied per-epoch before train/test split.

## Experiment 3: Subject-Independent Generalization
*   **Research Question:** Can temporal models accurately classify auditory deviance across unseen subjects?
*   **Dataset:** Dataset B (66 subjects).
*   **Target:** Deviant Tone vs Standard Tone.
*   **Validation Strategy:** LOSO CV.
*   **Models:** LSTM, Temporal Transformer.

## Experiment 4: Cross-Dataset/Domain-Shift Evaluation
*   **Research Question:** How significantly does model performance degrade when trained on passive tone listening and tested on an auditory oddball paradigm?
*   **Datasets:** Train on Dataset A, Test on Dataset B.
*   **Target:** Tone Present (from A) / Deviant Present (from B) vs Baseline.
*   **Metrics:** Performance drop (AUROC diff) between internal validation and external testing.
*   **Limitations:** The cognitive tasks differ (passive vs oddball), which may naturally limit generalizability.

## Experiment 5: Age-Related Generalization
*   **Research Question:** Do pupillary response representations learned from young adults generalize to older adults?
*   **Dataset:** Dataset C.
*   **Target:** Task Type (Rest vs Go/NoGo).
*   **Validation Strategy:** Train on Young Adult cohort (N=36), Test on Older Adult cohort (N=39).
*   **Metrics:** Macro-F1, AUROC.
*   **Limitations:** Older adults have smaller baseline pupils and slower reactivity (senile miosis), likely causing a severe domain shift.

## Experiment 6: Computer-Vision Pupil Extraction
*   **Research Question:** Can an open-source CV pipeline match the accuracy of the provided 1D pupil signal?
*   **Dataset:** Dataset A (IR Video).
*   **Target:** 1D Pupil Diameter.
*   **Models:** MediaPipe Iris / OpenFace.
*   **Metrics:** Pearson correlation, MAE between CV output and provided ground-truth signal.
*   **Limitations:** Subject to video quality and blink occlusion.

## Experiment 7: Optional Multimodal Analysis
*   **Research Question:** Does combining EEG, ECG, and Pupillometry improve auditory cognitive state classification?
*   **Dataset:** Dataset C.
*   **Target:** Task Type (Rest vs Go/NoGo).
*   **Models:** Multimodal Fusion Transformer.

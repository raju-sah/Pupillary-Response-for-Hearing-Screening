# Risk and Leakage Assessment

## 1. Identity / Subject Leakage
*   **The Risk:** Physiological signals are highly idiosyncratic. If Subject 1's data appears in both the training set and the test set, the model will learn Subject 1's baseline pupil size and resting dynamics rather than the auditory-evoked response.
*   **The Mitigation:** Random data splitting is strictly forbidden. All models must be evaluated using Leave-One-Subject-Out (LOSO) cross-validation or GroupKFold splitting based on subject ID.

## 2. Age-Related Confounders (Senile Miosis)
*   **The Risk:** As humans age, their resting pupil size decreases, and their pupil reactivity (amplitude of dilation) diminishes. A model trained exclusively on young adults will likely fail on older adults due to this physiological domain shift.
*   **The Mitigation:** Explicitly model and quantify this degradation in Experiment 5 by training on the young cohort and testing on the older cohort using Dataset C.

## 3. The Pupillary Light Reflex (PLR) Leakage
*   **The Risk:** Dataset A contains varying luminance conditions. The pupillary response to light (PLR) is biologically much stronger than the cognitive auditory-evoked response. If the model correlates a specific lighting change with a tone, it will learn the PLR instead of the auditory response.
*   **The Mitigation:** Subtractive or divisive baseline correction must be applied to every epoch independently to normalize the pre-stimulus pupil size to zero or one. 

## 4. Hardware and Preprocessing Leakage
*   **The Risk:** Dataset A and Dataset B were likely collected using different eye-trackers with different sampling rates, noise profiles, and proprietary smoothing algorithms. When performing cross-dataset generalization (Experiment 4), the model might simply learn to classify the hardware sensor noise.
*   **The Mitigation:** Both datasets must be strictly downsampled to the same frequency, standardized, and filtered using identical Butterworth filters before model ingestion.

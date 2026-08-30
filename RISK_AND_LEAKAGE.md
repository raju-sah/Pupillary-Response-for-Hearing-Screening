# Risk, Leakage, and Confounders

## 1. Dataset Leakage Risks
In physiological machine learning, data leakage is the most common reason for inflated, non-reproducible results.
*   **Subject Leakage (Identity Leakage):** You cannot randomly split a dataset by frames or epochs. The pupil dynamics of Subject A are highly correlated with themselves. If Subject A is in both the train and test sets, the model learns to identify Subject A's baseline pupil size, not the auditory response. **Requirement: Strict Leave-One-Subject-Out (LOSO) cross-validation.**
*   **Temporal Leakage:** When using time-series models (like bidirectional LSTMs or Transformers without causal masking) in real-time prediction scenarios, you risk allowing the model to "see the future" pupil state to predict a current auditory event.
*   **Hardware Leakage:** If the "hearing loss" group was recorded on a different camera or in a different room than the "control" group, the model will learn to classify the camera sensor noise or lighting, not the medical condition.

## 2. Potential Confounders (The Skeptic's Checklist)
Pupil dilation is a highly overloaded signal. You must control for, or mathematically regress out, the following:
*   **The Pupillary Light Reflex (PLR):** The pupil's response to light is an order of magnitude stronger than its response to sound/cognitive load. If the screen brightness changes (e.g., a video plays, or a white popup appears) during the auditory test, the data is entirely corrupted.
*   **Age (Senile Miosis):** Older adults have naturally smaller pupils that react less dynamically. If your models are trained on healthy 20-year-olds, they will fail on 70-year-olds (the primary demographic for hearing issues).
*   **Medication and Caffeine:** Stimulants (Adderall, high caffeine) and depressants/antidepressants drastically alter baseline pupil size and reactivity.
*   **Cognitive Fatigue:** The pupil response diminishes over time as the subject gets tired. The 50th auditory trial will have a smaller response than the 1st trial.
*   **Eye Color / Iris Pigmentation:** Computer vision algorithms often struggle to segment the pupil from dark brown irises compared to blue irises, creating demographic bias in the system's accuracy.

## 3. Ethical and Medical Limitations
*   **Misdiagnosis Risk:** Claiming this is a "screening" tool is dangerous. A lack of pupil response could be due to fatigue, medication, or inattention, leading to false positives for hearing loss.
*   **Diagnostic vs. Adjunct:** This technology should be positioned as an *adjunct* to understand a patient's cognitive burden (listening effort), not a standalone diagnostic tool.

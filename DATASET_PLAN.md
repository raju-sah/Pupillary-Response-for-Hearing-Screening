# Dataset Plan

This project relies exclusively on public datasets. No new human participant data will be collected.

## 1. Dataset A: "Pupil Data Upon Stimulation by Auditory Stimuli"
*   **DOI:** 10.5281/zenodo.10497437
*   **Subjects:** 16
*   **Task:** Passive listening to 60-100 pure audible tones under varying lighting conditions.
*   **Data Available:** 1D pupil size, shape, and raw ~60 FPS IR eye video.
*   **Role in Project:** Serves as the foundation for physiological baseline verification, initial ML model training (Target: Tone vs Baseline), and the Computer Vision benchmark experiment.

## 2. Dataset B: PsPM-AOB
*   **DOI:** 10.5281/zenodo.3608706
*   **Subjects:** 66
*   **Task:** Auditory Oddball Task (440 Hz standard tones, 660 Hz deviant tones).
*   **Data Available:** 1D pupil size.
*   **Role in Project:** Serves as the external validation dataset for cross-domain generalization (Dataset A to B) and subject-independent modeling for auditory deviance (Target: Standard vs Deviant).

## 3. Dataset C: OpenNeuro ds003690
*   **DOI:** 10.18112/openneuro.ds003690.v1.0.0
*   **Subjects:** 75 (36 young adults, 39 older adults)
*   **Task:** Rest, Simple Reaction Time (RT), and Go/NoGo tasks.
*   **Data Available:** 1D pupil size (240 Hz), EEG, ECG.
*   **Role in Project:** Facilitates age-related generalization experiments (Target: Young vs Older) and the optional multimodal extension (Pupil + EEG + ECG).

## 4. Limitations
*   Lack of speech-in-noise tasks prevents direct estimation of standard audiological "listening effort".
*   Dataset A contains varying luminance, which introduces a massive confounder (Pupillary Light Reflex) that models must learn to ignore.
*   Dataset C includes artifacts from the physical constraints of the EEG forehead rest.

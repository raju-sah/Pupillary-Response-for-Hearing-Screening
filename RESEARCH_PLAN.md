# Research Plan: Objective Hearing Screening from Auditory-Evoked Pupillary Responses (AEPR) Using Computer Vision

## 1. Critical Evaluation of the Research Question
**The premise:** Using computer vision to measure pupillary responses for objective hearing screening.

**The Skeptical View:** While pupillometry is a well-established physiological measure, using it as a direct replacement for pure-tone audiometry (detecting hearing thresholds) is scientifically tenuous.
*   **Why it's weak:** The pupillary response to sound is primarily driven by arousal, novelty (the orienting reflex), and cognitive load (listening effort), not the sheer mechanical ability to hear a sound. A lack of pupillary response does not definitively mean deafness; it could mean habituation, fatigue, or lack of attention. Conversely, a response might be triggered by non-auditory stimuli in a poorly controlled environment.
*   **The pivot:** Instead of "Hearing Threshold Screening," frame the project around **"Assessing Auditory-Cognitive Load"** or **"Screening for Hidden Hearing Loss / Auditory Processing Difficulties."** Pupillometry is the gold standard for measuring the *listening effort* required to understand speech in noise. This is clinically highly relevant, as standard audiograms fail to capture why some patients with "normal" hearing struggle in noisy environments.

## 2. Physiological Signal Preprocessing Recommendations
Pupillometry data is notoriously noisy. If working with 1D extracted signals, the following pipeline is mandatory for publishable research:
1.  **Blink Detection and Removal:** Blinks cause artificial drops to 0. Use velocity-based algorithms to detect blink onset and offset.
2.  **Interpolation:** Linearly or cubically interpolate missing data during blinks.
3.  **Filtering:** Apply a low-pass filter (e.g., Butterworth, cutoff ~4-10 Hz) to remove high-frequency camera noise, and a high-pass filter to remove slow baseline drift.
4.  **Baseline Correction:** This is critical. Subtract or divide the pupil size by a baseline window (e.g., 1 second before the sound stimulus). AEPRs are tiny (~0.1mm to 0.5mm changes), so relative change from baseline is the only reliable metric.

## 3. Computer Vision Viability
Most clinical pupillometry uses expensive, controlled infrared (IR) eyetrackers.
*   **The CV Challenge:** Standard RGB webcams struggle with low light (where pupils are larger and easier to measure), motion blur, and lack of contrast between dark irises and the pupil.
*   **The Opportunity:** Developing a robust, low-cost CV pipeline that can extract a reliable 1D pupil signal from a standard webcam, compensating for head movement and ambient light, is a major engineering contribution.

## 4. Overall Assessment for a PhD / RA Portfolio
This is a highly interdisciplinary and complex project. If executed well, it demonstrates skills in physiological signal processing, computer vision, time-series machine learning, and experimental design. To succeed, you must abandon the naive assumption that `pupil dilation == hearing ability` and embrace the complex neuroscience of the Locus Coeruleus-Norepinephrine (LC-NE) arousal system.

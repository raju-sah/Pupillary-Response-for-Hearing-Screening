# Dataset Plan

## 1. Public Datasets
Finding datasets containing *raw video* of eyes during auditory tasks is extremely difficult. Most datasets only provide the pre-extracted 1D pupil diameter time-series, which limits your ability to do novel Computer Vision research.

**Available 1D Time-Series Datasets:**
1.  **"Pupil Data Upon Stimulation by Auditory Stimuli" (Zenodo):** 16 subjects, pure tone stimuli under various lighting. Good for baseline 1D analysis, but tiny.
2.  **Auditory Aging Lab Open Data:** MEG and eye-movement data during gap detection tasks.
3.  **UCL Research Data Repository:** Datasets on listening effort in young/older listeners.
4.  **OpenNeuro:** Datasets involving cognitive load (e.g., Digit Span Task) often include pupillometry.

## 2. The Major Flaw & Recommendation
**The Flaw:** If you only use public 1D datasets, you aren't doing "Computer Vision"—you are doing 1D Time-Series Analysis. The CV aspect (extracting the pupil from video) was already done by the researchers who published the dataset using proprietary IR software.

**The Recommendation:** You must choose one of two paths:
*   **Path A (Time-Series Focus):** Abandon the "Computer Vision" aspect. Focus purely on deep learning architectures for 1D physiological signals using public datasets to predict listening effort.
*   **Path B (Computer Vision Focus - Highly Recommended):** You must collect your own dataset.

## 3. Proposed Dataset Collection Protocol (If Path B)
If you collect data, it will elevate the project immensely.
*   **Setup:** A participant sits in front of a laptop. Record them simultaneously with a standard 1080p RGB webcam and a low-cost IR camera (or specialized tracker if available).
*   **Stimuli:** 
    *   *Task 1: Auditory Oddball.* A sequence of standard tones (1000 Hz) with rare "deviant" tones (2000 Hz). This reliably triggers a pupil dilation response.
    *   *Task 2: Speech in Noise.* Playing sentences with varying levels of background babble noise.
*   **Lighting:** Must be constant. The screen must display a fixed gray background. Screen brightness changes will destroy the data via the Pupillary Light Reflex.
*   **Ground Truth:** Use an open-source CV tool like MediaPipe Iris or OpenFace to extract 1D signals as a baseline, and compare against your novel models.

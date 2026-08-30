# Dataset Target Matrix

This matrix independently verifies the actual capabilities and labels of the three public datasets, ensuring no targets are hallucinated or assumed.

| Feature / Metadata | Dataset A (Zenodo 10497437) | Dataset B (PsPM-AOB) | Dataset C (OpenNeuro ds003690) |
| :--- | :--- | :--- | :--- |
| **Number of Subjects** | 16 | 66 | 75 (36 young, 39 older) |
| **Recordings per Subject** | 60-100 tone exposures | Auditory oddball session | Rest, simpleRT, gonogo tasks |
| **Pupil Measurements** | 1D pupil size/shape | 1D pupil size | 1D pupil size (240 Hz) |
| **Raw Eye Video** | Yes (~60 FPS IR video) | No | No |
| **Auditory Stimuli** | Pure tones (constant freq/amp) | Tones (440 Hz standard, 660 Hz deviant) | Auditory cues for RT/Go-NoGo |
| **Experimental Conditions**| Varying luminance | Oddball paradigm | Rest, RT task, Go/NoGo task |
| **Behavioral Labels** | None (passive listening) | Deviant detection | Go/NoGo responses, RT |
| **Reaction-Time Info** | No | Yes (depending on task mode) | Yes |
| **Age Information** | General healthy adult | General healthy adult | Explicitly split (Young vs Older) |
| **EEG Availability** | No | No | Yes |
| **ECG Availability** | No | No | Yes |
| **Listening Effort Label** | **No** (Passive) | **No** (Attention/Surprise) | **No** (Motor Inhibition/RT) |
| **Cognitive Response** | Simple orienting reflex | Oddball effect / Deviancy | Attentional cueing, inhibition |
| **License** | CC-BY-4.0 | CC-BY-4.0 | CC0 |

## Defensible Prediction Targets

Based on the audit, "Listening Effort" or "Speech-in-Noise" targets DO NOT EXIST in these datasets. The defensible prediction targets are:

1.  **Stimulus Presence (Yes/No):** Supported by Dataset A (Tone vs Baseline).
2.  **Auditory Deviance (Standard/Deviant):** Supported by Dataset B.
3.  **Task Type / Cognitive State (Rest vs Go/NoGo):** Supported by Dataset C.
4.  **Age Group (Young vs Older):** Supported by Dataset C.

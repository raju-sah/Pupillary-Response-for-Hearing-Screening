# Dataset Manifest & Integrity Verification

This manifest documents all acquired public datasets, their verified cryptographic hashes, metadata, licensing, and standardized storage locations.

---

## 1. Dataset A: APURE (Audio-Stimulated Pupillometry with Video)
* **Official Title:** APURE - video of pupil response after audio stimulation
* **Permanent DOI:** [10.5281/zenodo.10497437](https://doi.org/10.5281/zenodo.10497437)
* **Associated Paper:** *Pupil Data Upon Stimulation by Auditory Stimuli*, MDPI Data (2024; 9(3):43)
* **License:** Creative Commons Attribution 4.0 International (**CC-BY-4.0**)
* **Archive File:** `APURE- pupil response with video.7z`
* **File Size:** 1,638,062,535 bytes (1.53 GiB / 1.64 GB)
* **Expected MD5:** `d41af6ea8af20013605a73dcfa0e576d`
* **Computed MD5:** `d41af6ea8af20013605a73dcfa0e576d` (**VERIFIED MATCH**)
* **Computed SHA256:** `50226f681d4282733d3984c3e20e6347213cc83cabf5ae615234055aaf45f1de`
* **Raw Storage Path:** `data/raw/dataset_a_zenodo/`
* **Extracted Path:** `data/raw/dataset_a_zenodo/extracted/`
* **Parsed Intermediate Path:** `data/intermediate/dataset_a/`

### Dataset A Properties:
* **Subjects:** 20 healthy adults (10 Females: `1F`-`10F`, 10 Males: `1M`-`10M`).
* **Recordings:** 40 recordings (2 conditions per subject: `audio_stimulation` and `resting_baseline`).
* **Sampling Rate:** ~60 Hz (mean interval ~16.6 ms).
* **Modalities Provided:**
  - Tabular pupil dimensions (Right Eye `_dx.xlsx`, Left Eye `_sx.xlsx`) with explicit blink markers, artifact markers, confidence scores, and audio onset triggers.
  - Raw Infrared (IR) eye-camera videos (`eye_left.mp4`, `eye_right.mp4` @ ~60 FPS).
  - Stimulus metadata (`audio_stimuli.xlsx`: 2000 Hz, 70 dB).

---

## 2. Dataset B: PsPM-AOB (Auditory Oddball Pupillometry)
* **Official Title:** PsPM-AOB: Eye tracker (including pupillometry) measurements from auditory oddball tasks
* **Permanent DOI:** [10.5281/zenodo.3608706](https://doi.org/10.5281/zenodo.3608706)
* **Associated Paper:** Korn & Bach (2016), *Journal of Vision*, 16:28, pp 1–16.
* **License:** Creative Commons Attribution 4.0 International (**CC-BY-4.0**)
* **Archive File:** `Data.zip`
* **File Size:** 201,592,633 bytes (192.25 MiB / 201.59 MB)
* **Expected MD5:** `4df575a0a5f0e035c117ba30435f4c62`
* **Computed MD5:** `4df575a0a5f0e035c117ba30435f4c62` (**VERIFIED MATCH**)
* **Computed SHA256:** `cd3f2fe56215be8c6743ecec7578f38277934037ded5758e1551fb22de5952fe`
* **Raw Storage Path:** `data/raw/dataset_b_pspm_aob/`
* **Extracted Path:** `data/raw/dataset_b_pspm_aob/extracted/Data/`
* **Parsed Intermediate Path:** `data/intermediate/dataset_b/`

### Dataset B Properties:
* **Subjects:** 66 healthy unmedicated participants (40 females, 26 males, aged 24.2 ± 3.9 years).
* **Recordings:** 66 recordings of auditory oddball tasks (inter-trial intervals of 1s, 2s, 3s).
* **Sampling Rate:** 500.0 Hz (Eyelink recording).
* **Modalities Provided:**
  - High-speed binocular pupillometry (`pupil_left`, `pupil_right` in mm).
  - Exact event marker timestamps (Marker 1 = 440 Hz standard tone, Marker 2 = 660 Hz oddball deviant).
  - Binocular gaze coordinates (`gaze_x_l`, `gaze_y_l`, `gaze_x_r`, `gaze_y_r`).
  - Behavioral keypresses, response times, demographics (`AOB_cogent_*.mat`).

---

## 3. Dataset C: OpenNeuro ds003690 (Multimodal Aging & Auditory RT)
* **Official Title:** EEG, ECG and pupil data from young and older adults: rest and auditory cued reaction time tasks
* **Permanent DOI:** [10.18112/openneuro.ds003690.v1.0.0](https://doi.org/10.18112/openneuro.ds003690.v1.0.0)
* **Associated Paper:** Ribeiro & Castelo-Branco (2019), *Neurobiology of Aging*, 73:177-189; (2022), *eLife*, 11:e75722.
* **License:** Creative Commons CC0 (**CC0 Public Domain Dedication**)
* **Target Modalities:** Synchronized Pupillometry (240 Hz, SMI iView X), 64-channel EEG, and ECG during auditory-cued reaction time (Go/NoGo, simpleRT) and resting state across 75 subjects (36 young adults, 39 older adults).

---

## 4. Standardized Data Directory Hierarchy
```
data/
├── raw/
│   ├── dataset_a_zenodo/
│   │   ├── APURE- pupil response with video.7z
│   │   ├── zenodo_metadata.json
│   │   └── extracted/
│   │       ├── 1F/ ... 10M/ (Excel files & raw MP4 videos)
│   │       └── audio_stimuli.xlsx
│   └── dataset_b_pspm_aob/
│       ├── Data.zip
│       ├── README.txt
│       ├── zenodo_metadata.json
│       └── extracted/
│           └── Data/ (AOB_pupil_*.mat, AOB_cogent_*.mat)
├── intermediate/
│   ├── dataset_a/ (Standardized Parquet recordings per subject/condition)
│   └── dataset_b/ (Standardized Parquet recordings per subject)
└── processed/
    └── reports/ (Data summaries, missingness distributions, schema reports)
```

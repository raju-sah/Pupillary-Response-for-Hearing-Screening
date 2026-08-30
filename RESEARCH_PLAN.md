# Research Plan: Deep Learning for Auditory Listening Effort Estimation from Pupillary Responses

## 1. The Core Research Question
**"Can pupillary dynamics provide robust representations of auditory listening effort across subjects and acoustic conditions?"**

This project pivots entirely away from "Objective Hearing Screening" or attempting to diagnose clinical hearing loss. Pupillary responses (Auditory-Evoked Pupillary Responses, or AEPR) reflect arousal, novelty, and cognitive load (listening effort), not raw sensory thresholds. Without clinical ground truth, diagnosing hearing loss from public data is scientifically indefensible.

Instead, this project focuses on **Auditory Listening Effort Estimation**. This is a highly relevant, mathematically rigorous problem that explores how the brain expends cognitive resources to process sound.

## 2. The Research Progression
The project will follow a logical progression, proving understanding at each step before adding complexity:
1.  **Auditory stimulus** → 2. **Pupil response** → 3. **Signal processing** → 4. **Listening / auditory condition** → 5. **ML model** → 6. **Robustness & Explainability**.

## 3. The "No Data Collection" Philosophy
A common misconception is that a strong RA/PhD portfolio requires collecting your own data. In reality, demonstrating the ability to take public, disparate datasets and design a rigorous, cross-dataset evaluation framework is scientifically more defensible and impressive than collecting a small, biased sample of 20 friends in an uncontrolled environment. 

The core contribution of this project is **methodology, rigorous experimental design, and reproducibility**, not data ownership.

## 4. Key Methodological Constraints (What NOT to do)
*   **No Random Train/Test Splitting:** Randomly splitting time-series windows across the dataset leads to massive subject-level leakage. All models will use strict **Leave-One-Subject-Out (LOSO)** or grouped k-fold cross-validation by subject identity.
*   **No "Hearing Loss" Classification:** Labels will be strictly derived from the dataset's actual experimental conditions (e.g., auditory stimulus condition, stimulus intensity, or listening condition).
*   **No Over-engineering:** The architecture will evolve from simple (Logistic Regression) to complex (Temporal Transformers), proving the necessity of complexity at each step.

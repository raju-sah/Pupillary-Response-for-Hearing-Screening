# Research Gaps & Methodological Contributions

Because this project relies entirely on public datasets, the scientific contribution is derived entirely from **methodological rigor, cross-dataset validation, and reproducibility**, rather than data collection.

## Core Contributions

1.  **Benchmarking Deep Temporal Models on AEPR:** 
    Existing literature heavily relies on classical statistical extraction (e.g., measuring the peak dilation window). This project provides a definitive benchmark comparing these classical methods against 1D-CNNs, LSTMs, and Temporal Transformers for physiological classification.
2.  **Quantifying Domain Shift in Pupillometry:**
    It is common for papers to report high accuracy on a single, homogenous dataset. This project explicitly quantifies the performance degradation that occurs when transferring a model trained on one experimental paradigm (passive listening in Dataset A) to another (auditory oddball in Dataset B).
3.  **Age-Robustness Analysis:**
    By leveraging Dataset C, this research explicitly addresses the algorithmic bias introduced by senile miosis, quantifying how models trained on young adults perform on older demographics.
4.  **Reproducibility Framework:**
    Open-sourcing a standardized preprocessing and validation framework for physiological time-series data, completely avoiding common pitfalls like identity leakage through random train/test splitting.

## Publication Potential
This rigorous approach to public data analysis is highly suitable for journals emphasizing medical informatics, biomedical engineering, and computational physiology (e.g., IEEE TBME, JMIR, Computers in Biology and Medicine), where reproducibility and robustness are highly valued over small-scale, proprietary data collection.

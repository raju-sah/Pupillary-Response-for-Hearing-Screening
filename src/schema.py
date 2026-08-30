"""
Schema definition and validation for standardized pupillary data.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "subject_id",
    "recording_id",
    "trial_id",
    "timestamp",
    "pupil_left",
    "pupil_right",
    "stimulus",
    "condition",
]

OPTIONAL_COLUMNS = [
    "pupil_left_valid",
    "pupil_right_valid",
    "eye_tracked",
    "sample_index",
]


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def validate_pupil_dataframe(df: pd.DataFrame) -> ValidationResult:
    """
    Validates that a DataFrame adheres to the standardized schema.
    
    Checks:
    1. Required columns exist.
    2. Data types are appropriate (timestamp numeric, pupil values numeric).
    3. Timestamp monotonicity per recording/trial.
    4. Missing value ratios.
    """
    errors = []
    warnings = []
    summary = {}

    # 1. Column existence
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings, summary=summary)

    # 2. Check types
    is_time_numeric = pd.api.types.is_numeric_dtype(df["timestamp"])
    if not is_time_numeric:
        errors.append("Column 'timestamp' must be numeric")

    for p_col in ["pupil_left", "pupil_right"]:
        if p_col in df.columns and not pd.api.types.is_numeric_dtype(df[p_col]):
            errors.append(f"Column '{p_col}' must be numeric")

    # 3. Check for at least one non-empty pupil signal
    left_all_nan = df["pupil_left"].isna().all()
    right_all_nan = df["pupil_right"].isna().all()
    if left_all_nan and right_all_nan:
        errors.append("Both 'pupil_left' and 'pupil_right' are entirely NaN / empty")
    elif left_all_nan:
        warnings.append("pupil_left is entirely NaN (monocular recording or only right eye available)")
    elif right_all_nan:
        warnings.append("pupil_right is entirely NaN (monocular recording or only left eye available)")

    # 4. Check timestamp monotonicity within (subject_id, recording_id) ONLY if timestamp is numeric
    if is_time_numeric and "subject_id" in df.columns and "recording_id" in df.columns:
        grouped = df.groupby(["subject_id", "recording_id"])
        for (subj, rec), group in grouped:
            diffs = group["timestamp"].diff().dropna()
            if (diffs < 0).any():
                warnings.append(f"Non-monotonic timestamps detected in subject {subj}, recording {rec}")
                break

    # Summary metrics
    n_rows = len(df)
    n_subjects = df["subject_id"].nunique() if "subject_id" in df.columns else 0
    n_recordings = df["recording_id"].nunique() if "recording_id" in df.columns else 0
    n_trials = df["trial_id"].nunique() if "trial_id" in df.columns else 0

    left_missing = df["pupil_left"].isna().mean() if "pupil_left" in df.columns else 1.0
    right_missing = df["pupil_right"].isna().mean() if "pupil_right" in df.columns else 1.0

    summary = {
        "num_rows": n_rows,
        "num_subjects": n_subjects,
        "num_recordings": n_recordings,
        "num_trials": n_trials,
        "left_missing_ratio": float(left_missing),
        "right_missing_ratio": float(right_missing),
        "conditions": list(df["condition"].unique()) if "condition" in df.columns else [],
        "stimulus_values": list(df["stimulus"].unique()) if "stimulus" in df.columns else [],
    }

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, summary=summary)

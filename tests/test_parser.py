"""
Unit tests for pupillometry schema validation and parser logic.
"""

import unittest
import pandas as pd
import numpy as np
from src.schema import validate_pupil_dataframe, REQUIRED_COLUMNS


def create_dummy_valid_dataframe(n_samples: int = 500) -> pd.DataFrame:
    """Helper to generate a structurally valid dummy pupil dataframe."""
    timestamps = np.linspace(0, 5, n_samples)
    pupil_left = 3.5 + 0.2 * np.sin(2 * np.pi * 0.5 * timestamps) + np.random.normal(0, 0.02, n_samples)
    pupil_right = 3.5 + 0.2 * np.sin(2 * np.pi * 0.5 * timestamps) + np.random.normal(0, 0.02, n_samples)
    stimulus = ["none"] * n_samples
    stimulus[100:150] = ["tone_on"] * 50

    return pd.DataFrame({
        "subject_id": "sub-01",
        "recording_id": "rec-01",
        "trial_id": 1,
        "timestamp": timestamps,
        "pupil_left": pupil_left,
        "pupil_right": pupil_right,
        "stimulus": stimulus,
        "condition": "standard_tone",
    })


class TestSchemaValidation(unittest.TestCase):

    def test_schema_valid_dataframe(self):
        """Test that a well-formed DataFrame passes validation."""
        df = create_dummy_valid_dataframe()
        result = validate_pupil_dataframe(df)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.summary["num_rows"], 500)
        self.assertEqual(result.summary["num_subjects"], 1)

    def test_schema_missing_required_column(self):
        """Test that missing required columns triggers validation failure."""
        df = create_dummy_valid_dataframe()
        df_missing = df.drop(columns=["pupil_left"])
        result = validate_pupil_dataframe(df_missing)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("pupil_left" in err for err in result.errors))

    def test_schema_non_numeric_timestamp(self):
        """Test that string timestamps trigger validation failure."""
        df = create_dummy_valid_dataframe()
        df["timestamp"] = [str(x) for x in df["timestamp"]]
        result = validate_pupil_dataframe(df)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("timestamp" in err for err in result.errors))

    def test_schema_both_pupils_nan(self):
        """Test that entirely empty pupil measurements trigger validation failure."""
        df = create_dummy_valid_dataframe()
        df["pupil_left"] = np.nan
        df["pupil_right"] = np.nan
        result = validate_pupil_dataframe(df)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("NaN" in err for err in result.errors))

    def test_schema_non_monotonic_timestamp_warning(self):
        """Test that non-monotonic timestamps trigger validation warning."""
        df = create_dummy_valid_dataframe()
        df["timestamp"] = np.random.permutation(df["timestamp"].values)
        result = validate_pupil_dataframe(df)
        self.assertTrue(any("non-monotonic" in warn.lower() for warn in result.warnings))


if __name__ == "__main__":
    unittest.main()

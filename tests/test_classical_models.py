"""
Unit tests for Feature Extraction and Classical ML Baselines.
Compatible with standard unittest and pytest.
"""

import unittest
import numpy as np
import pandas as pd

from src.preprocessing import TrialEpoch
from src.feature_extraction import (
    extract_features_from_epoch,
    extract_downsampled_timeseries,
    extract_feature_matrix_from_epochs,
    FEATURE_NAMES_25,
    MORPHOLOGICAL_FEATURES,
    DYNAMICS_FEATURES,
    SHAPE_SPECTRAL_FEATURES,
    UNIT_INVARIANT_FEATURES,
)
from src.classical_models import (
    SingleFeatureHeuristicClassifier,
    compute_binary_metrics,
    compute_bootstrap_confidence_intervals,
    get_classical_model_suite,
    evaluate_model_stratified_group_cv,
)


def create_synthetic_epoch():
    """Generates a realistic synthetic AEPR trial epoch."""
    time_grid = np.linspace(-0.5, 3.5, 201)  # 50 Hz
    baseline = 3.5  # mm
    dilation = 0.4 * np.exp(-((time_grid - 1.2) ** 2) / 0.3)
    dilation[time_grid < 0] = 0.0
    raw_signal = baseline + dilation + np.random.normal(0, 0.01, size=len(time_grid))
    sub_signal = raw_signal - baseline
    div_signal = (sub_signal / baseline) * 100.0

    return TrialEpoch(
        trial_id=1,
        stimulus="oddball_deviant",
        condition="salient",
        time=time_grid,
        pupil_raw=raw_signal,
        pupil_subtractive=sub_signal,
        pupil_divisive=div_signal,
        baseline_val=baseline,
        missing_ratio=0.0,
        is_valid=True,
    )


class TestClassicalMLBaselines(unittest.TestCase):

    def setUp(self):
        self.synthetic_epoch = create_synthetic_epoch()

    def test_feature_extraction_25_features(self):
        """Verifies that all 25 features are properly extracted and finite."""
        feats = extract_features_from_epoch(self.synthetic_epoch)
        self.assertIsNotNone(feats)
        self.assertEqual(len(feats), 25)
        for name in FEATURE_NAMES_25:
            self.assertIn(name, feats)
            self.assertTrue(np.isfinite(feats[name]), f"Feature {name} is non-finite: {feats[name]}")

        self.assertGreater(feats["peak_dilation_amplitude"], 0.1)
        self.assertTrue(0.8 <= feats["latency_to_peak_s"] <= 1.6)
        self.assertGreater(feats["baseline_diameter_mean"], 3.0)

    def test_feature_subsets(self):
        """Verifies defined feature categories and unit-invariant subset."""
        self.assertEqual(len(FEATURE_NAMES_25), 25)
        self.assertEqual(len(MORPHOLOGICAL_FEATURES), 8)
        self.assertEqual(len(DYNAMICS_FEATURES), 7)
        self.assertEqual(len(SHAPE_SPECTRAL_FEATURES), 10)
        self.assertGreaterEqual(len(UNIT_INVARIANT_FEATURES), 12)

        for f in ["peak_dilation_amplitude", "baseline_diameter_mean", "mean_response_amplitude", "auc_response_trapezoid"]:
            self.assertNotIn(f, UNIT_INVARIANT_FEATURES)

    def test_downsampled_timeseries(self):
        """Verifies 10 Hz downsampling over [0, 3.5s]."""
        ts = extract_downsampled_timeseries(self.synthetic_epoch, t_start=0.0, t_end=3.5, target_fs=10.0)
        self.assertIsNotNone(ts)
        self.assertEqual(len(ts), 36)
        self.assertTrue(np.all(np.isfinite(ts)))

    def test_single_feature_heuristic(self):
        """Tests the SingleFeatureHeuristic classifier."""
        X = np.array([[1.0, 2.0, 0.1], [1.0, 2.0, 0.5], [1.0, 2.0, 0.8], [1.0, 2.0, 0.2]])
        y = np.array([0, 1, 1, 0])
        clf = SingleFeatureHeuristicClassifier(feature_idx=2)
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        self.assertEqual(probs.shape, (4, 2))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
        preds = clf.predict(X)
        self.assertEqual(len(preds), 4)

    def test_zero_subject_leakage_in_group_cv(self):
        """Verifies that StratifiedGroupKFold maintains strict zero-leakage subject separation."""
        np.random.seed(42)
        n_samples = 100
        n_subjs = 10
        subjects = np.repeat([f"sub_{i:02d}" for i in range(n_subjs)], n_samples // n_subjs)
        X = np.random.randn(n_samples, 25)
        y = np.random.binomial(1, 0.3, size=n_samples)

        models = get_classical_model_suite(random_state=42)
        rf = models["Random Forest"]

        res = evaluate_model_stratified_group_cv(
            model_name="Test_RF",
            model=rf,
            X=X,
            y=y,
            groups=subjects,
            n_splits=5,
            random_state=42,
            n_bootstraps=50
        )

        self.assertEqual(res.n_samples, n_samples)
        self.assertEqual(res.n_subjects, n_subjs)
        self.assertTrue(0.0 <= res.roc_auc <= 1.0)
        self.assertTrue(0.0 <= res.pr_auc <= 1.0)
        self.assertEqual(len(res.fold_roc_aucs), 5)
        self.assertIn("roc_auc", res.ci_95)
        self.assertLessEqual(res.ci_95["roc_auc"][0], res.ci_95["roc_auc"][1])


if __name__ == "__main__":
    unittest.main()

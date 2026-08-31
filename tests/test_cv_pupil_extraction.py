"""
Unit tests for Computer Vision Pupil Extraction module (src/cv_pupil_extraction.py).
"""

import unittest
import numpy as np
import cv2
from src.cv_pupil_extraction import (
    extract_pupil_from_frame,
    compute_concordance_metrics,
)


class TestCVPupilExtraction(unittest.TestCase):
    def setUp(self):
        # Create a synthetic eye image: bright sclera/iris with a dark circular pupil in the center
        self.H, self.W = 200, 200
        self.frame = np.ones((self.H, self.W), dtype=np.uint8) * 180  # Bright background
        # Draw dark pupil circle of radius 25 at center (100, 100)
        cv2.circle(self.frame, (100, 100), 25, color=20, thickness=-1)

    def test_extract_pupil_from_synthetic_frame(self):
        res = extract_pupil_from_frame(self.frame, dark_threshold_pct=25.0)
        self.assertFalse(res["is_blink"])
        self.assertAlmostEqual(res["center_x"], 100.0, delta=2.0)
        self.assertAlmostEqual(res["center_y"], 100.0, delta=2.0)
        # Expected diameter = 2 * radius = 50 px
        self.assertAlmostEqual(res["equivalent_diameter"], 50.0, delta=3.0)
        self.assertTrue(res["confidence"] > 0.80)

    def test_extract_pupil_blink_frame(self):
        # Frame with zero variation (complete occlusion/closed eyelid)
        blink_frame = np.ones((self.H, self.W), dtype=np.uint8) * 180
        res = extract_pupil_from_frame(blink_frame, dark_threshold_pct=10.0)
        self.assertTrue(res["is_blink"])
        self.assertTrue(np.isnan(res["equivalent_diameter"]))

    def test_compute_concordance_metrics(self):
        # Perfect correlation test
        x = np.linspace(30.0, 50.0, 100)
        y = x + np.random.normal(0, 0.5, 100)
        metrics = compute_concordance_metrics(x, y)
        self.assertTrue(metrics["pearson_r"] > 0.95)
        self.assertTrue(metrics["spearman_rho"] > 0.95)
        self.assertAlmostEqual(metrics["bland_altman_bias"], 0.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()

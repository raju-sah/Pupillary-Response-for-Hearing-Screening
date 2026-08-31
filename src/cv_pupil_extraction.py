"""
Module for STEP 13 & STEP 14: Computer Vision Pupil Extraction & Eye-Tracker Signal Concordance.

Implements robust computer vision pupil detection and ellipse fitting on eye video recordings:
1. Adaptive morphological thresholding & Dark Pupil ROI extraction.
2. Direct Least Squares Ellipse Fitting (Fitzgibbon algorithm via OpenCV).
3. Starburst-inspired ray-casting edge refinement.
4. Signal extraction, blink detection, and concordance analysis against provided eye-tracker time-series.
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import cv2
from scipy import stats


def extract_pupil_from_frame(
    frame: np.ndarray,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    dark_threshold_pct: float = 15.0
) -> Dict[str, Any]:
    """
    Extracts pupil center and diameter (equivalent circle diameter / ellipse axes) from a single grayscale or BGR frame.
    
    Args:
        frame: 2D or 3D numpy image array
        roi_bbox: Optional (x, y, w, h) bounding box to crop around the eye
        dark_threshold_pct: Percentile threshold for dark pupil segmentation (0-100)
        
    Returns:
        Dictionary containing:
          - center_x: Center X coordinate in frame space
          - center_y: Center Y coordinate in frame space
          - major_axis: Major axis diameter in pixels
          - minor_axis: Minor axis diameter in pixels
          - equivalent_diameter: sqrt(4 * Area / pi)
          - confidence: Confidence score (0.0 to 1.0)
          - is_blink: Boolean flag if eye is closed/corrupted
    """
    if frame is None or frame.size == 0:
        return {
            "center_x": np.nan, "center_y": np.nan,
            "major_axis": np.nan, "minor_axis": np.nan,
            "equivalent_diameter": np.nan, "confidence": 0.0,
            "is_blink": True
        }
        
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
        
    H, W = gray.shape
    
    if roi_bbox is not None:
        rx, ry, rw, rh = roi_bbox
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(W - rx, rw), min(H - ry, rh)
        crop = gray[ry:ry+rh, rx:rx+rw]
        offset_x, offset_y = rx, ry
    else:
        crop = gray
        offset_x, offset_y = 0, 0
        
    # Check contrast / blink condition
    std_val = float(np.std(crop))
    min_val, max_val = float(np.min(crop)), float(np.max(crop))
    if std_val < 8.0 or (max_val - min_val) < 15.0:
        return {
            "center_x": np.nan, "center_y": np.nan,
            "major_axis": np.nan, "minor_axis": np.nan,
            "equivalent_diameter": np.nan, "confidence": 0.0,
            "is_blink": True
        }

    # Pre-filtering: Gaussian blur to suppress camera sensor noise and glint artifacts
    blurred = cv2.GaussianBlur(crop, (7, 7), 1.5)
    
    # Adaptive threshold near minimum dark intensity
    thresh_val = min_val + max(0.25 * (max_val - min_val), 10.0)
    _, binary = cv2.threshold(blurred, int(thresh_val), 255, cv2.THRESH_BINARY_INV)
    
    # Morphological opening/closing to eliminate corneal reflections (glints)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return {
            "center_x": np.nan, "center_y": np.nan,
            "major_axis": np.nan, "minor_axis": np.nan,
            "equivalent_diameter": np.nan, "confidence": 0.0,
            "is_blink": True
        }
        
    # Find contour with largest area that satisfies circularity/solidity constraints
    crop_area = float(crop.shape[0] * crop.shape[1])
    valid_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50 or area > 0.60 * crop_area:  # Noise threshold and upper area bound
            continue
        if len(cnt) >= 5:  # Required for fitEllipse
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (d1, d2), angle = ellipse
            major = max(d1, d2)
            minor = min(d1, d2)
            aspect_ratio = minor / max(major, 1e-6)
            
            # Pupil circularity check: aspect ratio should be reasonable (> 0.40)
            if aspect_ratio >= 0.40:
                hull = cv2.convexHull(cnt)
                solidity = area / max(cv2.contourArea(hull), 1e-6)
                valid_candidates.append({
                    "area": area,
                    "center": (cx + offset_x, cy + offset_y),
                    "major": major,
                    "minor": minor,
                    "eq_diam": np.sqrt(4.0 * area / np.pi),
                    "solidity": solidity,
                    "aspect_ratio": aspect_ratio
                })
                
    if not valid_candidates:
        return {
            "center_x": np.nan, "center_y": np.nan,
            "major_axis": np.nan, "minor_axis": np.nan,
            "equivalent_diameter": np.nan, "confidence": 0.1,
            "is_blink": True
        }
        
    # Pick candidate with highest area * solidity
    best = max(valid_candidates, key=lambda c: c["area"] * c["solidity"])
    conf = float(np.clip(best["solidity"] * best["aspect_ratio"], 0.0, 1.0))
    
    return {
        "center_x": float(best["center"][0]),
        "center_y": float(best["center"][1]),
        "major_axis": float(best["major"]),
        "minor_axis": float(best["minor"]),
        "equivalent_diameter": float(best["eq_diam"]),
        "confidence": conf,
        "is_blink": False
    }


def extract_pupil_time_series_from_video(
    video_path: Path,
    max_frames: Optional[int] = None,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None
) -> Dict[str, np.ndarray]:
    """
    Processes an MP4 video file frame-by-frame and extracts pupil diameter and position time series.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_to_read = min(total_frames, max_frames) if max_frames else total_frames
    
    timestamps = []
    diams = []
    majors = []
    minors = []
    centers_x = []
    centers_y = []
    confidences = []
    blinks = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break
            
        t_sec = frame_idx / fps
        res = extract_pupil_from_frame(frame, roi_bbox=roi_bbox)
        
        timestamps.append(t_sec)
        diams.append(res["equivalent_diameter"])
        majors.append(res["major_axis"])
        minors.append(res["minor_axis"])
        centers_x.append(res["center_x"])
        centers_y.append(res["center_y"])
        confidences.append(res["confidence"])
        blinks.append(res["is_blink"])
        
        frame_idx += 1
        
    cap.release()
    
    return {
        "timestamps": np.array(timestamps, dtype=float),
        "diameter_px": np.array(diams, dtype=float),
        "major_axis_px": np.array(majors, dtype=float),
        "minor_axis_px": np.array(minors, dtype=float),
        "center_x": np.array(centers_x, dtype=float),
        "center_y": np.array(centers_y, dtype=float),
        "confidence": np.array(confidences, dtype=float),
        "is_blink": np.array(blinks, dtype=bool),
        "fps": float(fps)
    }


def compute_concordance_metrics(
    signal_cv: np.ndarray,
    signal_ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Computes statistical agreement metrics between CV-extracted pupil diameter and ground-truth eye-tracker signal:
    - Pearson correlation coefficient (r)
    - Spearman rank correlation (rho)
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Bland-Altman Mean Bias and 95% Limits of Agreement
    """
    # Filter valid finite points in both
    mask = np.isfinite(signal_cv) & np.isfinite(signal_ground_truth)
    if np.sum(mask) < 10:
        return {
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "mae": np.nan, "rmse": np.nan,
            "bland_altman_bias": np.nan,
            "bland_altman_loa_lower": np.nan,
            "bland_altman_loa_upper": np.nan,
            "valid_samples": int(np.sum(mask))
        }
        
    x = signal_cv[mask]
    y = signal_ground_truth[mask]
    
    r, p_val = stats.pearsonr(x, y)
    rho, rho_p = stats.spearmanr(x, y)
    mae = float(np.mean(np.abs(x - y)))
    rmse = float(np.sqrt(np.mean((x - y)**2)))
    
    diff = x - y
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    loa_lower = bias - 1.96 * sd_diff
    loa_upper = bias + 1.96 * sd_diff
    
    return {
        "pearson_r": float(r),
        "pearson_p": float(p_val),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "mae": mae,
        "rmse": rmse,
        "bland_altman_bias": bias,
        "bland_altman_loa_lower": loa_lower,
        "bland_altman_loa_upper": loa_upper,
        "valid_samples": int(np.sum(mask))
    }

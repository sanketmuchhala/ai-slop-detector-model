"""Threshold calibration: find conservative and balanced operating thresholds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from detector.config import CalibrationConfig
from detector.eval.metrics import compute_roc_curve, fpr_at_tpr, threshold_at_fpr, threshold_at_tpr, tpr_at_fpr


def find_conservative_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    max_fpr: float = 0.01,
) -> dict:
    """Find threshold where FPR <= max_fpr on human examples, maximizing TPR.

    Returns: {threshold, max_fpr_target, measured_fpr_human, tpr}
    """
    fpr_arr, tpr_arr, thresholds = compute_roc_curve(y_true, y_scores)
    threshold = threshold_at_fpr(fpr_arr, thresholds, max_fpr)
    achieved_tpr = tpr_at_fpr(fpr_arr, tpr_arr, max_fpr)

    # Compute actual measured FPR at this threshold
    human_mask = y_true == 0
    if human_mask.sum() > 0:
        measured_fpr = float((y_scores[human_mask] >= threshold).mean())
    else:
        measured_fpr = 0.0

    return {
        "threshold": round(threshold, 6),
        "max_fpr_target": max_fpr,
        "measured_fpr_human": round(measured_fpr, 6),
        "tpr": round(achieved_tpr, 6),
    }


def find_balanced_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    target_tpr: float = 0.90,
) -> dict:
    """Find threshold where TPR >= target_tpr, reporting corresponding FPR.

    Returns: {threshold, target_tpr, measured_fpr, tpr}
    """
    fpr_arr, tpr_arr, thresholds = compute_roc_curve(y_true, y_scores)
    threshold = threshold_at_tpr(tpr_arr, thresholds, target_tpr)
    achieved_fpr = fpr_at_tpr(fpr_arr, tpr_arr, target_tpr)

    return {
        "threshold": round(threshold, 6),
        "target_tpr": target_tpr,
        "measured_fpr": round(achieved_fpr, 6),
        "tpr": round(target_tpr, 6),
    }


def calibrate(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    calibration_config: CalibrationConfig,
    calibration_set_name: str = "validation",
) -> dict:
    """Run full calibration: compute both conservative and balanced thresholds."""
    conservative = find_conservative_threshold(y_true, y_scores, calibration_config.conservative_max_fpr)
    balanced = find_balanced_threshold(y_true, y_scores, calibration_config.balanced_target_tpr)
    return {
        "conservative": conservative,
        "balanced": balanced,
        "calibration_set": calibration_set_name,
        "n_samples": len(y_true),
        "n_human": int((y_true == 0).sum()),
        "n_ai": int((y_true == 1).sum()),
    }


def save_thresholds(thresholds: dict, path: Path) -> None:
    """Save thresholds.json to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)


def load_thresholds(path: Path) -> dict:
    """Load thresholds.json from disk."""
    with open(path) as f:
        return json.load(f)

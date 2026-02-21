"""Tests for threshold calibration."""

import numpy as np
import pytest

from detector.calibrate import calibrate, find_balanced_threshold, find_conservative_threshold, load_thresholds, save_thresholds
from detector.config import CalibrationConfig


@pytest.fixture
def well_separated_scores():
    """Scores where human and AI are well separated."""
    rng = np.random.RandomState(42)
    human_scores = rng.beta(2, 8, size=500)  # mostly low
    ai_scores = rng.beta(8, 2, size=500)  # mostly high
    y_true = np.concatenate([np.zeros(500), np.ones(500)])
    y_scores = np.concatenate([human_scores, ai_scores])
    return y_true, y_scores


def test_conservative_fpr_constraint(well_separated_scores):
    y_true, y_scores = well_separated_scores
    result = find_conservative_threshold(y_true, y_scores, max_fpr=0.01)
    assert result["measured_fpr_human"] <= 0.01
    assert result["max_fpr_target"] == 0.01
    assert 0.0 <= result["threshold"] <= 1.0


def test_balanced_tpr_constraint(well_separated_scores):
    y_true, y_scores = well_separated_scores
    result = find_balanced_threshold(y_true, y_scores, target_tpr=0.90)
    assert result["target_tpr"] == 0.90
    assert 0.0 <= result["threshold"] <= 1.0
    assert 0.0 <= result["measured_fpr"] <= 1.0


def test_calibrate_returns_both(well_separated_scores):
    y_true, y_scores = well_separated_scores
    cal_config = CalibrationConfig(conservative_max_fpr=0.01, balanced_target_tpr=0.90)
    result = calibrate(y_true, y_scores, cal_config, "validation")
    assert "conservative" in result
    assert "balanced" in result
    assert result["calibration_set"] == "validation"
    assert result["n_samples"] == 1000
    assert result["n_human"] == 500
    assert result["n_ai"] == 500


def test_save_and_load_thresholds(well_separated_scores, tmp_path):
    y_true, y_scores = well_separated_scores
    cal_config = CalibrationConfig()
    result = calibrate(y_true, y_scores, cal_config)
    path = tmp_path / "thresholds.json"
    save_thresholds(result, path)
    loaded = load_thresholds(path)
    assert loaded["conservative"]["threshold"] == result["conservative"]["threshold"]
    assert loaded["balanced"]["threshold"] == result["balanced"]["threshold"]


def test_calibrate_metadata_fields(well_separated_scores):
    y_true, y_scores = well_separated_scores
    cal_config = CalibrationConfig()
    result = calibrate(y_true, y_scores, cal_config, "my_split")
    assert "n_samples" in result
    assert "n_human" in result
    assert "n_ai" in result
    assert "calibration_set" in result
    assert result["conservative"]["max_fpr_target"] == 0.01
    assert "measured_fpr_human" in result["conservative"]

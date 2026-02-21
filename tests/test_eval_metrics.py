"""Tests for eval metrics."""

import numpy as np
import pytest

from detector.eval.metrics import (
    compute_auc,
    compute_confusion,
    compute_ece,
    compute_metrics_at_thresholds,
    compute_roc_curve,
    fpr_at_tpr,
    plot_roc_curve,
    save_roc_csv,
    tpr_at_fpr,
)


def test_compute_roc_curve_shape(sample_scores):
    y_true, y_scores = sample_scores
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_scores)
    assert len(fpr) == len(tpr)
    assert len(fpr) > 0
    assert fpr[0] == 0.0
    assert tpr[-1] == 1.0


def test_compute_auc_perfect():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert compute_auc(y_true, y_scores) == 1.0


def test_compute_auc_random():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=1000)
    y_scores = rng.rand(1000)
    auc = compute_auc(y_true, y_scores)
    assert 0.3 <= auc <= 0.7  # should be near 0.5


def test_tpr_at_fpr_known(sample_scores):
    y_true, y_scores = sample_scores
    fpr, tpr, _ = compute_roc_curve(y_true, y_scores)
    result = tpr_at_fpr(fpr, tpr, 0.05)
    assert 0.0 <= result <= 1.0


def test_fpr_at_tpr_known(sample_scores):
    y_true, y_scores = sample_scores
    fpr, tpr, _ = compute_roc_curve(y_true, y_scores)
    result = fpr_at_tpr(fpr, tpr, 0.90)
    assert 0.0 <= result <= 1.0


def test_tpr_at_fpr_zero():
    fpr = np.array([0.0, 0.5, 1.0])
    tpr = np.array([0.0, 0.8, 1.0])
    assert tpr_at_fpr(fpr, tpr, 0.0) == 0.0


def test_fpr_at_tpr_unreachable():
    fpr = np.array([0.0, 0.5])
    tpr = np.array([0.0, 0.5])
    assert fpr_at_tpr(fpr, tpr, 0.99) == 1.0


def test_compute_confusion():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    cm = compute_confusion(y_true, y_pred)
    assert cm["tp"] == 1
    assert cm["fp"] == 1
    assert cm["tn"] == 1
    assert cm["fn"] == 1
    assert cm["tpr"] == 0.5
    assert cm["fpr"] == 0.5


def test_compute_ece_range(sample_scores):
    y_true, y_scores = sample_scores
    ece = compute_ece(y_true, y_scores)
    assert 0.0 <= ece <= 1.0


def test_compute_ece_empty():
    ece = compute_ece(np.array([]), np.array([]))
    assert ece == 0.0


def test_compute_metrics_at_thresholds(sample_scores):
    y_true, y_scores = sample_scores
    result = compute_metrics_at_thresholds(y_true, y_scores, fpr_targets=[0.01, 0.05], tpr_targets=[0.90])
    assert "auc" in result
    assert "ece" in result
    assert "tpr_at_fpr" in result
    assert "fpr_at_tpr" in result
    assert "0.01" in result["tpr_at_fpr"]
    assert "0.9" in result["fpr_at_tpr"]


def test_plot_roc_curve_saves_file(sample_scores, tmp_path):
    y_true, y_scores = sample_scores
    fpr, tpr, _ = compute_roc_curve(y_true, y_scores)
    auc = compute_auc(y_true, y_scores)
    save_path = tmp_path / "roc.png"
    plot_roc_curve(fpr, tpr, auc, conservative_point=(0.01, 0.5), balanced_point=(0.1, 0.9), save_path=save_path)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_save_roc_csv(sample_scores, tmp_path):
    y_true, y_scores = sample_scores
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_scores)
    save_path = tmp_path / "roc.csv"
    save_roc_csv(fpr, tpr, thresholds, save_path)
    assert save_path.exists()
    lines = save_path.read_text().strip().split("\n")
    assert lines[0] == "threshold,fpr,tpr"
    assert len(lines) > 1

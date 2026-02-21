"""Tests for training logic."""

import random

import numpy as np
import pytest
import torch

from detector.config import Config, load_config
from detector.train import _compute_ece, build_compute_metrics, build_training_args, set_deterministic


def test_set_deterministic():
    set_deterministic(42)
    a = random.random()
    b = np.random.rand()
    c = torch.rand(1).item()

    set_deterministic(42)
    assert random.random() == a
    assert np.random.rand() == b
    assert torch.rand(1).item() == c


def test_build_training_args(tmp_path):
    config = Config(training={"learning_rate": 3e-5, "num_epochs": 2})
    args = build_training_args(config, tmp_path)
    assert args.learning_rate == 3e-5
    assert args.num_train_epochs == 2
    assert args.seed == 42


def test_compute_metrics_fn():
    compute_metrics = build_compute_metrics()
    logits = np.array([[2.0, -1.0], [-1.0, 2.0], [1.0, -0.5], [-0.5, 1.0]])
    labels = np.array([0, 1, 0, 1])
    result = compute_metrics((logits, labels))
    assert "accuracy" in result
    assert "f1" in result
    assert "auc" in result
    assert "ece" in result
    assert 0.0 <= result["auc"] <= 1.0
    assert 0.0 <= result["ece"] <= 1.0


def test_compute_ece_perfect():
    """Perfect calibration should have low ECE."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    ece = _compute_ece(y_true, y_prob, n_bins=10)
    assert 0.0 <= ece <= 0.2


def test_compute_ece_range():
    """ECE should be between 0 and 1."""
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=100)
    y_prob = rng.rand(100)
    ece = _compute_ece(y_true, y_prob, n_bins=15)
    assert 0.0 <= ece <= 1.0

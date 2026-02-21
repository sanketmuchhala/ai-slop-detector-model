"""Shared test fixtures."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_scores():
    """Synthetic scores for metric testing: 100 human (label=0), 100 AI (label=1)."""
    rng = np.random.RandomState(42)
    human_scores = rng.beta(2, 5, size=100)  # skewed toward 0
    ai_scores = rng.beta(5, 2, size=100)  # skewed toward 1
    y_true = np.concatenate([np.zeros(100), np.ones(100)])
    y_scores = np.concatenate([human_scores, ai_scores])
    return y_true, y_scores


@pytest.fixture
def sample_config_dict():
    """Minimal valid config dict."""
    return {
        "seed": 42,
        "run_name": "test-run",
        "model": {"name": "bert-base-cased", "num_labels": 2, "use_lora": False},
        "data": {"dataset": "wiki_human_ai", "max_length": 128, "val_size": 0.1, "test_size": 0.1},
        "training": {"learning_rate": 2e-5, "num_epochs": 1, "per_device_train_batch_size": 2},
    }


@pytest.fixture
def mock_model():
    """Mock model that returns random logits."""

    class FakeOutput:
        def __init__(self, logits):
            self.logits = logits

    model = MagicMock()

    def forward(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        logits = torch.randn(batch_size, 2)
        return FakeOutput(logits)

    model.side_effect = forward
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.device = torch.device("cpu")
    return model


@pytest.fixture
def tmp_run_dir(tmp_path):
    """Temporary run directory."""
    run_dir = tmp_path / "test-run"
    run_dir.mkdir()
    return run_dir

"""Tests for model export."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from detector.config import Config
from detector.export import generate_model_card


def test_generate_model_card(tmp_path):
    config = Config()
    metrics = {"wiki": {"auc": 0.95, "ece": 0.03}}
    thresholds = {
        "conservative": {"threshold": 0.8, "measured_fpr_human": 0.009},
        "balanced": {"threshold": 0.5, "measured_fpr": 0.05},
    }
    path = tmp_path / "MODEL_CARD.md"
    generate_model_card(config, metrics, thresholds, path)
    assert path.exists()
    content = path.read_text()
    assert "bert-base-cased" in content
    assert "0.95" in content
    assert "Conservative" in content or "conservative" in content


def test_generate_model_card_no_metrics(tmp_path):
    config = Config()
    path = tmp_path / "MODEL_CARD.md"
    generate_model_card(config, None, None, path)
    assert path.exists()
    content = path.read_text()
    assert "N/A" in content


def test_generate_model_card_high_ece_warning(tmp_path):
    config = Config()
    metrics = {"wiki": {"auc": 0.90, "ece": 0.08}}
    path = tmp_path / "MODEL_CARD.md"
    generate_model_card(config, metrics, None, path)
    content = path.read_text()
    assert "ECE" in content

"""Integration tests for the full pipeline. Marked slow - require model downloads."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from detector.calibrate import calibrate, save_thresholds
from detector.config import CalibrationConfig, Config, load_config
from detector.eval.metrics import compute_metrics_at_thresholds, compute_roc_curve, plot_roc_curve, save_roc_csv
from detector.infer import TextDetector


def _make_fake_wiki_dataset(n: int = 40):
    """Create a small fake dataset for testing."""
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "page_title": [f"Page_{i}" for i in range(n)],
                    "human_text": [f"Human authored text about topic number {i} with some detail" for i in range(n)],
                    "ai_text": [
                        f"AI generated comprehensive text about topic number {i} covering multiple aspects"
                        for i in range(n)
                    ],
                    "split": ["train"] * n,
                }
            )
        }
    )


@pytest.mark.slow
def test_smoke_eval_calibrate_pipeline(tmp_path):
    """Test the eval -> calibrate pipeline with synthetic data."""
    # Create synthetic scores
    rng = np.random.RandomState(42)
    n = 200
    y_true = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    human_scores = rng.beta(2, 8, size=n // 2)
    ai_scores = rng.beta(8, 2, size=n // 2)
    y_scores = np.concatenate([human_scores, ai_scores])

    run_dir = tmp_path / "test-run"
    run_dir.mkdir()

    # Compute metrics
    metrics = compute_metrics_at_thresholds(y_true, y_scores, fpr_targets=[0.01, 0.05], tpr_targets=[0.90])
    assert metrics["auc"] > 0.8

    # Save ROC artifacts
    fpr_arr, tpr_arr, thresholds = compute_roc_curve(y_true, y_scores)
    save_roc_csv(fpr_arr, tpr_arr, thresholds, run_dir / "roc.csv")
    plot_roc_curve(fpr_arr, tpr_arr, metrics["auc"], save_path=run_dir / "roc.png")

    assert (run_dir / "roc.csv").exists()
    assert (run_dir / "roc.png").exists()

    # Calibrate
    cal_config = CalibrationConfig(conservative_max_fpr=0.01, balanced_target_tpr=0.90)
    cal_result = calibrate(y_true, y_scores, cal_config, "validation")
    save_thresholds(cal_result, run_dir / "thresholds.json")

    assert (run_dir / "thresholds.json").exists()
    assert cal_result["conservative"]["measured_fpr_human"] <= 0.01

    # Save metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"wiki": metrics}, f, indent=2)

    # Verify all artifacts
    assert (run_dir / "metrics.json").exists()
    loaded = json.loads((run_dir / "metrics.json").read_text())
    assert "wiki" in loaded
    assert "auc" in loaded["wiki"]


@pytest.mark.slow
def test_infer_normalize_and_confidence():
    """Test inference normalization and confidence computation end-to-end."""
    # Test normalization pipeline
    detector_cls = TextDetector

    text = "  Hello   world\n\n\ntest  "
    normalized = detector_cls.normalize_text(text)
    assert normalized == "hello world\ntest" or normalized == "Hello world\ntest"

    # Test confidence formula at various points
    for score_ai in [0.0, 0.1, 0.5, 0.9, 1.0]:
        for threshold in [0.1, 0.5, 0.9]:
            conf = detector_cls._get_confidence(score_ai, threshold)
            assert 0.0 <= conf <= 1.0

    # Specific known values
    assert abs(detector_cls._get_confidence(0.8, 0.7) - 0.1 / 0.7) < 1e-6
    assert abs(detector_cls._get_confidence(0.3, 0.7) - 0.4 / 0.7) < 1e-6


@pytest.mark.slow
def test_config_all_yamls_valid():
    """All config YAML files should load without error."""
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    for yaml_file in configs_dir.glob("*.yaml"):
        config = load_config(yaml_file)
        assert config.seed == 42
        assert config.model.name == "bert-base-cased"

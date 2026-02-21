"""Tests for config loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from detector.config import Config, apply_overrides, load_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_load_baseline_config():
    config = load_config(CONFIGS_DIR / "baseline.yaml")
    assert config.model.name == "bert-base-cased"
    assert config.model.use_lora is False
    assert config.model.lora is None
    assert config.training.learning_rate == 2e-5
    assert config.data.max_length == 512
    assert config.seed == 42


def test_load_lora_config():
    config = load_config(CONFIGS_DIR / "lora.yaml")
    assert config.model.use_lora is True
    assert config.model.lora is not None
    assert config.model.lora.r == 16
    assert config.model.lora.lora_alpha == 16
    assert config.model.lora.target_modules == "all-linear"
    assert config.training.learning_rate == 5e-5


def test_load_smoke_config():
    config = load_config(CONFIGS_DIR / "smoke.yaml")
    assert config.data.max_train_samples == 200
    assert config.data.max_eval_samples == 100
    assert config.training.num_epochs == 1
    assert config.data.max_length == 128


def test_lora_required_when_enabled():
    with pytest.raises(ValidationError):
        Config(model={"name": "bert-base-cased", "use_lora": True})


def test_invalid_dataset_name():
    with pytest.raises(ValidationError):
        Config(data={"dataset": "invalid_name"})


def test_default_values():
    config = Config()
    assert config.seed == 42
    assert config.model.name == "bert-base-cased"
    assert config.data.dataset == "wiki_human_ai"
    assert config.calibration.conservative_max_fpr == 0.01


def test_config_serialization_roundtrip():
    config = load_config(CONFIGS_DIR / "baseline.yaml")
    d = config.model_dump()
    config2 = Config(**d)
    assert config == config2


def test_override_nested_value():
    raw = {"seed": 42, "training": {"learning_rate": 2e-5}}
    raw = apply_overrides(raw, ["training.learning_rate=1e-4"])
    assert raw["training"]["learning_rate"] == 1e-4


def test_override_creates_missing_keys():
    raw = {"seed": 42}
    raw = apply_overrides(raw, ["training.learning_rate=1e-4"])
    assert raw["training"]["learning_rate"] == 1e-4


def test_override_invalid_format():
    with pytest.raises(ValueError, match="Invalid override format"):
        apply_overrides({}, ["no_equals_sign"])

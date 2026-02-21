"""Configuration loading and validation via Pydantic v2."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


class LoraConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    bias: str = "none"
    use_rslora: bool = True
    modules_to_save: list[str] = ["classifier"]


class ModelConfig(BaseModel):
    name: str = "bert-base-cased"
    num_labels: int = 2
    use_lora: bool = False
    lora: LoraConfig | None = None

    @model_validator(mode="after")
    def validate_lora(self):
        if self.use_lora and self.lora is None:
            raise ValueError("lora config required when use_lora=True")
        return self


class DataConfig(BaseModel):
    dataset: Literal["wiki_human_ai", "raid", "both"] = "wiki_human_ai"
    train_on_raid: bool = False
    raid_include_adversarial: bool = False
    max_length: int = 512
    val_size: float = 0.1
    test_size: float = 0.1
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


class TrainingConfig(BaseModel):
    learning_rate: float = 2e-5
    num_epochs: int = 5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    save_strategy: str = "epoch"
    save_steps: int | None = None
    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    metric_for_best_model: str = "eval_auc"
    greater_is_better: bool = True
    load_best_model_at_end: bool = True
    save_total_limit: int = 2
    logging_steps: int = 50
    dataloader_num_workers: int = 4


class EvalConfig(BaseModel):
    raid_split: Literal["train", "extra"] = "train"
    raid_include_adversarial: bool = True
    fpr_targets: list[float] = [0.001, 0.005, 0.01, 0.02, 0.05]
    tpr_targets: list[float] = [0.80, 0.90, 0.95]
    threshold_sweep_steps: int = 1000
    max_eval_samples: int | None = None


class CalibrationConfig(BaseModel):
    conservative_max_fpr: float = 0.01
    balanced_target_tpr: float = 0.90
    calibration_corpus: str | None = None  # future hook for custom corpus


class ExportConfig(BaseModel):
    onnx: bool = True
    onnx_opset: int = 14
    model_version: str = "v0.1.0"


class OutputConfig(BaseModel):
    runs_dir: str = "runs"


class Config(BaseModel):
    seed: int = 42
    run_name: str = "default"
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    eval: EvalConfig = EvalConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    export: ExportConfig = ExportConfig()
    output: OutputConfig = OutputConfig()


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def apply_overrides(config_dict: dict, overrides: list[str]) -> dict:
    """Apply dot-notation overrides to a config dict.

    Example: training.learning_rate=1e-4 sets config_dict["training"]["learning_rate"] = 1e-4
    """
    for override in overrides:
        key, _, value = override.partition("=")
        if not value:
            raise ValueError(f"Invalid override format: {override}. Expected key=value")

        parts = key.split(".")
        target = config_dict
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]

        # Try to parse value as yaml for type coercion (int, float, bool, null)
        parsed = yaml.safe_load(value)
        # yaml.safe_load doesn't parse scientific notation like 1e-4 as float
        if isinstance(parsed, str):
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        target[parts[-1]] = parsed

    return config_dict

"""Training logic for the AI text detector."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from detector.config import Config
from detector.data.wiki_human_ai import prepare_wiki_dataset, tokenize_dataset


def set_deterministic(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(config: Config) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base model and tokenizer. Apply LoRA if configured."""
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.name,
        num_labels=config.model.num_labels,
        id2label={0: "HUMAN", 1: "AI"},
        label2id={"HUMAN": 0, "AI": 1},
    )

    if config.model.use_lora and config.model.lora is not None:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model

        lora_cfg = config.model.lora
        peft_config = PeftLoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=lora_cfg.target_modules,
            bias=lora_cfg.bias,
            use_rslora=lora_cfg.use_rslora,
            modules_to_save=lora_cfg.modules_to_save,
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def build_compute_metrics() -> callable:
    """Build a compute_metrics function for HF Trainer.

    Computes: accuracy, f1, auc, ece.
    """
    metric_accuracy = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.nn.functional.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
        preds = np.argmax(logits, axis=1)

        accuracy = metric_accuracy.compute(predictions=preds, references=labels)["accuracy"]
        f1 = metric_f1.compute(predictions=preds, references=labels)["f1"]

        try:
            auc = roc_auc_score(labels, probs[:, 1])
        except ValueError:
            auc = 0.0

        # ECE (Expected Calibration Error)
        ece = _compute_ece(labels, probs[:, 1], n_bins=15)

        return {"accuracy": accuracy, "f1": f1, "auc": auc, "ece": ece}

    return compute_metrics


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        ece += mask.sum() * abs(bin_accuracy - bin_confidence)
    return float(ece / len(y_true))


def build_training_args(config: Config, output_dir: Path) -> TrainingArguments:
    """Convert our Config into HF TrainingArguments."""
    tc = config.training
    kwargs = {
        "output_dir": str(output_dir / "trainer_output"),
        "num_train_epochs": tc.num_epochs,
        "per_device_train_batch_size": tc.per_device_train_batch_size,
        "per_device_eval_batch_size": tc.per_device_eval_batch_size,
        "gradient_accumulation_steps": tc.gradient_accumulation_steps,
        "learning_rate": tc.learning_rate,
        "weight_decay": tc.weight_decay,
        "warmup_ratio": tc.warmup_ratio,
        "fp16": tc.fp16 and torch.cuda.is_available(),
        "eval_strategy": tc.eval_strategy,
        "save_strategy": tc.save_strategy,
        "metric_for_best_model": tc.metric_for_best_model,
        "greater_is_better": tc.greater_is_better,
        "load_best_model_at_end": tc.load_best_model_at_end,
        "save_total_limit": tc.save_total_limit,
        "logging_steps": tc.logging_steps,
        "dataloader_num_workers": tc.dataloader_num_workers,
        "seed": config.seed,
        "report_to": "none",
    }

    if tc.save_steps is not None:
        kwargs["save_steps"] = tc.save_steps
    if tc.eval_steps is not None:
        kwargs["eval_steps"] = tc.eval_steps

    return TrainingArguments(**kwargs)


def _save_env_info(output_dir: Path) -> None:
    """Save environment info to env.json."""
    import transformers

    info = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "python_version": sys.version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)

    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_commit"] = "unknown"

    with open(output_dir / "env.json", "w") as f:
        json.dump(info, f, indent=2)


def train(config: Config) -> Path:
    """Main training entrypoint.

    Returns Path to the run output directory.
    """
    set_deterministic(config.seed)

    # Create run directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{config.run_name}_{timestamp}"
    output_dir = Path(config.output.runs_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    import yaml

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)

    # Save environment info
    _save_env_info(output_dir)

    # Load data
    print(f"Loading dataset: {config.data.dataset}")
    dataset = prepare_wiki_dataset(config.data, config.seed)
    print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

    # Load model and tokenizer
    print(f"Loading model: {config.model.name} (LoRA: {config.model.use_lora})")
    model, tokenizer = load_model_and_tokenizer(config)

    # Tokenize
    tokenized = tokenize_dataset(dataset, tokenizer, config.data.max_length)

    # Build trainer
    training_args = build_training_args(config, output_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics = build_compute_metrics()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save best model
    checkpoint_dir = output_dir / "checkpoint"
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    # Save training metrics
    train_metrics = trainer.evaluate(tokenized["validation"])
    with open(output_dir / "train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    print(f"Training complete. Run directory: {output_dir}")
    print(f"Val AUC: {train_metrics.get('eval_auc', 'N/A')}")
    return output_dir

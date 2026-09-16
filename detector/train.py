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
from datasets import ClassLabel, concatenate_datasets
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
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from detector.model import DebertaV3ForSlopDetection

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

    if "deberta-v3" in config.model.name.lower():
        model = DebertaV3ForSlopDetection.from_pretrained(
            config.model.name,
            num_labels=config.model.num_labels,
            num_dropout=config.model.num_dropout,
            dropout_rate=config.model.dropout_rate,
            label_smoothing=config.training.label_smoothing_factor,
            id2label={0: "HUMAN", 1: "AI"},
            label2id={"HUMAN": 0, "AI": 1},
        )
    else:
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

        try:
            from sklearn.metrics import average_precision_score
            pr_auc = float(average_precision_score(labels, probs[:, 1]))
        except ValueError:
            pr_auc = 0.0

        # ECE (Expected Calibration Error)
        ece = _compute_ece(labels, probs[:, 1], n_bins=15)

        return {"accuracy": accuracy, "f1": f1, "auc": auc, "pr_auc": pr_auc, "ece": ece}

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
        "max_grad_norm": tc.max_grad_norm,
        "label_smoothing_factor": tc.label_smoothing_factor,
        "lr_scheduler_type": tc.lr_scheduler_type,
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

    # Optional TrainingArguments fields based on transformers version
    if hasattr(tc, "warmup_ratio"):
        try:
            from inspect import signature
            if "warmup_ratio" in signature(TrainingArguments.__init__).parameters:
                kwargs["warmup_ratio"] = tc.warmup_ratio
        except Exception:
            pass

    if hasattr(tc, "bf16") and tc.bf16:
        kwargs["bf16"] = tc.bf16 and torch.cuda.is_bf16_supported()
        if kwargs["bf16"]:
            kwargs["fp16"] = False

    if hasattr(tc, "gradient_checkpointing") and tc.gradient_checkpointing:
        kwargs["gradient_checkpointing"] = tc.gradient_checkpointing

    if tc.save_steps is not None:
        kwargs["save_steps"] = tc.save_steps
    if tc.eval_steps is not None:
        kwargs["eval_steps"] = tc.eval_steps

    if "deberta-v3" in config.model.name.lower():
        kwargs["label_smoothing_factor"] = 0.0 # Our custom model handles label smoothing internally

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
    print(f"Wiki — Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

    if config.data.train_on_raid:
        try:
            from detector.data.raid import prepare_raid_for_training
            from datasets import interleave_datasets

            raid_splits = ["train"]
            if config.data.raid_extra_split:
                raid_splits.append("extra")

            raid_parts = []
            for split_name in raid_splits:
                print(f"Loading RAID '{split_name}' split for training...")
                part = prepare_raid_for_training(
                    split=split_name,
                    include_adversarial=config.data.raid_include_adversarial,
                    seed=config.seed,
                )
                raid_parts.append(part.cast_column("label", ClassLabel(names=["human", "ai"])))

            raid_combined = concatenate_datasets(raid_parts) if len(raid_parts) > 1 else raid_parts[0]

            # Use datasets.interleave_datasets to mix RAID and Wikipedia at 80/20 ratio
            print("Interleaving RAID and Wikipedia datasets with 80/20 ratio...")
            dataset["train"] = interleave_datasets(
                [raid_combined, dataset["train"]],
                probabilities=[0.8, 0.2],
                seed=config.seed,
                stopping_strategy="all_exhausted"
            )
            print(f"Combined dataset length: {len(dataset['train'])}")
        except ImportError:
            print("Warning: raid-bench not installed — training on wiki only. Run: pip install raid-bench")

    # Lightweight Data Augmentation
    def add_noise(examples):
        import random
        import string
        texts = examples["text"]
        noisy_texts = []
        for text in texts:
            if random.random() < 0.1: # 10% chance to perturb text
                if random.random() < 0.5:
                    text = text.lower() # casing perturbation
                else:
                    # punctuation jitter
                    if len(text) > 5:
                        idx = random.randint(0, len(text)-1)
                        char = random.choice(string.punctuation)
                        text = text[:idx] + char + text[idx+1:]
            noisy_texts.append(text)
        examples["text"] = noisy_texts
        return examples

    dataset["train"] = dataset["train"].map(add_noise, batched=True)

    # Load model and tokenizer
    print(f"Loading model: {config.model.name} (LoRA: {config.model.use_lora})")
    model, tokenizer = load_model_and_tokenizer(config)

    # Tokenize
    # Stride cannot be >= max_length (adjust for small max_length in smoke tests)
    stride = min(128, max(16, config.data.max_length // 4))
    tokenized = tokenize_dataset(dataset, tokenizer, config.data.max_length, stride=stride)

    # Build trainer
    training_args = build_training_args(config, output_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics = build_compute_metrics()

    optimizer = None
    lr_scheduler = None

    if "deberta-v3" in config.model.name.lower():
        opt_parameters = []
        named_parameters = list(model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        init_lr = config.training.learning_rate
        layer_decay = config.training.layer_decay if hasattr(config.training, "layer_decay") else 0.9
        weight_decay = config.training.weight_decay

        num_layers = getattr(model.config, "num_hidden_layers", 12)

        for n, p in named_parameters:
            if not p.requires_grad:
                continue

            wd = 0.0 if any(nd in n for nd in no_decay) else weight_decay
            lr = init_lr

            if "deberta.encoder.layer" in n:
                try:
                    layer_num = int(n.split("deberta.encoder.layer.")[1].split(".")[0])
                    lr = init_lr * (layer_decay ** (num_layers - layer_num))
                except Exception:
                    pass
            elif "deberta.embeddings" in n:
                lr = init_lr * (layer_decay ** (num_layers + 1))

            opt_parameters.append({"params": p, "weight_decay": wd, "lr": lr})

        optimizer = AdamW(opt_parameters, lr=init_lr)

        num_update_steps_per_epoch = len(tokenized["train"]) // (config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps)
        if len(tokenized["train"]) % (config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps) != 0:
            num_update_steps_per_epoch += 1
        max_steps = int(config.training.num_epochs * num_update_steps_per_epoch)
        warmup_steps = int(max_steps * config.training.warmup_ratio)

        if config.training.lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler) if optimizer is not None else (None, None),
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

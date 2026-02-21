"""Evaluation orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from detector.config import Config, load_config
from detector.data.wiki_human_ai import prepare_wiki_dataset
from detector.eval.metrics import (
    compute_auc,
    compute_confusion,
    compute_ece,
    compute_metrics_at_thresholds,
    compute_roc_curve,
    plot_roc_curve,
    save_roc_csv,
)


def load_checkpoint(
    checkpoint_dir: Path,
    device: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model + tokenizer from a training run checkpoint.

    Handles both regular and LoRA checkpoints.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = Path(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))

    # Try loading as PEFT model first
    try:
        from peft import PeftModel

        # Check if adapter_config.json exists
        if (checkpoint_dir / "adapter_config.json").exists():
            from peft import AutoPeftModelForSequenceClassification

            model = AutoPeftModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
            model = model.to(device)
            model.eval()
            return model, tokenizer
    except ImportError:
        pass

    # Standard model loading
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    model = model.to(device)
    model.eval()
    return model, tokenizer


def score_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    batch_size: int = 32,
    max_length: int = 512,
    device: str | None = None,
) -> np.ndarray:
    """Score a list of texts. Returns array of P(AI) scores, shape (N,)."""
    if device is None:
        device = str(model.device) if hasattr(model, "device") else "cpu"

    all_scores = []
    model.eval()

    for i in tqdm(range(0, len(texts), batch_size), desc="Scoring"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            ai_probs = probs[:, 1].cpu().numpy()
            all_scores.extend(ai_probs.tolist())

    return np.array(all_scores)


def run_wiki_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
    run_dir: Path,
) -> dict:
    """Run evaluation on the wiki_human_ai test split."""
    dataset = prepare_wiki_dataset(config.data, config.seed)
    test_ds = dataset["test"]

    if config.eval.max_eval_samples is not None:
        n = min(config.eval.max_eval_samples, len(test_ds))
        test_ds = test_ds.select(range(n))

    texts = test_ds["text"]
    labels = np.array(test_ds["label"])

    scores = score_dataset(model, tokenizer, texts, max_length=config.data.max_length)

    metrics = compute_metrics_at_thresholds(labels, scores, config.eval.fpr_targets, config.eval.tpr_targets)
    metrics["dataset"] = "wiki_human_ai"
    metrics["n_samples"] = len(labels)
    metrics["n_human"] = int((labels == 0).sum())
    metrics["n_ai"] = int((labels == 1).sum())

    # ROC curve artifacts
    fpr_arr, tpr_arr, thresholds = compute_roc_curve(labels, scores)
    save_roc_csv(fpr_arr, tpr_arr, thresholds, run_dir / "roc.csv")
    plot_roc_curve(fpr_arr, tpr_arr, metrics["auc"], save_path=run_dir / "roc.png")

    return metrics


def run_raid_labeled_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
    run_dir: Path,
) -> dict:
    """Run evaluation on RAID labeled splits (train/extra). Computes full metrics + slices."""
    from detector.data.raid import build_raid_detector_fn, load_raid_dataframe
    from detector.eval.slices import build_eval_dataframe, compute_all_slices

    df = load_raid_dataframe(
        split=config.eval.raid_split,
        include_adversarial=config.eval.raid_include_adversarial,
    )

    # Derive labels: model=="human" -> 0, else -> 1
    labels = (df["model"] != "human").astype(int).values
    texts = df["generation"].tolist()

    if config.eval.max_eval_samples is not None:
        n = min(config.eval.max_eval_samples, len(texts))
        texts = texts[:n]
        labels = labels[:n]
        df = df.iloc[:n]

    scores = score_dataset(model, tokenizer, texts, max_length=config.data.max_length)

    # Overall metrics
    metrics = compute_metrics_at_thresholds(labels, scores, config.eval.fpr_targets, config.eval.tpr_targets)
    metrics["dataset"] = f"raid_{config.eval.raid_split}"
    metrics["n_samples"] = len(labels)
    metrics["n_human"] = int((labels == 0).sum())
    metrics["n_ai"] = int((labels == 1).sum())

    # Slice analysis
    metadata = {}
    for col in ["domain", "model", "decoding", "attack"]:
        if col in df.columns:
            vals = df[col].tolist()
            if config.eval.max_eval_samples is not None:
                vals = vals[:n]
            metadata[col] = vals

    eval_df = build_eval_dataframe(labels, scores, metadata)
    slices = compute_all_slices(eval_df, config.eval.fpr_targets, config.eval.tpr_targets)

    # Save RAID-specific artifacts
    with open(run_dir / "raid_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(run_dir / "slices.json", "w") as f:
        json.dump(slices, f, indent=2)

    # ROC for RAID
    fpr_arr, tpr_arr, thresholds = compute_roc_curve(labels, scores)
    save_roc_csv(fpr_arr, tpr_arr, thresholds, run_dir / "raid_roc.csv")
    plot_roc_curve(fpr_arr, tpr_arr, metrics["auc"], save_path=run_dir / "raid_roc.png")

    return metrics


def run_raid_submission(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
    run_dir: Path,
) -> Path:
    """Generate predictions.json for RAID leaderboard submission (unlabeled test split).

    No metrics computed — labels are not available.
    """
    from detector.data.raid import build_raid_detector_fn, load_raid_dataframe

    try:
        from raid import run_detection
        from raid.utils import load_data

        df = load_data(split="test")
        detector_fn = build_raid_detector_fn(model, tokenizer, config.data.max_length)
        predictions = run_detection(detector_fn, df)

        output_path = run_dir / "predictions.json"
        with open(output_path, "w") as f:
            json.dump(predictions, f)

        print(f"RAID leaderboard predictions saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"RAID submission generation failed: {e}")
        raise


def run_full_eval(config: Config, run_dir: Path, dataset: str = "all") -> dict:
    """Run evaluations based on dataset flag.

    dataset: "wiki", "raid_labeled", "raid_test_unlabeled", "all"
    """
    run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoint"
    model, tokenizer = load_checkpoint(checkpoint_dir)

    results = {}

    if dataset in ("wiki", "all"):
        print("Running wiki evaluation...")
        wiki_metrics = run_wiki_eval(model, tokenizer, config, run_dir)
        results["wiki"] = wiki_metrics

    if dataset in ("raid_labeled", "all"):
        print("Running RAID labeled evaluation...")
        raid_metrics = run_raid_labeled_eval(model, tokenizer, config, run_dir)
        results["raid"] = raid_metrics

    if dataset == "raid_test_unlabeled":
        print("Generating RAID leaderboard submission...")
        run_raid_submission(model, tokenizer, config, run_dir)
        results["raid_submission"] = {"status": "generated", "path": str(run_dir / "predictions.json")}

    # Save combined metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete. Results at: {run_dir}")
    return results

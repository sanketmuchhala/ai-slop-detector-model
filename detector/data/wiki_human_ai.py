"""Wikipedia Human-AI dataset loader."""

from __future__ import annotations

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase

from detector.config import DataConfig


def load_wiki_human_ai_raw() -> DatasetDict:
    """Load raw dataset from HuggingFace Hub.

    Returns DatasetDict with 'train' split containing columns:
    page_title, human_text, ai_text, split
    """
    return load_dataset("gouwsxander/wikipedia-human-ai")


def prepare_wiki_dataset(data_config: DataConfig, seed: int = 42) -> DatasetDict:
    """Load, flatten, and split the wiki dataset.

    1. Load from HF Hub
    2. Create separate rows for human_text (label=0) and ai_text (label=1)
    3. Shuffle with seed
    4. Stratified split into train/val/test

    Returns DatasetDict with splits: train, validation, test
    Each example has: text (str), label (int 0 or 1)
    """
    raw = load_wiki_human_ai_raw()
    # Dataset has a single 'train' split
    ds = raw["train"]

    # Create human rows (label=0)
    human_ds = ds.map(
        lambda x: {"text": x["human_text"], "label": 0},
        remove_columns=ds.column_names,
    )

    # Create AI rows (label=1)
    ai_ds = ds.map(
        lambda x: {"text": x["ai_text"], "label": 1},
        remove_columns=ds.column_names,
    )

    # Combine and shuffle
    combined = concatenate_datasets([human_ds, ai_ds])
    combined = combined.shuffle(seed=seed)

    # Cast label to ClassLabel so stratify_by_column works
    combined = combined.cast_column("label", ClassLabel(names=["human", "ai"]))

    # Stratified split: train / test+val first, then split test+val
    test_val_size = data_config.val_size + data_config.test_size
    split1 = combined.train_test_split(
        test_size=test_val_size,
        seed=seed,
        stratify_by_column="label",
    )

    # Split the test+val portion into val and test
    relative_test_size = data_config.test_size / test_val_size
    split2 = split1["test"].train_test_split(
        test_size=relative_test_size,
        seed=seed,
        stratify_by_column="label",
    )

    result = DatasetDict(
        {
            "train": split1["train"],
            "validation": split2["train"],
            "test": split2["test"],
        }
    )

    # Apply sample limits for smoke testing
    if data_config.max_train_samples is not None:
        n = min(data_config.max_train_samples, len(result["train"]))
        result["train"] = result["train"].select(range(n))

    if data_config.max_eval_samples is not None:
        for split_name in ["validation", "test"]:
            n = min(data_config.max_eval_samples, len(result[split_name]))
            result[split_name] = result[split_name].select(range(n))

    return result


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> DatasetDict:
    """Tokenize all splits. Adds input_ids, attention_mask, preserves label."""

    def _tokenize(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            max_length=max_length,
            truncation=True,
        )

    return dataset.map(_tokenize, batched=True, remove_columns=["text"])

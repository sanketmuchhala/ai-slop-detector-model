"""Tests for wiki_human_ai dataset loader."""

import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset, DatasetDict

from detector.config import DataConfig
from detector.data.wiki_human_ai import prepare_wiki_dataset, tokenize_dataset


def _make_fake_dataset(n: int = 100):
    """Create a fake wiki dataset with the expected schema."""
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "page_title": [f"Page_{i}" for i in range(n)],
                    "human_text": [f"Human text for page {i}" for i in range(n)],
                    "ai_text": [f"AI generated text for page {i}" for i in range(n)],
                    "split": ["train"] * n,
                }
            )
        }
    )


@patch("detector.data.wiki_human_ai.load_dataset")
def test_prepare_shapes(mock_load):
    mock_load.return_value = _make_fake_dataset(100)
    config = DataConfig(val_size=0.1, test_size=0.1)
    ds = prepare_wiki_dataset(config, seed=42)
    assert "train" in ds
    assert "validation" in ds
    assert "test" in ds
    total = len(ds["train"]) + len(ds["validation"]) + len(ds["test"])
    assert total == 200  # 100 human + 100 ai


@patch("detector.data.wiki_human_ai.load_dataset")
def test_labels_are_binary(mock_load):
    mock_load.return_value = _make_fake_dataset(50)
    config = DataConfig(val_size=0.1, test_size=0.1)
    ds = prepare_wiki_dataset(config, seed=42)
    for split in ds:
        labels = set(ds[split]["label"])
        assert labels.issubset({0, 1})


@patch("detector.data.wiki_human_ai.load_dataset")
def test_stratified_split(mock_load):
    mock_load.return_value = _make_fake_dataset(200)
    config = DataConfig(val_size=0.1, test_size=0.1)
    ds = prepare_wiki_dataset(config, seed=42)
    for split in ds:
        labels = ds[split]["label"]
        n_human = sum(1 for l in labels if l == 0)
        n_ai = sum(1 for l in labels if l == 1)
        ratio = n_human / (n_human + n_ai)
        assert 0.35 <= ratio <= 0.65, f"Split {split} has skewed ratio: {ratio}"


@patch("detector.data.wiki_human_ai.load_dataset")
def test_max_train_samples(mock_load):
    mock_load.return_value = _make_fake_dataset(100)
    config = DataConfig(val_size=0.1, test_size=0.1, max_train_samples=20)
    ds = prepare_wiki_dataset(config, seed=42)
    assert len(ds["train"]) == 20


@patch("detector.data.wiki_human_ai.load_dataset")
def test_max_eval_samples(mock_load):
    mock_load.return_value = _make_fake_dataset(100)
    config = DataConfig(val_size=0.1, test_size=0.1, max_eval_samples=5)
    ds = prepare_wiki_dataset(config, seed=42)
    assert len(ds["validation"]) == 5
    assert len(ds["test"]) == 5


@patch("detector.data.wiki_human_ai.load_dataset")
def test_tokenize_dataset(mock_load):
    mock_load.return_value = _make_fake_dataset(20)
    config = DataConfig(val_size=0.1, test_size=0.1)
    ds = prepare_wiki_dataset(config, seed=42)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    tokenized = tokenize_dataset(ds, tokenizer, max_length=64)
    for split in tokenized:
        assert "input_ids" in tokenized[split].column_names
        assert "attention_mask" in tokenized[split].column_names
        assert "label" in tokenized[split].column_names
        assert "text" not in tokenized[split].column_names
        for ids in tokenized[split]["input_ids"]:
            assert len(ids) <= 64

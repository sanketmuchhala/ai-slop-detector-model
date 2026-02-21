"""Tests for RAID data loader."""

import pytest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from detector.data.raid import RAID_INSTALL_MSG, _check_raid_installed, build_raid_detector_fn, raid_df_to_labels


def test_raid_import_error_message():
    """Verify helpful error message when raid-bench is not installed."""
    with patch.dict("sys.modules", {"raid": None, "raid.utils": None}):
        with pytest.raises(ImportError, match="pip install raid-bench"):
            _check_raid_installed()


def test_raid_df_to_labels():
    df = pd.DataFrame({"model": ["human", "gpt4", "human", "llama"]})
    labels = raid_df_to_labels(df)
    np.testing.assert_array_equal(labels, [0, 1, 0, 1])


def test_build_detector_fn_signature(mock_model):
    """Detector fn should accept list[str] and return list[float]."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    import torch

    def fake_call(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        return MagicMock(logits=torch.randn(batch_size, 2))

    mock_model.side_effect = fake_call
    mock_model.device = torch.device("cpu")

    fn = build_raid_detector_fn(mock_model, tokenizer, max_length=64, batch_size=2, device="cpu")
    results = fn(["Hello world", "Test text"])
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(x, float) for x in results)


def test_detector_fn_output_range(mock_model):
    """All outputs should be in [0, 1]."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    import torch

    def fake_call(**kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        return MagicMock(logits=torch.randn(batch_size, 2))

    mock_model.side_effect = fake_call
    mock_model.device = torch.device("cpu")

    fn = build_raid_detector_fn(mock_model, tokenizer, max_length=64, batch_size=4, device="cpu")
    results = fn(["text " + str(i) for i in range(10)])
    assert all(0.0 <= x <= 1.0 for x in results)

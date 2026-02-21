"""Tests for slice-based evaluation."""

import numpy as np
import pandas as pd
import pytest

from detector.eval.slices import build_eval_dataframe, compute_all_slices, compute_slice_metrics


def _make_eval_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "label": rng.randint(0, 2, size=n),
            "score": rng.rand(n),
            "domain": rng.choice(["news", "wiki", "reddit"], size=n),
            "model": rng.choice(["human", "gpt4", "llama"], size=n),
            "decoding": rng.choice(["greedy", "sampling"], size=n),
            "attack": rng.choice(["none", "paraphrase", "homoglyph"], size=n),
        }
    )


def test_build_eval_dataframe():
    y_true = np.array([0, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.2, 0.8])
    metadata = {"domain": ["news", "wiki", "news", "wiki"]}
    df = build_eval_dataframe(y_true, y_scores, metadata)
    assert "label" in df.columns
    assert "score" in df.columns
    assert "domain" in df.columns
    assert len(df) == 4


def test_compute_slice_metrics():
    df = _make_eval_df(300)
    results = compute_slice_metrics(df, "domain", fpr_targets=[0.01], tpr_targets=[0.90])
    assert isinstance(results, dict)
    for domain, metrics in results.items():
        assert "auc" in metrics
        assert "count" in metrics
        assert "tpr_at_fpr" in metrics


def test_compute_all_slices():
    df = _make_eval_df(300)
    results = compute_all_slices(df, fpr_targets=[0.01], tpr_targets=[0.90])
    assert "overall" in results
    assert "by_domain" in results
    assert "by_model" in results
    assert "by_decoding" in results
    assert "by_attack" in results
    assert results["overall"]["count"] == 300


def test_min_samples_filter():
    df = _make_eval_df(60)
    # With min_samples=50, some slices might be filtered
    results = compute_slice_metrics(df, "domain", min_samples=50)
    for domain, metrics in results.items():
        assert metrics["count"] >= 50


def test_missing_column():
    df = pd.DataFrame({"label": [0, 1], "score": [0.1, 0.9]})
    results = compute_slice_metrics(df, "nonexistent_column")
    assert results == {}

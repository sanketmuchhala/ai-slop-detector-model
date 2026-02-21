"""Slice-based evaluation: group predictions by metadata axes and compute per-slice metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from detector.eval.metrics import compute_auc, compute_ece, compute_roc_curve, fpr_at_tpr, tpr_at_fpr

SLICE_AXES = ["domain", "model", "decoding", "attack"]


def build_eval_dataframe(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metadata: dict[str, list],
) -> pd.DataFrame:
    """Combine scores, labels, and metadata into a single DataFrame for slicing."""
    df = pd.DataFrame({"label": y_true, "score": y_scores})
    for col, values in metadata.items():
        df[col] = values
    return df


def _compute_slice_metrics_single(
    labels: np.ndarray,
    scores: np.ndarray,
    fpr_targets: list[float],
    tpr_targets: list[float],
) -> dict:
    """Compute metrics for a single slice."""
    n_unique = len(np.unique(labels))
    result = {"count": len(labels), "n_human": int((labels == 0).sum()), "n_ai": int((labels == 1).sum())}

    if n_unique < 2:
        result["auc"] = None
        result["ece"] = None
        result["tpr_at_fpr"] = {str(t): None for t in fpr_targets}
        result["fpr_at_tpr"] = {str(t): None for t in tpr_targets}
        return result

    try:
        result["auc"] = round(compute_auc(labels, scores), 6)
    except ValueError:
        result["auc"] = None

    result["ece"] = round(compute_ece(labels, scores), 6)

    fpr_arr, tpr_arr, _ = compute_roc_curve(labels, scores)
    result["tpr_at_fpr"] = {str(t): round(tpr_at_fpr(fpr_arr, tpr_arr, t), 6) for t in fpr_targets}
    result["fpr_at_tpr"] = {str(t): round(fpr_at_tpr(fpr_arr, tpr_arr, t), 6) for t in tpr_targets}

    return result


def compute_slice_metrics(
    eval_df: pd.DataFrame,
    slice_col: str,
    fpr_targets: list[float] | None = None,
    tpr_targets: list[float] | None = None,
    min_samples: int = 50,
) -> dict[str, dict]:
    """Compute metrics for each unique value in a slice column.

    Skips slices with fewer than min_samples.
    """
    if fpr_targets is None:
        fpr_targets = [0.001, 0.005, 0.01, 0.02, 0.05]
    if tpr_targets is None:
        tpr_targets = [0.80, 0.90, 0.95]

    if slice_col not in eval_df.columns:
        return {}

    results = {}
    for value, group in eval_df.groupby(slice_col):
        if len(group) < min_samples:
            continue
        labels = group["label"].values
        scores = group["score"].values
        results[str(value)] = _compute_slice_metrics_single(labels, scores, fpr_targets, tpr_targets)

    return results


def compute_all_slices(
    eval_df: pd.DataFrame,
    fpr_targets: list[float] | None = None,
    tpr_targets: list[float] | None = None,
) -> dict[str, dict]:
    """Compute per-slice metrics across all SLICE_AXES.

    Returns: {
        "overall": {...},
        "by_domain": {value: {...}, ...},
        "by_model": {value: {...}, ...},
        ...
    }
    """
    if fpr_targets is None:
        fpr_targets = [0.001, 0.005, 0.01, 0.02, 0.05]
    if tpr_targets is None:
        tpr_targets = [0.80, 0.90, 0.95]

    result = {}

    # Overall
    result["overall"] = _compute_slice_metrics_single(
        eval_df["label"].values, eval_df["score"].values, fpr_targets, tpr_targets
    )

    # Per-axis
    for axis in SLICE_AXES:
        key = f"by_{axis}"
        result[key] = compute_slice_metrics(eval_df, axis, fpr_targets, tpr_targets)

    return result

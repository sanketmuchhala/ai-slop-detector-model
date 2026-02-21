"""Core metric computation: AUC, ROC, confusion matrices, threshold-based metrics, ECE."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve. Returns (fpr_array, tpr_array, thresholds_array)."""
    return roc_curve(y_true, y_scores)


def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute AUC-ROC score."""
    return float(roc_auc_score(y_true, y_scores))


def tpr_at_fpr(
    fpr_arr: np.ndarray,
    tpr_arr: np.ndarray,
    target_fpr: float,
) -> float:
    """Find TPR at a given FPR threshold. Returns highest TPR where FPR <= target_fpr."""
    idx = np.where(fpr_arr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr_arr[idx[-1]])


def fpr_at_tpr(
    fpr_arr: np.ndarray,
    tpr_arr: np.ndarray,
    target_tpr: float,
) -> float:
    """Find FPR at a given TPR threshold. Returns lowest FPR where TPR >= target_tpr."""
    idx = np.where(tpr_arr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr_arr[idx[0]])


def threshold_at_fpr(
    fpr_arr: np.ndarray,
    thresholds: np.ndarray,
    target_fpr: float,
) -> float:
    """Find the score threshold that achieves a target FPR."""
    idx = np.where(fpr_arr <= target_fpr)[0]
    if len(idx) == 0:
        return 1.0
    return float(thresholds[idx[-1]])


def threshold_at_tpr(
    tpr_arr: np.ndarray,
    thresholds: np.ndarray,
    target_tpr: float,
) -> float:
    """Find the score threshold that achieves a target TPR."""
    idx = np.where(tpr_arr >= target_tpr)[0]
    if len(idx) == 0:
        return 0.0
    return float(thresholds[idx[0]])


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute confusion matrix and return as a dict."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * tpr_val / (precision + tpr_val) if (precision + tpr_val) > 0 else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "tpr": round(tpr_val, 6),
        "fpr": round(fpr_val, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6),
    }


def compute_ece(y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error.

    Measures how well predicted probabilities match actual frequencies.
    Returns value in [0, 1]. Lower is better. Flag if > 0.05.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        mask = (y_scores > bin_boundaries[i]) & (y_scores <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = y_scores[mask].mean()
        bin_accuracy = y_true[mask].mean()
        ece += mask.sum() * abs(float(bin_accuracy) - float(bin_confidence))
    return float(ece / n)


def compute_metrics_at_thresholds(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    fpr_targets: list[float],
    tpr_targets: list[float],
) -> dict:
    """Compute TPR@FPR and FPR@TPR for all target operating points."""
    fpr_arr, tpr_arr, thresholds = compute_roc_curve(y_true, y_scores)
    auc = compute_auc(y_true, y_scores)
    ece = compute_ece(y_true, y_scores)

    tpr_at_fpr_results = {}
    for target in fpr_targets:
        tpr_at_fpr_results[str(target)] = round(tpr_at_fpr(fpr_arr, tpr_arr, target), 6)

    fpr_at_tpr_results = {}
    for target in tpr_targets:
        fpr_at_tpr_results[str(target)] = round(fpr_at_tpr(fpr_arr, tpr_arr, target), 6)

    return {
        "auc": round(auc, 6),
        "ece": round(ece, 6),
        "tpr_at_fpr": tpr_at_fpr_results,
        "fpr_at_tpr": fpr_at_tpr_results,
    }


def plot_roc_curve(
    fpr_arr: np.ndarray,
    tpr_arr: np.ndarray,
    auc_score: float,
    conservative_point: tuple[float, float] | None = None,
    balanced_point: tuple[float, float] | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot ROC curve with AUC in legend, mark operating points."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, label=f"ROC (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)

    if conservative_point:
        ax.scatter(
            *conservative_point,
            color="red",
            zorder=5,
            s=80,
            label=f"Conservative (FPR={conservative_point[0]:.3f})",
        )
    if balanced_point:
        ax.scatter(
            *balanced_point,
            color="blue",
            zorder=5,
            s=80,
            label=f"Balanced (TPR={balanced_point[1]:.3f})",
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_roc_csv(
    fpr_arr: np.ndarray,
    tpr_arr: np.ndarray,
    thresholds: np.ndarray,
    save_path: Path,
) -> None:
    """Save ROC data as CSV with columns: threshold, fpr, tpr."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "fpr", "tpr"])
        for t, fp, tp in zip(thresholds, fpr_arr, tpr_arr):
            writer.writerow([round(float(t), 6), round(float(fp), 6), round(float(tp), 6)])

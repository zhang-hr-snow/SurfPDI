"""
Input:
    Residue-level logits and labels.

Output:
    Paper-style binary classification metrics:
    Specificity, Recall, Precision, F1, AUC, MCC, Accuracy.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_probabilities(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[-1] == 2:
        return torch.softmax(logits.float(), dim=-1)[:, 1]
    return torch.sigmoid(logits.float().view(-1))


def binary_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    labels_cpu = labels.detach().long().cpu()
    probs_cpu = positive_probabilities(logits.detach()).cpu()

    y_true = labels_cpu.numpy()
    y_score = probs_cpu.numpy()
    y_pred = (y_score >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_score))
        aupr = float(average_precision_score(y_true, y_score))
    else:
        auc = math.nan
        aupr = math.nan

    return {
        "specificity": float(specificity),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "aupr": aupr,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def metrics_from_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = labels.astype(np.int64)
    y_score = probabilities.astype(np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_score))
        aupr = float(average_precision_score(y_true, y_score))
    else:
        auc = math.nan
        aupr = math.nan

    return {
        "specificity": float(specificity),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
        "aupr": aupr,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def tune_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    metric: str = "mcc",
    num_thresholds: int = 200,
) -> dict[str, float]:
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_threshold = 0.5
    best_score = -float("inf")
    best_metrics: dict[str, float] | None = None

    for threshold in thresholds:
        metrics = metrics_from_probabilities(probabilities, labels, float(threshold))
        score = float(metrics[metric])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    assert best_metrics is not None
    return {
        "metric": metric,
        "threshold": best_threshold,
        "score": best_score,
        "metrics": best_metrics,
    }

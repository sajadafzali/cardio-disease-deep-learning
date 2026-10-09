"""
metrics.py - classification metrics used for every model (baselines and MLP alike).

Why these metrics for THIS problem (balanced classes, medical screening):
  accuracy  - fair here because the classes are 50/50.
  ROC-AUC   - threshold-free ranking quality; best single number for comparing models.
  log loss  - rewards well-calibrated probabilities, not just correct labels.
  precision - of patients flagged as sick, how many really are.
  recall    - of patients who really are sick, how many we catch (missing one is costly).
  F1        - balance of precision and recall.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)


def binary_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    """All metrics for predicted probabilities `prob` of the positive class."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "auc": roc_auc_score(y_true, prob),
        "loss": log_loss(y_true, np.clip(prob, 1e-7, 1 - 1e-7)),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
    }

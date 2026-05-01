"""Discrimination metrics for confidence estimation.

Each function takes a list of (confidence, correctness) pairs, where
``confidence`` is in [0, 1] and ``correctness`` is 0 or 1.
"""

import math

from sklearn.metrics import average_precision_score, roc_auc_score


def nce(pairs):
    """Normalized Cross Entropy [Siu, Gish, Richardson, 1997].

    Returns a value in (-inf, 1]; higher is better. Returns 0.0 when the
    label distribution is degenerate (all correct or all incorrect), since
    NCE is undefined in that case.
    """
    eps = 1e-9
    num_samples = len(pairs)
    p_correct = sum(c for _, c in pairs) / num_samples
    p_wrong = 1.0 - p_correct

    if p_correct in (0.0, 1.0):
        return 0.0

    h_c = -(p_correct * math.log(p_correct + eps) + p_wrong * math.log(p_wrong + eps))

    h_cp = 0.0
    for p, c in pairs:
        p = min(max(p, eps), 1.0 - eps)
        h_cp += -math.log(p) if c == 1 else -math.log(1.0 - p)
    h_cp /= num_samples

    return (h_c - h_cp) / h_c


def aucroc(pairs):
    """AUC-ROC; returns NaN if labels are all the same class."""
    confidences = [p for p, _ in pairs]
    labels = [c for _, c in pairs]
    if len(set(labels)) < 2:
        return float("nan")
    return roc_auc_score(labels, confidences)


def aucpr(pairs):
    """AUC-PR with correct predictions as the positive class."""
    confidences = [p for p, _ in pairs]
    labels = [c for _, c in pairs]
    return average_precision_score(labels, confidences)


def aucpr_neg(pairs):
    """AUC-PR with errors (incorrect predictions) as the positive class."""
    confidences = [p for p, _ in pairs]
    labels = [1 - c for _, c in pairs]
    return average_precision_score(labels, confidences)


def all_metrics(pairs):
    """Return (NCE, AUC-ROC, AUC-PR, AUC-PR-neg) as a 4-tuple."""
    return nce(pairs), aucroc(pairs), aucpr(pairs), aucpr_neg(pairs)

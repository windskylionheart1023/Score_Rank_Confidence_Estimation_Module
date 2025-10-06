import math
from sklearn.metrics import roc_auc_score, average_precision_score

def nce(pairs):
    eps = 1e-9
    correctness = [c for _, c in pairs]
    confidences = [p for p, _ in pairs]

    # H(c): entropy of correctness labels
    num_samples = len(correctness)
    num_correct = sum(correctness)
    p_correct = num_correct / num_samples
    p_wrong = 1 - p_correct

    if p_correct in [0, 1]:
        h_c = 0.0  # log(0) case handled: entropy is 0 if labels are all the same
    else:
        h_c = - (p_correct * math.log(p_correct + eps) + p_wrong * math.log(p_wrong + eps))

    # H(c, p): cross-entropy between correctness and predicted confidence
    h_cp = 0.0
    for p, c in pairs:
        p = min(max(p, eps), 1 - eps)  # clamp
        if c == 1:
            h_cp += -math.log(p)
        else:
            h_cp += -math.log(1 - p)
    h_cp /= num_samples

    # Normalized Cross Entropy
    if h_c == 0:
        return 0.0  # undefined NCE if entropy is 0, can also return float('nan')
    else:
        nce = (h_c - h_cp) / h_c
        return nce

def aucroc(pairs):
    """
    pairs: list of (confidence, correctness) where correctness is 0 or 1
    returns: AUC-ROC (float in [0,1])
    """
    confidences = [p for p, _ in pairs]
    labels      = [c for _, c in pairs]
    # if all labels are the same, roc_auc_score is undefined
    if len(set(labels)) < 2:
        return float('nan')
    return roc_auc_score(labels, confidences)


def aucpr(pairs):
    """
    Traditional AUC-PR (correct=positive)
    """
    confidences = [p for p, _ in pairs]
    labels      = [c for _, c in pairs]
    # average_precision_score gracefully handles imbalanced classes
    return average_precision_score(labels, confidences)


def aucpr_neg(pairs):
    """
    AUC-PR with errors treated as positives (i.e. flip labels)
    """
    confidences = [p for p, _ in pairs]
    labels      = [1 - c for _, c in pairs]
    return average_precision_score(labels, confidences)


def all_metrics(pairs):
    """
    Returns a tuple (aucroc, aucpr, aucpr_neg)
    """
    return (
        nce(pairs),
        aucroc(pairs),
        aucpr(pairs),
        aucpr_neg(pairs),
    )

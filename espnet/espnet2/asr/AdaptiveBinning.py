"""Adaptive binning for calibration metrics (ECE / MCE).

Implementation of the adaptive binning scheme from:

    Ding, Liu, Xiong, Shi.
    "Revisiting the evaluation of uncertainty estimation and its application
    to explore model complexity-uncertainty trade-off."
    CVPR Workshops, 2020.

Unlike fixed-width binning, the bin boundaries here are determined by the
data distribution, with bin sizes constrained so that confidence and
accuracy estimates per bin are statistically meaningful.
"""

import matplotlib.pyplot as plt


def AdaptiveBinning(infer_results, show_reliability_diagram=True):
    """Compute adaptive-binning ECE and MCE.

    Args:
        infer_results: list of [confidence, correctness] pairs, where
            ``confidence`` is in [0, 1] and ``correctness`` is True/False
            (or 1/0).
        show_reliability_diagram: if True, draw the reliability diagram on
            the current matplotlib figure (caller is responsible for saving
            or showing it).

    Returns:
        (AECE, AMCE, cof_min, cof_max, confidence, accuracy):
            AECE: adaptive expected calibration error.
            AMCE: adaptive maximum calibration error.
            cof_min, cof_max: per-bin min/max confidence.
            confidence, accuracy: per-bin mean confidence/accuracy.
    """
    infer_results = sorted(infer_results, key=lambda x: x[0], reverse=True)
    n_total_sample = len(infer_results)

    assert (
        infer_results[0][0] <= 1 and infer_results[-1][0] >= 0
    ), "Confidence score should be in [0, 1]"

    # 1.645 corresponds to a 90% confidence interval; together with the
    # 0.25 factor (worst-case Bernoulli variance) it gives a target sample
    # count per bin so the per-bin frequency estimate is within +/-(width/2)
    # of the true probability.
    z = 1.645

    num = [0] * n_total_sample
    final_num = [0] * n_total_sample
    correct = [0] * n_total_sample
    confidence = [0] * n_total_sample
    cof_min = [1] * n_total_sample
    cof_max = [0] * n_total_sample

    ind = 0
    target_number_samples = float("inf")

    # First pass: greedy initial binning.
    for i, (confidence_score, correctness) in enumerate(infer_results):
        if num[ind] > target_number_samples:
            # Avoid creating a tiny last bin: only split if at least 40
            # samples remain and the new bin's range would exceed 0.05.
            if (n_total_sample - i) > 40 and cof_min[ind] - infer_results[-1][0] > 0.05:
                ind += 1
                target_number_samples = float("inf")

        num[ind] += 1
        confidence[ind] += confidence_score
        if bool(correctness):
            correct[ind] += 1
        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)

        if cof_max[ind] == cof_min[ind]:
            target_number_samples = float("inf")
        else:
            target_number_samples = (z / (cof_max[ind] - cof_min[ind])) ** 2 * 0.25

    n_bins = ind + 1

    # Rebalance the last bin if it is undersized.
    if target_number_samples - num[ind] > 0:
        needed = target_number_samples - num[ind]
        extract = [0] * (n_bins - 1)
        final_num[n_bins - 1] = num[n_bins - 1]
        for i in range(n_bins - 1):
            extract[i] = int(needed * num[ind] / n_total_sample)
            final_num[i] = num[i] - extract[i]
            final_num[n_bins - 1] += extract[i]
    else:
        final_num = num
    final_num = final_num[:n_bins]

    # Second pass: assign samples to bins per ``final_num`` and compute stats.
    num = [0] * n_bins
    correct = [0] * n_bins
    confidence = [0] * n_bins
    cof_min = [1] * n_bins
    cof_max = [0] * n_bins
    accuracy = [0] * n_bins
    gap = [0] * n_bins
    neg_gap = [0] * n_bins
    x_location = [0] * n_bins
    width = [0] * n_bins

    ind = 0
    for confidence_score, correctness in infer_results:
        num[ind] += 1
        confidence[ind] += confidence_score
        if bool(correctness):
            correct[ind] += 1
        cof_min[ind] = min(cof_min[ind], confidence_score)
        cof_max[ind] = max(cof_max[ind], confidence_score)

        if num[ind] == final_num[ind]:
            confidence[ind] = confidence[ind] / num[ind] if num[ind] > 0 else 0
            accuracy[ind] = correct[ind] / num[ind] if num[ind] > 0 else 0
            left, right = cof_min[ind], cof_max[ind]
            x_location[ind] = (left + right) / 2
            width[ind] = (right - left) * 0.9
            if confidence[ind] - accuracy[ind] > 0:
                gap[ind] = confidence[ind] - accuracy[ind]
            else:
                neg_gap[ind] = confidence[ind] - accuracy[ind]
            ind += 1

    AMCE = 0.0
    AECE = 0.0
    for i in range(n_bins):
        AECE += abs(accuracy[i] - confidence[i]) * final_num[i] / n_total_sample
        AMCE = max(AMCE, abs(accuracy[i] - confidence[i]))

    if show_reliability_diagram:
        fig, ax = plt.subplots()
        ax.bar(x_location, accuracy, width)
        ax.bar(x_location, gap, width, bottom=accuracy)
        ax.bar(x_location, neg_gap, width, bottom=accuracy)
        ax.legend(
            ["Accuracy", "Overconfident", "Underconfident"], fontsize=14, loc=2
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontsize=15)
        ax.set_ylabel("Accuracy", fontsize=15)

    return AECE, AMCE, cof_min, cof_max, confidence, accuracy

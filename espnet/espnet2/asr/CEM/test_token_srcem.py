#!/usr/bin/env python3
"""Evaluate a trained token-level SR-CEM and the softmax baseline.

For both the SR-CEM predictions and the raw softmax confidence (kept as
feature index 0 of the dataset, see ``dataset_token_srcem.py``), this script
reports NCE / AUC-ROC / AUC-PR / AUC-PR_neg (from :mod:`metrics`) and the
adaptive-binning ECE / MCE (from :mod:`AdaptiveBinning`).
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from espnet2.asr.AdaptiveBinning import AdaptiveBinning
from espnet2.asr.CEM.metrics import all_metrics
from espnet2.asr.CEM.model import NamedTensorDataset, Score_CEM


def load_test_dataset(path: str, remove_features, filter_top1: bool):
    """Load test dataset, drop features, optionally filter rank-1 tokens out.

    Returns ``(TensorDataset, list_of_softmax_confidences)``. The softmax
    confidence is read BEFORE the feature drop, so it is preserved as the
    baseline even when ``confidence`` is in ``remove_features`` (the usual
    case at training time).
    """
    dataset: NamedTensorDataset = torch.load(path)

    # Extract the softmax baseline before projecting features away.
    if "confidence" in dataset.feature_names:
        conf_idx = dataset.feature_names.index("confidence")
        native_confidence = dataset.data[:, conf_idx].tolist()
    else:
        native_confidence = dataset.data[:, 0].tolist()

    keep_indices = [
        i for i, name in enumerate(dataset.feature_names) if name not in remove_features
    ]
    rank_idx_after = next(
        (j for j, i in enumerate(keep_indices) if dataset.feature_names[i] == "rank"),
        None,
    )
    dataset.data = dataset.data[:, keep_indices]
    dataset.feature_names = [dataset.feature_names[i] for i in keep_indices]
    print("Using features:", dataset.feature_names)

    tensor_ds = TensorDataset(dataset.data, dataset.targets)

    if filter_top1:
        if rank_idx_after is None:
            raise ValueError("--filter-top1 requires the 'rank' feature in the dataset.")
        x, y = tensor_ds.tensors
        mask = (x[:, rank_idx_after] != 1.0).tolist()
        x = x[mask]
        y = y[mask]
        native_confidence = [v for v, k in zip(native_confidence, mask) if k]
        tensor_ds = TensorDataset(x, y)

    return tensor_ds, native_confidence


def run_inference(model, dataloader, device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_data, _ in dataloader:
            outputs = model(batch_data.to(device)).squeeze(1)
            predictions.extend(outputs.cpu().tolist())
    return predictions


def evaluate_predictions(conf_corr_list, label: str, plot_path: str):
    AECE, AMCE, *_ = AdaptiveBinning(conf_corr_list, show_reliability_diagram=True)
    nce, aucroc, aucpr, aucpr_neg = all_metrics(conf_corr_list)

    print(f"{label} - ECE (adaptive binning): {AECE * 100:.2f}%")
    print(f"{label} - MCE (adaptive binning): {AMCE * 100:.2f}%")
    print(f"{label} - NCE:        {nce:.4f}")
    print(f"{label} - AUC-ROC:    {aucroc:.3f}")
    print(f"{label} - AUC-PR:     {aucpr:.3f}")
    print(f"{label} - AUC-PR_neg: {aucpr_neg:.3f}")

    if plot_path:
        os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
        plt.savefig(plot_path)
        print(f"{label} reliability diagram saved to {plot_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dataset", required=True, help="Path to the test .pt dataset.")
    parser.add_argument("--model-path", required=True, help="Trained Score_CEM checkpoint.")
    parser.add_argument(
        "--remove-features",
        nargs="*",
        default=["confidence"],
        help="Feature names to drop before inference (must match training).",
    )
    parser.add_argument("--prediction-out", default=None, help="Optional .pt to save (conf, correctness) pairs for SR-CEM.")
    parser.add_argument("--native-prediction-out", default=None, help="Optional .pt for the softmax baseline.")
    parser.add_argument("--cem-plot", default=None, help="Optional path for the SR-CEM reliability diagram.")
    parser.add_argument("--native-plot", default=None, help="Optional path for the softmax reliability diagram.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument(
        "--filter-top1",
        action="store_true",
        help="Drop rank-1 tokens before evaluating (focuses on competitive tokens).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    test_dataset, native_confidence = load_test_dataset(
        args.test_dataset, args.remove_features, args.filter_top1
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    _, target = test_dataset.tensors
    correctness = target.squeeze(1).tolist()

    input_size = test_dataset.tensors[0].shape[1]
    model = Score_CEM(input_size=input_size, hidden_size=args.hidden_size).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))

    predictions = run_inference(model, test_loader, args.device)

    cem_combined = [[p, c] for p, c in zip(predictions, correctness)]
    native_combined = [[p, c] for p, c in zip(native_confidence, correctness)]

    if args.prediction_out:
        os.makedirs(os.path.dirname(args.prediction_out) or ".", exist_ok=True)
        torch.save(cem_combined, args.prediction_out)
    if args.native_prediction_out:
        os.makedirs(os.path.dirname(args.native_prediction_out) or ".", exist_ok=True)
        torch.save(native_combined, args.native_prediction_out)

    print("\n=== SR-CEM ===")
    evaluate_predictions(cem_combined, "SR-CEM", args.cem_plot)
    print("\n=== Softmax baseline ===")
    evaluate_predictions(native_combined, "Softmax", args.native_plot)


if __name__ == "__main__":
    main()

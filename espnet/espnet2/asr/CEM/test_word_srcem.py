#!/usr/bin/env python3
"""Evaluate a trained word-level SR-CEM and the softmax baseline.

Same metrics and outputs as ``test_token_srcem.py`` but for the word-level
NamedTensorDataset (5 features after dropping ``confidence``).
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from espnet2.asr.AdaptiveBinning import AdaptiveBinning
from espnet2.asr.CEM.metrics import all_metrics
from espnet2.asr.CEM.model import NamedTensorDataset, Score_CEM


def load_test_dataset(path: str, remove_features):
    """Load test dataset and drop features.

    The softmax baseline confidence is read BEFORE the feature drop so it
    is preserved even when ``confidence`` is in ``remove_features``.
    """
    dataset: NamedTensorDataset = torch.load(path)

    if "confidence" in dataset.feature_names:
        conf_idx = dataset.feature_names.index("confidence")
        native_confidence = dataset.data[:, conf_idx].tolist()
    else:
        native_confidence = dataset.data[:, 0].tolist()

    keep_indices = [
        i for i, name in enumerate(dataset.feature_names) if name not in remove_features
    ]
    dataset.data = dataset.data[:, keep_indices]
    dataset.feature_names = [dataset.feature_names[i] for i in keep_indices]
    print("Using features:", dataset.feature_names)

    return TensorDataset(dataset.data, dataset.targets), native_confidence


def run_inference(model, dataloader, device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_data, _ in dataloader:
            outputs = model(batch_data.to(device)).squeeze(1)
            predictions.extend(outputs.cpu().tolist())
    return predictions


def evaluate(conf_corr_list, label: str, plot_path: str):
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    test_dataset, native_confidence = load_test_dataset(args.test_dataset, args.remove_features)
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
    evaluate(cem_combined, "SR-CEM", args.cem_plot)
    print("\n=== Softmax baseline ===")
    evaluate(native_combined, "Softmax", args.native_plot)


if __name__ == "__main__":
    main()

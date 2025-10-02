#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from espnet2.asr.CEM.model import Score_CEM, NamedTensorDataset
from espnet2.asr.CEM.metrics import all_metrics
from espnet2.asr.AdaptiveBinning import AdaptiveBinning


###############################################################################
#                                CONFIG                                       #
###############################################################################
SUFFIX = "_ours_libapt"
CEM_PATH = "/esat/audioslave/yjia/espnet/espnet/espnet2/asr/CEM"

TEST_DATASET_PATH = os.path.join(CEM_PATH, "dataset", "test_dataset_token_ours_libapt_s4.pt")
MODEL_PATH = os.path.join(CEM_PATH, "model", f"cem_model_token{SUFFIX}.pt")

REMOVE_FEATURES = ["confidence"]
DEVICE = "cuda"
BATCH_SIZE = 1024
FILTER = False


###############################################################################
#                          DATA PREPARATION                                   #
###############################################################################
def load_test_dataset(path: str, remove_features: list[str], filter: bool):
    """Load test dataset, drop features, apply optional filtering, and return TensorDataset."""
    dataset: NamedTensorDataset = torch.load(path)

    # Remove features
    keep_indices = [i for i, name in enumerate(dataset.feature_names) if name not in remove_features]
    dataset.data = dataset.data[:, keep_indices]
    dataset.feature_names = [dataset.feature_names[i] for i in keep_indices]

    print("Using features:", dataset.feature_names)

    native_confidence = dataset.data[:, 0].tolist()  # assumes confidence was first

    tensor_ds = TensorDataset(dataset.data, dataset.targets)

    if filter:
        x, y = tensor_ds.tensors
        mask = [r != 1.0 for r in x[:, 1].tolist()]  # example filter on second feature
        x = x[mask]
        y = y[mask]
        native_confidence = [native_confidence[i] for i, keep in enumerate(mask) if keep]
        tensor_ds = TensorDataset(x, y)

    return tensor_ds, native_confidence


###############################################################################
#                          MODEL INFERENCE                                    #
###############################################################################
def run_inference(model, dataloader):
    """Run inference and return predictions as a list."""
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(DEVICE)
            outputs = model(batch_data).squeeze(1)
            predictions.extend(outputs.cpu().tolist())
    return predictions


###############################################################################
#                          METRICS & PLOTTING                                 #
###############################################################################
def evaluate_predictions(conf_corr_list, label: str, plot_path: str):
    """Compute calibration and discrimination metrics, save reliability diagram."""
    AECE, AMCE, _, _, _, _ = AdaptiveBinning(conf_corr_list, True)
    print(f"{label} - ECE (Adaptive Binning): {AECE*100:.2f}%")
    print(f"{label} - MCE (Adaptive Binning): {AMCE*100:.2f}%")

    nce, aucroc, aucpr, aucpr_neg = all_metrics(conf_corr_list)
    print(f"{label} - NCE: {nce:.4f}")
    print(f"{label} - AUC-ROC:   {aucroc:.3f}")
    print(f"{label} - AUC-PR:    {aucpr:.3f}")
    print(f"{label} - AUC-PRNEG: {aucpr_neg:.3f}")

    plt.savefig(plot_path)
    plt.clf()
    print(f"{label} plot saved to {plot_path}")


###############################################################################
#                                   MAIN                                      #
###############################################################################
if __name__ == "__main__":
    # Load dataset
    test_dataset, native_confidence = load_test_dataset(TEST_DATASET_PATH, REMOVE_FEATURES, FILTER)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Ground truth
    _, target = test_dataset.tensors
    correctness = target.squeeze(1).tolist()

    # Model
    input_size = test_dataset.tensors[0].shape[1]
    model = Score_CEM(input_size=input_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Predictions
    predictions = run_inference(model, test_loader)

    # Save combined predictions
    os.makedirs(os.path.join(CEM_PATH, "prediction"), exist_ok=True)
    cem_combined = [[p, c] for p, c in zip(predictions, correctness)]
    native_combined = [[p, c] for p, c in zip(native_confidence, correctness)]

    torch.save(cem_combined, os.path.join(CEM_PATH, "prediction", f"cem_prediction{SUFFIX}.pt"))
    torch.save(native_combined, os.path.join(CEM_PATH, "prediction", f"native_prediction{SUFFIX}.pt"))

    # Metrics
    os.makedirs(os.path.join(CEM_PATH, "plot"), exist_ok=True)
    print("CEM metrics:")
    evaluate_predictions(cem_combined, "CEM", os.path.join(CEM_PATH, "plot", f"rd_cem_token{SUFFIX}.png"))

    print("Native metrics:")
    evaluate_predictions(native_combined, "Native", os.path.join(CEM_PATH, "plot", f"rd_naive_token{SUFFIX}.png"))

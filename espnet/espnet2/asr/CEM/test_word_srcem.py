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
SUFFIX = "_ours_libapt_FT"
CEM_PATH = "/esat/audioslave/yjia/espnet/espnet/espnet2/asr/CEM"

TEST_DATASET_PATH = os.path.join(CEM_PATH, "dataset", "test_dataset_word_ours_libapt_s4.pt")
MODEL_PATH = os.path.join(CEM_PATH, "model", f"cem_model_word{SUFFIX}.pt")

REMOVE_FEATURES = ["confidence"]
DEVICE = "cuda"
BATCH_SIZE = 1024


###############################################################################
#                          DATA PREPARATION                                   #
###############################################################################
def load_test_dataset(path: str, remove_features: list[str]):
    """Load test dataset, drop features, and return TensorDataset + native conf list."""
    dataset: NamedTensorDataset = torch.load(path)

    # Select features
    keep_indices = [i for i, name in enumerate(dataset.feature_names) if name not in remove_features]
    dataset.data = dataset.data[:, keep_indices]
    dataset.feature_names = [dataset.feature_names[i] for i in keep_indices]

    print("Using features:", dataset.feature_names)

    native_confidence = dataset.data[:, 0].tolist()  # assumes first col was native conf
    return TensorDataset(dataset.data, dataset.targets), native_confidence


###############################################################################
#                          MODEL INFERENCE                                    #
###############################################################################
def run_inference(model, dataloader):
    """Run inference and return predictions as a list."""
    preds = []
    model.eval()
    with torch.no_grad():
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(DEVICE)
            outputs = model(batch_data).squeeze(1)
            preds.extend(outputs.cpu().tolist())
    return preds


###############################################################################
#                          METRICS & PLOTTING                                 #
###############################################################################
def evaluate(conf_corr_list, label: str, save_path: str):
    """Compute calibration/discrimination metrics and save plot."""
    AECE, AMCE, _, _, _, _ = AdaptiveBinning(conf_corr_list, True)
    print(f"{label} - ECE (Adaptive Binning): {AECE*100:.2f}%")
    print(f"{label} - MCE (Adaptive Binning): {AMCE*100:.2f}%")

    nce, aucroc, aucpr, aucpr_neg = all_metrics(conf_corr_list)
    print(f"{label} - NCE:      {nce:.4f}")
    print(f"{label} - AUC-ROC:  {aucroc:.3f}")
    print(f"{label} - AUC-PR:   {aucpr:.3f}")
    print(f"{label} - AUC-PRNEG:{aucpr_neg:.3f}")

    plt.savefig(save_path)
    plt.clf()
    print(f"{label} reliability diagram saved to {save_path}")


###############################################################################
#                                   MAIN                                      #
###############################################################################
if __name__ == "__main__":
    # Load dataset
    test_dataset, native_confidence = load_test_dataset(TEST_DATASET_PATH, REMOVE_FEATURES)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Ground truth labels
    _, target = test_dataset.tensors
    correctness = target.squeeze(1).tolist()

    # Model
    input_size = test_dataset.tensors[0].shape[1]
    model = Score_CEM(input_size=input_size).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Predictions
    cem_preds = run_inference(model, test_loader)

    # Combine [confidence, correctness]
    cem_combined = [[p, c] for p, c in zip(cem_preds, correctness)]
    native_combined = [[p, c] for p, c in zip(native_confidence, correctness)]

    # Save predictions
    os.makedirs(os.path.join(CEM_PATH, "prediction"), exist_ok=True)
    torch.save(cem_combined, os.path.join(CEM_PATH, "prediction", f"cem_prediction{SUFFIX}.pt"))
    torch.save(native_combined, os.path.join(CEM_PATH, "prediction", f"native_prediction{SUFFIX}.pt"))

    # Metrics + plots
    os.makedirs(os.path.join(CEM_PATH, "plot"), exist_ok=True)
    evaluate(cem_combined, "CEM", os.path.join(CEM_PATH, "plot", f"rd_cem_word{SUFFIX}.png"))
    evaluate(native_combined, "Native", os.path.join(CEM_PATH, "plot", f"rd_naive_word{SUFFIX}.png"))

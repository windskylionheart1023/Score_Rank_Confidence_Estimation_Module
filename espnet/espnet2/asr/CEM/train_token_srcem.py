#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

from espnet2.asr.CEM.model import NamedTensorDataset, Score_CEM


###############################################################################
#                                CONFIG                                       #
###############################################################################
SUFFIX = "_ours_libapt"
CEM_PATH = "/esat/audioslave/yjia/espnet/espnet/espnet2/asr/CEM"

TRAIN_DATASET_PATH = os.path.join(CEM_PATH, "dataset", "train_dataset_token_ours_libapt_s123.pt")

REMOVE_FEATURES = ["confidence"]
NUM_EPOCHS = 20
VAL_RATIO = 0.2
BATCH_SIZE = 128
DEVICE = "cuda"

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

HIDDEN_SIZE_1 = 16  # currently unused, Score_CEM defines its own arch


###############################################################################
#                          DATASET PREPARATION                                #
###############################################################################
def load_dataset(path: str, remove_features: list[str]):
    """Load dataset, drop features if needed, and return TensorDataset."""
    dataset: NamedTensorDataset = torch.load(path)

    keep_indices = [i for i, name in enumerate(dataset.feature_names) if name not in remove_features]
    dataset.data = dataset.data[:, keep_indices]
    dataset.feature_names = [dataset.feature_names[i] for i in keep_indices]

    print("Using features:", dataset.feature_names)
    print("Dataset shape:", dataset.data.shape)

    return TensorDataset(dataset.data, dataset.targets)


def split_dataset(dataset: TensorDataset, val_ratio: float, seed: int = 42):
    """Split dataset into train and validation subsets."""
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - val_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


###############################################################################
#                          TRAINING UTILITIES                                 #
###############################################################################
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        for batch_data, batch_target in train_loader:
            batch_data, batch_target = batch_data.to(DEVICE), batch_target.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_data.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ---- Validation ----
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_data, val_target in val_loader:
                val_data, val_target = val_data.to(DEVICE), val_target.to(DEVICE)
                val_outputs = model(val_data)
                val_loss = criterion(val_outputs, val_target)
                val_running_loss += val_loss.item() * val_data.size(0)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch+1}: Model updated (val loss {best_val_loss:.4f})")

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    return train_losses, val_losses, best_val_loss


def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


###############################################################################
#                                   MAIN                                      #
###############################################################################
if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset(TRAIN_DATASET_PATH, REMOVE_FEATURES)

    # Split
    train_subset, val_subset = split_dataset(dataset, VAL_RATIO)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")

    # Model
    input_size = dataset.tensors[0].shape[1]
    model = Score_CEM(input_size=input_size).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())} "
          f"(trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)})")

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()

    # Train
    os.makedirs(os.path.join(CEM_PATH, "model"), exist_ok=True)
    model_name = f"cem_model_token{SUFFIX}.pt"
    best_model_path = os.path.join(CEM_PATH, "model", model_name)

    train_losses, val_losses, best_val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        NUM_EPOCHS, best_model_path
    )

    # Plot losses
    os.makedirs(os.path.join(CEM_PATH, "loss"), exist_ok=True)
    plot_path = os.path.join(CEM_PATH, "loss", f"{model_name}_train_val_loss.png")
    plot_losses(train_losses, val_losses, plot_path)

    print(f"Best model saved to {best_model_path} (val loss {best_val_loss:.4f})")

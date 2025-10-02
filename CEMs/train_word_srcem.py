#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from espnet2.asr.CEM.model import NamedTensorDataset, Score_CEM


###############################################################################
#                                CONFIG                                       #
###############################################################################
SUFFIX = "_ours_libapt_FT"
CEM_PATH = "/esat/audioslave/yjia/espnet/espnet/espnet2/asr/CEM"

TRAIN_DATASET_PATH = os.path.join(CEM_PATH, "dataset", "train_dataset_word_ours_libapt_s123.pt")

REMOVE_FEATURES = ["confidence"]
NUM_EPOCHS = 20
VAL_RATIO = 0.2
BATCH_SIZE = 128
DEVICE = "cuda"

LEARNING_RATE = 0.001


###############################################################################
#                          DATASET PREPARATION                                #
###############################################################################
def load_dataset(path: str, remove_features: list[str]) -> TensorDataset:
    """Load dataset, remove features, and return a TensorDataset."""
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


###############################################################################
#                                   MAIN                                      #
###############################################################################
if __name__ == "__main__":
    # Dataset
    dataset = load_dataset(TRAIN_DATASET_PATH, REMOVE_FEATURES)
    train_subset, val_subset = split_dataset(dataset, VAL_RATIO)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Training samples: {len(train_subset)}, Validation samples: {len(val_subset)}")

    # Model
    input_size = dataset.tensors[0].shape[1]
    model = Score_CEM(input_size=input_size).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params} (trainable: {trainable_params})")

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # Train
    os.makedirs(os.path.join(CEM_PATH, "model"), exist_ok=True)
    model_name = f"cem_model_word{SUFFIX}.pt"
    best_model_path = os.path.join(CEM_PATH, "model", model_name)

    train_losses, val_losses, best_val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, best_model_path
    )

    print(f"Best model saved to {best_model_path} (val loss {best_val_loss:.4f})")

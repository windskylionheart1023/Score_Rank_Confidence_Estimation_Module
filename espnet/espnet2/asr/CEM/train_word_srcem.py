#!/usr/bin/env python3
"""Train the word-level SR-CEM (Eq. 10 in the paper).

Identical training loop to ``train_token_srcem.py`` but consumes a word-level
NamedTensorDataset built by ``dataset_word_srcem.py`` (5 features: word
score, max token-rank, prev/after cumulative scores, token count).
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from espnet2.asr.CEM.model import NamedTensorDataset, Score_CEM


def load_dataset(path: str, remove_features) -> TensorDataset:
    dataset: NamedTensorDataset = torch.load(path)

    keep_indices = [
        i for i, name in enumerate(dataset.feature_names) if name not in remove_features
    ]
    dataset.data = dataset.data[:, keep_indices]
    dataset.feature_names = [dataset.feature_names[i] for i in keep_indices]

    print("Using features:", dataset.feature_names)
    print("Dataset shape:", tuple(dataset.data.shape))

    return TensorDataset(dataset.data, dataset.targets)


def split_dataset(dataset: TensorDataset, val_ratio: float, seed: int = 42):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        for batch_data, batch_target in train_loader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_data), batch_target)
            loss.backward()
            optimizer.step()
            running += loss.item() * batch_data.size(0)
        train_loss = running / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for batch_data, batch_target in val_loader:
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)
                running += criterion(model(batch_data), batch_target).item() * batch_data.size(0)
        val_loss = running / len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch + 1}: best model updated (val loss {best_val_loss:.4f})")

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    return train_losses, val_losses, best_val_loss


def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="s", label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("SR-CEM (word) training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", required=True, help="Path to the training .pt dataset.")
    parser.add_argument("--model-out", required=True, help="Where to save the best checkpoint.")
    parser.add_argument("--loss-plot", default=None, help="Optional path to save train/val loss plot.")
    parser.add_argument(
        "--remove-features",
        nargs="*",
        default=["confidence"],
        help="Feature names to drop before training.",
    )
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(args.train_dataset, args.remove_features)
    train_subset, val_subset = split_dataset(dataset, args.val_ratio, args.seed)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    print(f"Training samples: {len(train_subset)}, validation samples: {len(val_subset)}")

    input_size = dataset.tensors[0].shape[1]
    model = Score_CEM(input_size=input_size, hidden_size=args.hidden_size).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params}")

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.BCELoss()

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    train_losses, val_losses, best_val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.num_epochs, args.device, args.model_out,
    )

    if args.loss_plot:
        os.makedirs(os.path.dirname(args.loss_plot) or ".", exist_ok=True)
        plot_losses(train_losses, val_losses, args.loss_plot)

    print(f"Best model saved to {args.model_out} (val loss {best_val_loss:.4f})")


if __name__ == "__main__":
    main()

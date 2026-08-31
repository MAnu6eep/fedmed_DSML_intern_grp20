"""fedmed/core/training.py
Local model training loop, validation engine, and compound loss computation for 3D MRI segmentation.
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Executes a single local training epoch over the hospital data loader."""
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    if total_batches == 0:
        return 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / total_batches


def evaluate_local(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluates local validation partition and computes validation loss and Dice score."""
    model.eval()
    running_loss = 0.0
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    total_batches = len(dataloader)

    if total_batches == 0:
        return 0.0, 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            # Binarize output predictions for Dice scoring (sigmoid > 0.5)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            dice_metric(y_pred=preds, y=labels)

    avg_loss = running_loss / total_batches
    avg_dice = float(dice_metric.aggregate().item())
    dice_metric.reset()

    return avg_loss, avg_dice


def run_local_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Full localized training pipeline for a single federated client round."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)

    train_loss = 0.0
    for _ in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

    val_loss, val_dice = evaluate_local(model, val_loader, loss_fn, device)

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_dice": val_dice,
    }

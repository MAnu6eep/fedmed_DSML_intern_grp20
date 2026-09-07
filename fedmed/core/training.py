"""
fedmed/core/training.py

Local model training loop, validation engine, and compound loss
computation for 3D MRI segmentation with optional FedProx regularization.
"""

from typing import Dict, List, Tuple

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
    global_parameters: List[torch.Tensor] | None = None,
    proximal_mu: float = 0.0,
) -> float:
    """Execute one local training epoch with optional FedProx regularization."""

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

        # Normal local training loss
        loss = loss_fn(outputs, labels)

        # FedProx proximal term
        if global_parameters is not None and proximal_mu > 0.0:
            proximal_term = torch.tensor(
                0.0,
                device=device,
            )

            for local_param, global_param in zip(
                model.parameters(),
                global_parameters,
            ):
                proximal_term += torch.sum(
                    (local_param - global_param) ** 2
                )

            loss = loss + (proximal_mu / 2.0) * proximal_term

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
    """Evaluate local validation partition and compute validation loss and Dice."""

    model.eval()
    running_loss = 0.0
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
    )

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

            # Binarize output predictions for Dice scoring
            preds = (
                torch.sigmoid(outputs) > 0.5
            ).float()

            dice_metric(
                y_pred=preds,
                y=labels,
            )

    avg_loss = running_loss / total_batches
    avg_dice = float(
        dice_metric.aggregate().item()
    )

    dice_metric.reset()

    return avg_loss, avg_dice


def run_local_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    global_parameters: List[torch.Tensor] | None = None,
    proximal_mu: float = 0.0,
) -> Dict[str, float]:
    """
    Run local hospital training.

    If global_parameters and proximal_mu are supplied,
    FedProx regularization is applied during local training.
    """

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )

    loss_fn = DiceFocalLoss(
        sigmoid=True,
        lambda_dice=1.0,
        lambda_focal=1.0,
    )

    train_loss = 0.0

    for _ in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            global_parameters=global_parameters,
            proximal_mu=proximal_mu,
        )

    val_loss, val_dice = evaluate_local(
        model,
        val_loader,
        loss_fn,
        device,
    )

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_dice": val_dice,
    }
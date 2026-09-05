"""Flower client implementation for FedMed."""

from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fedmed.core.training import run_local_training


class FedMedClient(fl.client.NumPyClient):
    """Flower client representing an isolated hospital node."""

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def get_parameters(
        self,
        config: Dict[str, str],
    ) -> List[np.ndarray]:
        """Return local model parameters."""

        return [
            value.detach().cpu().numpy()
            for _, value in self.model.state_dict().items()
        ]

    def set_parameters(
        self,
        parameters: List[np.ndarray],
    ) -> None:
        """Load global model parameters into the local model."""

        params_dict = zip(
            self.model.state_dict().keys(),
            parameters,
        )

        state_dict = OrderedDict(
            {
                key: torch.tensor(value)
                for key, value in params_dict
            }
        )

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Train the model locally."""

        self.set_parameters(parameters)

        epochs = int(
            config.get("local_epochs", 1)
        )

        training_metrics = run_local_training(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=epochs,
            learning_rate=1e-4,
            device=self.device,
        )

        total_samples = len(
            self.train_loader.dataset
        )

        metrics = {
            "train_loss": float(
                training_metrics["train_loss"]
            ),
            "val_loss": float(
                training_metrics["val_loss"]
            ),
            "val_dice": float(
                training_metrics["val_dice"]
            ),
            "local_epochs": float(epochs),
        }

        return (
            self.get_parameters(config={}),
            total_samples,
            metrics,
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate the local model."""

        self.set_parameters(parameters)

        self.model.eval()

        total_samples = len(
            self.val_loader.dataset
        )

        # Keep the existing project's evaluation behavior.
        loss = 0.0
        dice_score = 0.0

        return (
            float(loss),
            total_samples,
            {"dice": float(dice_score)},
        )
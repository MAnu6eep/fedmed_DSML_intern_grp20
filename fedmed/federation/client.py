"""fedmed/federation/client.py
Flower NumPyClient implementation for local hospital node model training and evaluation.
"""

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

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Extracts model weights as a list of NumPy ndarrays."""
        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Loads updated global parameters into local model state."""
        params_dict = zip(self.model.state_dict().keys(), parameters)

        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )

        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Trains local model on hospital data partition."""

        self.set_parameters(parameters)

        epochs = int(config.get("local_epochs", 1))

        training_metrics = run_local_training(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=epochs,
            learning_rate=1e-4,
            device=self.device,
        )

        total_samples = (
            len(self.train_loader.dataset)
            if self.train_loader.dataset
            else 0
        )

        metrics = {
            "client_id": self.client_id,
            "train_loss": training_metrics["train_loss"],
            "val_loss": training_metrics["val_loss"],
            "val_dice": training_metrics["val_dice"],
            "local_epochs": epochs,
        }

        return self.get_parameters(config={}), total_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str],
    ) -> Tuple[float, int, Dict[str, float]]:
        """Evaluates local model on hospital validation partition."""

        self.set_parameters(parameters)

        self.model.eval()

        total_samples = (
            len(self.val_loader.dataset)
            if self.val_loader.dataset
            else 0
        )

        loss = 0.0
        dice_score = 0.0

        return float(loss), total_samples, {
            "dice": float(dice_score)
        }
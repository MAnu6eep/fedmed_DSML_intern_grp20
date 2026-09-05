"""
fedmed/federation/client.py

Flower NumPyClient implementation for a local hospital node.
Each client trains the model on its own local data partition.
"""

from collections import OrderedDict
from typing import Dict, List, Tuple, Union

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
        config: Dict[str, Union[bool, bytes, float, int, str]],
    ) -> List[np.ndarray]:
        """Return the current model parameters as NumPy arrays."""

        return [
            value.detach().cpu().numpy()
            for value in self.model.state_dict().values()
        ]

    def set_parameters(
        self,
        parameters: List[np.ndarray],
    ) -> None:
        """Load global parameters into the local model."""

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
        config: Dict[str, Union[bool, bytes, float, int, str]],
    ) -> Tuple[
        List[np.ndarray],
        int,
        Dict[str, Union[bool, bytes, float, int, str]],
    ]:
        """
        Receive global parameters, train locally,
        and return updated parameters.
        """

        # Load the global model received from the server
        self.set_parameters(parameters)

        # Read number of local epochs
        epochs = int(config.get("local_epochs", 1))

        # Train on this hospital's local data
        training_metrics = run_local_training(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=epochs,
            learning_rate=1e-4,
            device=self.device,
        )

        # Number of local training examples
        total_samples = len(self.train_loader.dataset)

        metrics = {
            "client_id": self.client_id,
            "train_loss": float(training_metrics["train_loss"]),
            "val_loss": float(training_metrics["val_loss"]),
            "val_dice": float(training_metrics["val_dice"]),
            "local_epochs": epochs,
        }

        # Send updated local model back to server
        return (
            self.get_parameters(config={}),
            total_samples,
            metrics,
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Union[bool, bytes, float, int, str]],
    ) -> Tuple[
        float,
        int,
        Dict[str, Union[bool, bytes, float, int, str]],
    ]:
        """
        Evaluate the received global model on the local
        hospital validation dataset.

        NOTE:
        The actual validation calculation will be connected
        once the project's evaluation/loss function is finalized.
        """

        # Load global parameters
        self.set_parameters(parameters)

        # Evaluation mode
        self.model.eval()

        total_samples = len(self.val_loader.dataset)

        # Temporary values until the project's evaluation
        # function is connected here.
        loss = 0.0
        dice_score = 0.0

        return (
            float(loss),
            total_samples,
            {
                "dice": float(dice_score),
            },
        )
"""fedmed/federation/client.py

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

from fedmed.core.evaluation import evaluate_sliding_window
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

    def _get_global_trainable_parameters(
        self,
        parameters: List[np.ndarray],
    ) -> List[torch.Tensor]:
        """
        Extract the global trainable parameters in the same
        order as model.parameters().
        """
        state_keys = list(self.model.state_dict().keys())

        global_state = {
            key: torch.tensor(
                value,
                device=self.device,
            )
            for key, value in zip(
                state_keys,
                parameters,
            )
        }

        return [
            global_state[name].detach().clone()
            for name, _ in self.model.named_parameters()
        ]

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
        # Save the received global model parameters for the FedProx proximal term.
        global_trainable_parameters = self._get_global_trainable_parameters(parameters)

        # Load global parameters into the local model.
        self.set_parameters(parameters)

        # Read local training configuration.
        epochs = int(config.get("local_epochs", 1))
        proximal_mu = float(config.get("proximal_mu", 0.0))

        # Train on this hospital's local data.
        training_metrics = run_local_training(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=epochs,
            learning_rate=1e-4,
            device=self.device,
            global_parameters=global_trainable_parameters,
            proximal_mu=proximal_mu,
        )

        # Number of local training examples.
        total_samples = (
            len(self.train_loader.dataset)
            if self.train_loader and hasattr(self.train_loader, "dataset")
            else 0
        )

        metrics = {
            "client_id": self.client_id,
            "train_loss": float(training_metrics["train_loss"]),
            "val_loss": float(training_metrics["val_loss"]),
            "val_dice": float(training_metrics["val_dice"]),
            "local_epochs": epochs,
            "proximal_mu": proximal_mu,
        }

        # Send updated local model back to the server.
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
        """
        # Load global parameters.
        self.set_parameters(parameters)

        # Evaluation mode.
        self.model.eval()

        # Number of local validation examples.
        total_samples = (
            len(self.val_loader.dataset)
            if self.val_loader and hasattr(self.val_loader, "dataset")
            else 0
        )

        # Execute 3D sliding-window evaluation over local hospital validation loader.
        if self.val_loader and len(self.val_loader) > 0:
            val_results = evaluate_sliding_window(
                model=self.model,
                dataloader=self.val_loader,
                device=self.device,
            )
            loss = val_results.get("val_loss", 0.0)
            dice_score = val_results.get("dice", 0.0)
            iou_score = val_results.get("iou", 0.0)
        else:
            loss = 0.0
            dice_score = 0.0
            iou_score = 0.0

        return (
            float(loss),
            total_samples,
            {
                "dice": float(dice_score),
                "iou": float(iou_score),
                "loss": float(loss),
            },
        )
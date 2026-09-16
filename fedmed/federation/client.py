"""fedmed/federation/client.py

Flower NumPyClient implementation for a local hospital node.

Supports:
    - FedAvg
    - FedProx
    - SCAFFOLD

FedAvg and FedProx behaviour remains unchanged when SCAFFOLD
is not enabled.
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
from fedmed.federation.scaffold import (
    ClientControlVariate,
    deserialize_control_variate,
    serialize_control_variate,
)


Config = Dict[str, Union[bool, bytes, float, int, str]]


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

        # Each hospital maintains its own SCAFFOLD
        # client control variate c_i.
        self.scaffold_control = ClientControlVariate(
            client_id=client_id
        )

    # ------------------------------------------------------------------
    # Model parameters
    # ------------------------------------------------------------------

    def get_parameters(
        self,
        config: Config,
    ) -> List[np.ndarray]:
        """Return the complete model state as NumPy arrays."""

        return [
            value.detach()
            .cpu()
            .numpy()
            .copy()
            for value in self.model.state_dict().values()
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

    def _get_global_trainable_parameters(
        self,
        parameters: List[np.ndarray],
    ) -> List[torch.Tensor]:
        """Extract received global trainable parameters."""

        state_keys = list(
            self.model.state_dict().keys()
        )

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
            global_state[name]
            .detach()
            .clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]

    def _get_model_trainable_parameters(
        self,
    ) -> List[np.ndarray]:
        """Return local trainable parameters only."""

        return [
            parameter.detach()
            .cpu()
            .numpy()
            .copy()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

    # ------------------------------------------------------------------
    # SCAFFOLD
    # ------------------------------------------------------------------

    def _initialize_scaffold(
        self,
        global_trainable_parameters: List[np.ndarray],
    ) -> None:
        """Initialize the client control variate if necessary."""

        if not self.scaffold_control.is_initialized():
            self.scaffold_control.initialize(
                global_trainable_parameters
            )

    def _receive_server_control(
        self,
        config: Config,
    ) -> List[np.ndarray]:
        """Deserialize the server control variate c."""

        server_control_data = config.get(
            "server_control_variate"
        )

        if isinstance(
            server_control_data,
            bytes,
        ):
            return deserialize_control_variate(
                server_control_data
            )

        if isinstance(
            server_control_data,
            bytearray,
        ):
            return deserialize_control_variate(
                bytes(server_control_data)
            )

        raise ValueError(
            "SCAFFOLD requires "
            "'server_control_variate' to be "
            "transmitted as bytes."
        )

    def _update_scaffold_client_control(
        self,
        global_trainable_parameters: List[np.ndarray],
        local_trainable_parameters: List[np.ndarray],
        server_control: List[np.ndarray],
        epochs: int,
        learning_rate: float,
    ) -> None:
        """Update c_i after local SCAFFOLD training."""

        batches_per_epoch = len(
            self.train_loader
        )

        local_steps = (
            batches_per_epoch * epochs
        )

        if local_steps <= 0:
            return

        self.scaffold_control.update(
            global_parameters=(
                global_trainable_parameters
            ),
            local_parameters=(
                local_trainable_parameters
            ),
            server_control_variate=server_control,
            learning_rate=learning_rate,
            local_steps=local_steps,
        )

    # ------------------------------------------------------------------
    # Flower fit
    # ------------------------------------------------------------------

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Config,
    ) -> Tuple[
        List[np.ndarray],
        int,
        Dict[str, Union[bool, bytes, float, int, str]],
    ]:
        """Receive global parameters, train locally, and return updates."""

        # --------------------------------------------------------------
        # Determine whether SCAFFOLD is enabled.
        # --------------------------------------------------------------

        scaffold_enabled = bool(
            config.get(
                "scaffold",
                False,
            )
        )

        # --------------------------------------------------------------
        # Extract global trainable parameters.
        #
        # These are needed by FedProx and SCAFFOLD.
        # --------------------------------------------------------------

        global_trainable_tensors = (
            self._get_global_trainable_parameters(
                parameters
            )
        )

        global_trainable_parameters = [
            parameter.detach()
            .cpu()
            .numpy()
            .copy()
            for parameter in global_trainable_tensors
        ]

        # --------------------------------------------------------------
        # Load global model.
        # --------------------------------------------------------------

        self.set_parameters(parameters)

        # --------------------------------------------------------------
        # Training configuration.
        # --------------------------------------------------------------

        epochs = int(
            config.get(
                "local_epochs",
                1,
            )
        )

        learning_rate = float(
            config.get(
                "learning_rate",
                1e-4,
            )
        )

        proximal_mu = float(
            config.get(
                "proximal_mu",
                0.0,
            )
        )

        # --------------------------------------------------------------
        # SCAFFOLD setup.
        # --------------------------------------------------------------

        server_control = None

        if scaffold_enabled:
            # Initialize c_i using only trainable parameters.
            self._initialize_scaffold(
                global_trainable_parameters
            )

            # Receive global server control variate c.
            server_control = (
                self._receive_server_control(
                    config
                )
            )

            if len(server_control) != len(
                global_trainable_parameters
            ):
                raise ValueError(
                    "SCAFFOLD server control variate "
                    "does not match the number of "
                    "trainable model parameters."
                )

        # --------------------------------------------------------------
        # Local training.
        #
        # FedAvg:
        #   scaffold controls are None and normal training occurs.
        #
        # FedProx:
        #   proximal_mu > 0 adds FedProx regularization.
        #
        # SCAFFOLD:
        #   server/client control variates correct gradients.
        # --------------------------------------------------------------

        training_metrics = run_local_training(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=self.device,
            global_parameters=(
                global_trainable_tensors
            ),
            proximal_mu=proximal_mu,
            scaffold_server_control=(
                server_control
            ),
            scaffold_client_control=(
                self.scaffold_control.get_values()
                if scaffold_enabled
                else None
            ),
        )

        # --------------------------------------------------------------
        # Complete local model state.
        #
        # Flower still returns the complete model state because that
        # is what the existing FedAvg/FedProx pipeline expects.
        # --------------------------------------------------------------

        local_parameters = self.get_parameters(
            config={}
        )

        # --------------------------------------------------------------
        # Update client control variate c_i.
        # --------------------------------------------------------------

        if scaffold_enabled:
            local_trainable_parameters = (
                self._get_model_trainable_parameters()
            )

            self._update_scaffold_client_control(
                global_trainable_parameters=(
                    global_trainable_parameters
                ),
                local_trainable_parameters=(
                    local_trainable_parameters
                ),
                server_control=server_control,
                epochs=epochs,
                learning_rate=learning_rate,
            )

        # --------------------------------------------------------------
        # Number of local training examples.
        # --------------------------------------------------------------

        total_samples = (
            len(
                self.train_loader.dataset
            )
            if (
                self.train_loader
                and hasattr(
                    self.train_loader,
                    "dataset",
                )
            )
            else 0
        )

        # --------------------------------------------------------------
        # Training metrics.
        # --------------------------------------------------------------

        metrics: Dict[
            str,
            Union[bool, bytes, float, int, str],
        ] = {
            "client_id": self.client_id,
            "train_loss": float(
                training_metrics[
                    "train_loss"
                ]
            ),
            "val_loss": float(
                training_metrics[
                    "val_loss"
                ]
            ),
            "val_dice": float(
                training_metrics[
                    "val_dice"
                ]
            ),
            "local_epochs": epochs,
            "proximal_mu": proximal_mu,
        }

        # --------------------------------------------------------------
        # Return SCAFFOLD control variate to server.
        # --------------------------------------------------------------

        if scaffold_enabled:
            metrics.update(
                {
                    "scaffold": True,
                    "scaffold_client_id": (
                        self.client_id
                    ),
                    "scaffold_control_initialized": (
                        self.scaffold_control
                        .is_initialized()
                    ),
                    "scaffold_control_variate": (
                        serialize_control_variate(
                            self.scaffold_control
                            .get_values()
                        )
                    ),
                }
            )

        return (
            local_parameters,
            total_samples,
            metrics,
        )

    # ------------------------------------------------------------------
    # Flower evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Config,
    ) -> Tuple[
        float,
        int,
        Dict[str, Union[bool, bytes, float, int, str]],
    ]:
        """Evaluate the received global model."""

        self.set_parameters(parameters)

        self.model.eval()

        total_samples = (
            len(
                self.val_loader.dataset
            )
            if (
                self.val_loader
                and hasattr(
                    self.val_loader,
                    "dataset",
                )
            )
            else 0
        )

        if (
            self.val_loader
            and len(self.val_loader) > 0
        ):
            val_results = (
                evaluate_sliding_window(
                    model=self.model,
                    dataloader=self.val_loader,
                    device=self.device,
                )
            )

            loss = val_results.get(
                "val_loss",
                0.0,
            )

            dice_score = val_results.get(
                "dice",
                0.0,
            )

            iou_score = val_results.get(
                "iou",
                0.0,
            )

        else:
            loss = 0.0
            dice_score = 0.0
            iou_score = 0.0

        return (
            float(loss),
            total_samples,
            {
                "dice": float(
                    dice_score
                ),
                "iou": float(
                    iou_score
                ),
                "loss": float(
                    loss
                ),
            },
        )
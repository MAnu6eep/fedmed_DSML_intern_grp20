"""fedmed/federation/server.py

Flower server configuration for federated model training.
"""

from collections import OrderedDict
from typing import List

import flwr as fl
import numpy as np
import torch
import torch.nn as nn


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as NumPy arrays."""
    return [
        value.detach().cpu().numpy()
        for value in model.state_dict().values()
    ]


def set_parameters(
    model: nn.Module,
    parameters: List[np.ndarray],
) -> None:
    """Load NumPy parameters into the model."""
    params_dict = zip(model.state_dict().keys(), parameters)

    state_dict = OrderedDict(
        {
            key: torch.tensor(value)
            for key, value in params_dict
        }
    )

    model.load_state_dict(state_dict, strict=True)


def create_strategy() -> fl.server.strategy.FedAvg:
    """Create the baseline FedAvg strategy."""

    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )


def create_server_config() -> fl.server.ServerConfig:
    """Create the Flower server configuration."""

    return fl.server.ServerConfig(
        num_rounds=20,
    )
"""fedmed/federation/server.py

Flower server configuration for federated model training.
"""
from fedmed.privacy.secagg_config import SecAggPlusConfig
from fedmed.privacy.secure_aggregation import (
    SecureAggregationManager,
)
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
    """Create the baseline FedAvg strategy with SecAgg+ requirements."""

    secagg_config = SecAggPlusConfig(
        num_clients=3,
        threshold=2,
    )

    secagg_manager = SecureAggregationManager(secagg_config)

    requirements = secagg_manager.get_round_requirements()

    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=requirements["num_clients"],
        min_evaluate_clients=requirements["num_clients"],
        min_available_clients=requirements["num_clients"],
    )


def create_server_config() -> fl.server.ServerConfig:
    """Create the Flower server configuration."""

    return fl.server.ServerConfig(
        num_rounds=20,
    )
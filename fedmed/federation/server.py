"""
fedmed/federation/server.py

Flower server configuration for federated model training
using the FedAvg strategy.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn

from flwr.common import (
    Metrics,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
)
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg


def weighted_average_metrics(
    metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """Aggregate validation Dice scores using sample counts."""

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {}

    weighted_dice = sum(
        num_examples * float(metric.get("dice", 0.0))
        for num_examples, metric in metrics
    )

    return {
        "val_dice": weighted_dice / total_examples
    }


def get_parameters(
    model: nn.Module,
) -> List[np.ndarray]:
    """Extract model state as NumPy arrays."""

    return [
        value.detach().cpu().numpy()
        for value in model.state_dict().values()
    ]


def set_parameters(
    model: nn.Module,
    parameters: List[np.ndarray],
) -> None:
    """Load NumPy parameters into a PyTorch model."""

    params_dict = zip(
        model.state_dict().keys(),
        parameters,
    )

    state_dict = OrderedDict(
        {
            key: torch.tensor(value)
            for key, value in params_dict
        }
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )


def get_initial_parameters(
    model: nn.Module,
) -> Parameters:
    """
    Convert the initial PyTorch model parameters
    into Flower Parameters.
    """

    ndarrays: NDArrays = get_parameters(model)

    return ndarrays_to_parameters(ndarrays)


def create_server_strategy(
    initial_parameters: Optional[Parameters] = None,
    fraction_fit: float = 1.0,
    min_fit_clients: int = 3,
    min_available_clients: int = 3,
) -> FedAvg:
    """Create the FedAvg strategy."""

    return FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_available_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )


def create_server_config(
    num_rounds: int = 20,
) -> fl.server.ServerConfig:
    """Create Flower server configuration."""

    return fl.server.ServerConfig(
        num_rounds=num_rounds,
    )


def build_server_app(
    strategy: Optional[fl.server.strategy.Strategy] = None,
    num_rounds: int = 20,
) -> ServerApp:
    """
    Build the Flower ServerApp.

    The actual model and initial parameters should be supplied
    by the project entry point.
    """

    if strategy is None:
        strategy = create_server_strategy()

    config = create_server_config(
        num_rounds=num_rounds,
    )

    return ServerApp(
        strategy=strategy,
        config=config,
    )
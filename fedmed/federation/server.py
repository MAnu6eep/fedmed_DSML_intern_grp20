"""fedmed/federation/server.py

Flower server configuration for federated model training.
"""

from collections import OrderedDict
from typing import List

import flwr as fl
import numpy as np
import torch
import torch.nn as nn

from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import (
    Metrics,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

import numpy as np
import torch
import torch.nn as nn


def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates local validation metrics across hospital nodes."""
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    weighted_dice = sum(
        num_examples * float(m.get("dice", 0.0))
        for num_examples, m in metrics
    )

    return {"val_dice": weighted_dice / total_examples}

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

def get_initial_parameters(model: nn.Module) -> Parameters:
    """Extracts PyTorch model weights and converts them to Flower Parameters."""
    ndarrays: NDArrays = [
        val.cpu().numpy()
        for _, val in model.state_dict().items()
    ]

    return ndarrays_to_parameters(ndarrays)



def create_server_strategy(
    initial_parameters: Optional[Parameters] = None,
    fraction_fit: float = 1.0,
    min_fit_clients: int = 3,
    min_available_clients: int = 3,
) -> FedAvg:
    """Instantiates the baseline FedAvg aggregation strategy."""

    return FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_available_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )

def create_server_config() -> fl.server.ServerConfig:
    """Create the Flower server configuration."""

    return fl.server.ServerConfig(
        num_rounds=20,
    )

def build_server_app(
    strategy: Optional[fl.server.strategy.Strategy] = None,
    num_rounds: int = 20,
) -> ServerApp:
    """Constructs the Flower ServerApp instance ready for execution."""

    strat = strategy or create_server_strategy()
    config = ServerConfig(num_rounds=num_rounds)

    return ServerApp(strategy=strat, config=config)
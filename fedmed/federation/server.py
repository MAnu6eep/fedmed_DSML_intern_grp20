"""fedmed/federation/server.py

Flower server configuration for federated model training
using FedAvg, FedProx, and SecAgg+ privacy-preserving workflows.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
import torch
import torch.nn as nn

from flwr.common import (
    Context,
    Metrics,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
)

from flwr.server import (
    Grid,
    LegacyContext,
    ServerApp,
    ServerConfig,
)

from flwr.server.strategy import FedAvg, FedProx
from flwr.server.workflow import (
    DefaultWorkflow,
    SecAggPlusWorkflow,
)

from fedmed.core.model import get_model
from fedmed.privacy.secagg_config import SecAggPlusConfig
from fedmed.privacy.secure_aggregation import SecureAggregationManager


def weighted_average_metrics(
    metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """Aggregate local validation metrics across hospital nodes."""
    total_examples = sum(
        num_examples for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    metric_keys = set()

    for _, metric in metrics:
        for key, value in metric.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metric_keys.add(key)

    for key in metric_keys:
        weighted_sum = sum(
            num_examples * float(metric.get(key, 0.0))
            for num_examples, metric in metrics
            if key in metric
            and isinstance(metric[key], (int, float))
            and not isinstance(metric[key], bool)
        )
        aggregated[key] = round(
            weighted_sum / total_examples,
            6,
        )

    return aggregated


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
    """Convert model parameters into Flower Parameters."""
    ndarrays: NDArrays = get_parameters(model)
    return ndarrays_to_parameters(ndarrays)


def create_secagg_config() -> SecAggPlusConfig:
    """Create the project SecAgg+ configuration."""
    return SecAggPlusConfig(
        num_clients=3,
        threshold=2,
        modulus_range=2**31,
        clipping_bound=10.0,
        quantization_bits=16,
        enable_dropouts=True,
    )


def create_secagg_requirements() -> Dict[str, int]:
    """Create SecAgg+ client participation requirements."""
    secagg_config = create_secagg_config()
    secagg_manager = SecureAggregationManager(secagg_config)
    return secagg_manager.get_round_requirements()


def create_strategy(
    initial_parameters: Optional[Parameters] = None,
    evaluate_fn: Optional[Callable] = None,
) -> FedAvg:
    """Create the FedAvg strategy used by the SecAgg+ workflow."""
    config = create_secagg_config()

    return FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.num_clients,
        min_evaluate_clients=config.num_clients,
        min_available_clients=config.num_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
        fit_metrics_aggregation_fn=weighted_average_metrics,
    )


def create_fedprox_strategy(
    proximal_mu: float = 0.01,
    initial_parameters: Optional[Parameters] = None,
    fraction_fit: float = 1.0,
    min_fit_clients: int = 3,
    min_available_clients: int = 3,
    evaluate_fn: Optional[Callable] = None,
) -> FedProx:
    """Create a configurable FedProx strategy for non-IID datasets."""
    if proximal_mu < 0:
        raise ValueError("proximal_mu must be non-negative.")

    return FedProx(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_available_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
        fit_metrics_aggregation_fn=weighted_average_metrics,
        proximal_mu=proximal_mu,
    )


def create_secagg_workflow() -> SecAggPlusWorkflow:
    """Create the Flower SecAgg+ workflow."""
    config = create_secagg_config()

    return SecAggPlusWorkflow(
        num_shares=config.num_clients,
        reconstruction_threshold=config.threshold,
        clipping_range=config.clipping_bound,
        modulus_range=config.modulus_range,
        quantization_range=2**config.quantization_bits,
    )


def create_server_config(
    num_rounds: int = 20,
) -> fl.server.ServerConfig:
    """Create the Flower server configuration."""
    return fl.server.ServerConfig(
        num_rounds=num_rounds,
    )


app = ServerApp()


@app.main()
def main(
    grid: Grid,
    context: Context,
) -> None:
    """Run federated learning with the project's SecAgg+ workflow."""
    secagg_config = create_secagg_config()

    # Create the initial global model
    model = get_model(
        in_channels=4,
        out_channels=1,
    )

    initial_parameters = get_initial_parameters(model)

    strategy = create_strategy(
        initial_parameters=initial_parameters,
    )

    num_rounds = int(
        context.run_config.get(
            "num-server-rounds",
            2,
        )
    )

    # SecAgg+ is implemented as a workflow around FedAvg
    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(
            num_rounds=num_rounds,
        ),
        strategy=strategy,
    )

    workflow = DefaultWorkflow(
        fit_workflow=create_secagg_workflow(),
    )

    print("\n===== FedMed SecAgg+ Configuration =====")
    print(f"Clients: {secagg_config.num_clients}")
    print(f"Threshold: {secagg_config.threshold}")
    print(f"Clipping bound: {secagg_config.clipping_bound}")
    print(f"Quantization bits: {secagg_config.quantization_bits}")
    print(f"Dropout recovery: {secagg_config.enable_dropouts}")
    print(f"Federated rounds: {num_rounds}")
    print("========================================\n")

    # Execute the SecAgg+ workflow
    workflow(
        grid,
        legacy_context,
    )


def build_server_app(
    strategy: Optional[fl.server.strategy.Strategy] = None,
    num_rounds: int = 20,
) -> ServerApp:
    """Build a Flower ServerApp for callers that use the helper API."""
    if strategy is None:
        strategy = create_strategy()

    config = create_server_config(
        num_rounds=num_rounds,
    )

    return ServerApp(
        strategy=strategy,
        config=config,
    )
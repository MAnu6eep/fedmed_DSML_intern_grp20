"""Flower ServerApp with SecAgg+ secure aggregation."""

from collections import OrderedDict
from typing import List, Optional, Tuple

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
from flwr.server.strategy import FedAvg
from flwr.server.workflow import (
    DefaultWorkflow,
    SecAggPlusWorkflow,
)

from fedmed.core.model import get_model
from fedmed.privacy.secagg_config import (
    SecAggPlusConfig,
)


def weighted_average_metrics(
    metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """Aggregate local validation metrics."""

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {}

    weighted_dice = sum(
        num_examples
        * float(metric.get("dice", 0.0))
        for num_examples, metric in metrics
    )

    return {
        "val_dice": weighted_dice / total_examples,
    }


def get_parameters(
    model: nn.Module,
) -> List[np.ndarray]:
    """Extract model parameters."""

    return [
        value.detach().cpu().numpy()
        for value in model.state_dict().values()
    ]


def set_parameters(
    model: nn.Module,
    parameters: List[np.ndarray],
) -> None:
    """Load parameters into a model."""

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
    """Convert model parameters to Flower Parameters."""

    ndarrays: NDArrays = [
        value.cpu().numpy()
        for _, value in model.state_dict().items()
    ]

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


def create_secagg_workflow() -> SecAggPlusWorkflow:
    """Create Flower SecAgg+ workflow."""

    config = create_secagg_config()

    return SecAggPlusWorkflow(
        num_shares=config.num_clients,
        reconstruction_threshold=config.threshold,
        clipping_range=config.clipping_bound,
        modulus_range=config.modulus_range,
        quantization_range=2**config.quantization_bits,
    )


def create_strategy(
    initial_parameters: Optional[Parameters] = None,
) -> FedAvg:
    """Create FedAvg strategy used by SecAgg+."""

    config = create_secagg_config()

    return FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.num_clients,
        min_evaluate_clients=config.num_clients,
        min_available_clients=config.num_clients,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=(
            weighted_average_metrics
        ),
    )


app = ServerApp()


@app.main()
def main(
    grid: Grid,
    context: Context,
) -> None:
    """Run federated learning with SecAgg+."""

    secagg_config = create_secagg_config()

    # Create initial global model.
    model = get_model(
        in_channels=4,
        out_channels=1,
    )

    initial_parameters = get_initial_parameters(
        model
    )

    strategy = create_strategy(
        initial_parameters=initial_parameters
    )

    num_rounds = int(
        context.run_config.get(
            "num-server-rounds",
            2,
        )
    )

    # SecAgg+ is implemented as a workflow
    # around the existing FedAvg strategy.
    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(
            num_rounds=num_rounds
        ),
        strategy=strategy,
    )

    workflow = DefaultWorkflow(
        fit_workflow=create_secagg_workflow()
    )

    print(
        "\n===== FedMed SecAgg+ Configuration ====="
    )
    print(
        f"Clients: {secagg_config.num_clients}"
    )
    print(
        f"Threshold: {secagg_config.threshold}"
    )
    print(
        f"Clipping bound: "
        f"{secagg_config.clipping_bound}"
    )
    print(
        f"Quantization bits: "
        f"{secagg_config.quantization_bits}"
    )
    print(
        f"Dropout recovery: "
        f"{secagg_config.enable_dropouts}"
    )
    print(
        f"Federated rounds: {num_rounds}"
    )
    print(
        "========================================\n"
    )

    # THIS actually executes SecAgg+.
    workflow(
        grid,
        legacy_context,
    )
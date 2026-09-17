"""fedmed/federation/server.py

Flower server configuration for FedAvg, FedProx, SCAFFOLD,
SecAgg+, and selective CKKS encrypted model updates.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

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
    parameters_to_ndarrays,
    FitIns,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedProx
from flwr.server import (
    LegacyContext,
    ServerApp,
    ServerConfig,
)
from flwr.server.workflow import (
    DefaultWorkflow,
    SecAggPlusWorkflow,
)

from fedmed.core.model import get_model
from fedmed.federation.scaffold import (
    SCAFFOLDConfig,
    ControlVariateState,
    deserialize_control_variate,
    serialize_control_variate,
)
from fedmed.privacy.encrypted_update import (
    aggregate_encrypted_parameters,
    deserialize_encrypted_update,
)
from fedmed.privacy.tenseal_engine import TenSEALEngine


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def weighted_average_metrics(
    metrics: List[Tuple[int, Metrics]],
) -> Metrics:
    """Aggregate numerical metrics using example-weighted averages."""

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {}

    aggregated_metrics: Metrics = {}
    metric_keys = set()

    for _, metric in metrics:
        for key, value in metric.items():
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                metric_keys.add(key)

    for key in metric_keys:
        weighted_sum = 0.0
        valid_metric = False

        for num_examples, metric in metrics:
            value = metric.get(key)

            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                weighted_sum += (
                    num_examples * float(value)
                )
                valid_metric = True

        if valid_metric:
            aggregated_metrics[key] = round(
                weighted_sum / total_examples,
                6,
            )

    return aggregated_metrics


# ---------------------------------------------------------------------------
# Model parameter helpers
# ---------------------------------------------------------------------------


def get_parameters(
    model: nn.Module,
) -> List[np.ndarray]:
    """Extract complete model state as NumPy arrays."""

    return [
        value.detach()
        .cpu()
        .numpy()
        .copy()
        for value in model.state_dict().values()
    ]


def get_trainable_parameters(
    model: nn.Module,
) -> List[np.ndarray]:
    """Extract only trainable model parameters."""

    return [
        parameter.detach()
        .cpu()
        .numpy()
        .copy()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def set_parameters(
    model: nn.Module,
    parameters: List[np.ndarray],
) -> None:
    """Load NumPy model state into PyTorch model."""

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

    return ndarrays_to_parameters(
        get_parameters(model)
    )


# ---------------------------------------------------------------------------
# Selective encrypted aggregation
# ---------------------------------------------------------------------------


class EncryptedFedAvgStrategy(FedAvg):
    """FedAvg strategy with selective CKKS encrypted aggregation.

    Normal model parameters continue through the existing Flower
    FedAvg path.

    Selected parameters are additionally received as serialized
    CKKS
    ciphertexts through the fit metrics payload and aggregated on the
    server without decrypting them.
    """

    def __init__(
        self,
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            **kwargs,
        )

        self.tenseal_engine = TenSEALEngine()

        # Stores the most recent aggregated ciphertexts.
        #
        # The values remain encrypted. No secret-key decryption is
        # performed by the server during aggregation.
        self.aggregated_encrypted_parameters: Dict[
            str,
            bytes,
        ] = {}

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate plaintext and encrypted client updates."""

        # --------------------------------------------------------------
        # Existing FedAvg aggregation
        # --------------------------------------------------------------
        #
        # This keeps the existing model update path working.
        # --------------------------------------------------------------

        aggregated_parameters, metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures,
            )
        )

        # --------------------------------------------------------------
        # Collect encrypted updates from clients.
        # --------------------------------------------------------------

        encrypted_updates = []

        for _, fit_res in results:
            encrypted_payload = fit_res.metrics.get(
                "encrypted_payload"
            )

            encrypted_enabled = fit_res.metrics.get(
                "encrypted_update",
                False,
            )

            if (
                encrypted_enabled
                and isinstance(
                    encrypted_payload,
                    bytes,
                )
            ):
                encrypted_update = (
                    deserialize_encrypted_update(
                        encrypted_payload
                    )
                )

                encrypted_updates.append(
                    encrypted_update
                )

        # --------------------------------------------------------------
        # No encrypted updates in this round.
        # --------------------------------------------------------------

        if not encrypted_updates:
            return (
                aggregated_parameters,
                metrics,
            )

        # --------------------------------------------------------------
        # Aggregate CKKS ciphertexts.
        #
        # IMPORTANT:
        #
        # aggregate_encrypted_parameters() performs ciphertext
        # addition only. It does NOT decrypt the values.
        # --------------------------------------------------------------

        aggregated_encrypted = (
            aggregate_encrypted_parameters(
                engine=self.tenseal_engine,
                encrypted_updates=encrypted_updates,
            )
        )

        self.aggregated_encrypted_parameters = (
            aggregated_encrypted
        )

        # --------------------------------------------------------------
        # Add metadata to server metrics.
        # --------------------------------------------------------------

        metrics = dict(
            metrics or {}
        )

        metrics.update(
            {
                "encrypted_update": True,
                "encrypted_parameter_count": len(
                    aggregated_encrypted
                ),
                "encrypted_aggregation_round": (
                    server_round
                ),
            }
        )

        return (
            aggregated_parameters,
            metrics,
        )


def create_encrypted_strategy(
    initial_parameters: Optional[Parameters] = None,
    evaluate_fn: Optional[Callable] = None,
) -> EncryptedFedAvgStrategy:
    """Create FedAvg with selective CKKS aggregation."""

    config = create_secagg_config()

    return EncryptedFedAvgStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.num_clients,
        min_evaluate_clients=config.num_clients,
        min_available_clients=config.num_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=(
            weighted_average_metrics
        ),
        fit_metrics_aggregation_fn=(
            weighted_average_metrics
        ),
    )


# ---------------------------------------------------------------------------
# SecAgg+
# ---------------------------------------------------------------------------


def create_secagg_config():
    """Create the project SecAgg+ configuration."""

    from fedmed.privacy.secagg_config import (
        SecAggPlusConfig,
    )

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

    from fedmed.privacy.secagg_config import (
        SecAggPlusConfig,
    )
    from fedmed.privacy.secure_aggregation import (
        SecureAggregationManager,
    )

    secagg_config = SecAggPlusConfig(
        num_clients=3,
        threshold=2,
    )

    secagg_manager = SecureAggregationManager(
        secagg_config
    )

    return secagg_manager.get_round_requirements()


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------


def create_strategy(
    initial_parameters: Optional[Parameters] = None,
    evaluate_fn: Optional[Callable] = None,
) -> FedAvg:
    """Create the existing FedAvg strategy."""

    config = create_secagg_config()

    return FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.num_clients,
        min_evaluate_clients=config.num_clients,
        min_available_clients=config.num_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=(
            weighted_average_metrics
        ),
        fit_metrics_aggregation_fn=(
            weighted_average_metrics
        ),
    )


# ---------------------------------------------------------------------------
# SCAFFOLD
# ---------------------------------------------------------------------------


class SCAFFOLDStrategy(FedAvg):
    """Flower FedAvg strategy extended with SCAFFOLD control variates."""

    def __init__(
        self,
        initial_parameters: Optional[Parameters] = None,
        scaffold_config: Optional[SCAFFOLDConfig] = None,
        trainable_parameters: Optional[NDArrays] = None,
        **kwargs,
    ):
        config = (
            scaffold_config
            or SCAFFOLDConfig()
        )

        super().__init__(
            fraction_fit=config.fraction_fit,
            fraction_evaluate=config.fraction_evaluate,
            min_fit_clients=config.min_fit_clients,
            min_evaluate_clients=(
                config.min_evaluate_clients
            ),
            min_available_clients=(
                config.min_available_clients
            ),
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=(
                weighted_average_metrics
            ),
            evaluate_metrics_aggregation_fn=(
                weighted_average_metrics
            ),
            **kwargs,
        )

        self.scaffold_config = config

        self.control_variates = (
            ControlVariateState()
        )

        if trainable_parameters is not None:
            self.control_variates.initialize(
                parameters=trainable_parameters
            )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ):
        """Send global model and server control variate to clients."""

        client_config_pairs = (
            super().configure_fit(
                server_round,
                parameters,
                client_manager,
            )
        )

        server_control = (
            self.control_variates.server
        )

        if not server_control.is_initialized():
            model = get_model(
                in_channels=4,
                out_channels=1,
            )

            server_control.initialize(
                get_trainable_parameters(model)
            )

        control_bytes = serialize_control_variate(
            server_control.get_values()
        )

        configured = []

        for client, fit_ins in client_config_pairs:
            config = dict(
                fit_ins.config
            )

            config.update(
                {
                    "scaffold": True,
                    "learning_rate": (
                        self.scaffold_config.learning_rate
                    ),
                    "local_epochs": (
                        self.scaffold_config.local_epochs
                    ),
                    "server_control_variate": (
                        control_bytes
                    ),
                }
            )

            configured.append(
                (
                    client,
                    FitIns(
                        parameters=fit_ins.parameters,
                        config=config,
                    ),
                )
            )

        return configured

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate model parameters and update server control variate."""

        aggregated_parameters, metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures,
            )
        )

        if aggregated_parameters is None:
            return None, metrics

        client_ids = []
        client_control_variates = []

        for _, fit_res in results:
            client_id = fit_res.metrics.get(
                "scaffold_client_id"
            )

            control_bytes = fit_res.metrics.get(
                "scaffold_control_variate"
            )

            if (
                isinstance(client_id, str)
                and isinstance(control_bytes, bytes)
            ):
                client_cv = (
                    deserialize_control_variate(
                        control_bytes
                    )
                )

                client_ids.append(client_id)
                client_control_variates.append(
                    client_cv
                )

        if client_control_variates:
            self.control_variates.server.update(
                client_control_variates=(
                    client_control_variates
                ),
                learning_rate=(
                    self.scaffold_config
                    .server_learning_rate
                ),
            )

        metrics = dict(
            metrics
        )

        metrics.update(
            {
                "scaffold": True,
                "scaffold_round": server_round,
                "scaffold_clients": len(
                    client_control_variates
                ),
            }
        )

        return aggregated_parameters, metrics


def create_scaffold_strategy(
    initial_parameters: Optional[Parameters] = None,
    evaluate_fn: Optional[Callable] = None,
) -> SCAFFOLDStrategy:
    """Create the SCAFFOLD strategy."""

    model = get_model(
        in_channels=4,
        out_channels=1,
    )

    trainable_parameters = (
        get_trainable_parameters(model)
    )

    config = SCAFFOLDConfig(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        local_epochs=1,
        learning_rate=1e-4,
        server_learning_rate=1.0,
        client_learning_rate=1.0,
    )

    return SCAFFOLDStrategy(
        initial_parameters=initial_parameters,
        scaffold_config=config,
        trainable_parameters=trainable_parameters,
        evaluate_fn=evaluate_fn,
    )


# ---------------------------------------------------------------------------
# FedProx
# ---------------------------------------------------------------------------


def create_fedprox_strategy(
    proximal_mu: float = 0.01,
    initial_parameters: Optional[Parameters] = None,
    fraction_fit: float = 1.0,
    min_fit_clients: int = 3,
    min_available_clients: int = 3,
    evaluate_fn: Optional[Callable] = None,
) -> FedProx:
    """Create the existing FedProx strategy."""

    if proximal_mu < 0:
        raise ValueError(
            "proximal_mu must be non-negative."
        )

    return FedProx(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=(
            weighted_average_metrics
        ),
        fit_metrics_aggregation_fn=(
            weighted_average_metrics
        ),
        proximal_mu=proximal_mu,
    )


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


def create_server_strategy(
    strategy_name: str = "fedavg",
    proximal_mu: float = 0.01,
    initial_parameters: Optional[Parameters] = None,
    fraction_fit: float = 1.0,
    min_fit_clients: int = 3,
    min_available_clients: int = 3,
    evaluate_fn: Optional[Callable] = None,
):
    """Create FedAvg, encrypted FedAvg, FedProx, or SCAFFOLD strategy."""

    strategy_name = strategy_name.lower()

    if strategy_name == "fedavg":
        return FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=(
                weighted_average_metrics
            ),
            evaluate_metrics_aggregation_fn=(
                weighted_average_metrics
            ),
        )

    if strategy_name in (
        "encrypted",
        "encrypted_fedavg",
        "fedavg_encrypted",
    ):
        return create_encrypted_strategy(
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
        )

    if strategy_name == "fedprox":
        return create_fedprox_strategy(
            proximal_mu=proximal_mu,
            initial_parameters=initial_parameters,
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
        )

    if strategy_name == "scaffold":
        return create_scaffold_strategy(
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
        )

    raise ValueError(
        f"Unsupported strategy: {strategy_name}. "
        "Expected 'fedavg', 'encrypted', "
        "'fedprox', or 'scaffold'."
    )


# ---------------------------------------------------------------------------
# SecAgg+ workflow
# ---------------------------------------------------------------------------


def create_secagg_workflow() -> SecAggPlusWorkflow:
    """Create the Flower SecAgg+ workflow."""

    config = create_secagg_config()

    return SecAggPlusWorkflow(
        num_shares=config.num_clients,
        reconstruction_threshold=config.threshold,
        clipping_range=config.clipping_bound,
        modulus_range=config.modulus_range,
        quantization_range=(
            2**config.quantization_bits
        ),
    )


def create_server_config(
    num_rounds: int = 20,
) -> fl.server.ServerConfig:
    """Create Flower server configuration."""

    return fl.server.ServerConfig(
        num_rounds=num_rounds,
    )


# ---------------------------------------------------------------------------
# ServerApp
# ---------------------------------------------------------------------------


app = ServerApp()


@app.main()
def main(
    grid: Optional[object],
    context: Context,
) -> None:
    """Run the selected federated learning strategy."""

    secagg_config = create_secagg_config()

    model = get_model(
        in_channels=4,
        out_channels=1,
    )

    initial_parameters = get_initial_parameters(
        model
    )

    strategy_name = str(
        context.run_config.get(
            "strategy",
            "fedavg",
        )
    ).lower()

    strategy = create_server_strategy(
        strategy_name=strategy_name,
        initial_parameters=initial_parameters,
    )

    num_rounds = int(
        context.run_config.get(
            "num-server-rounds",
            2,
        )
    )

    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(
            num_rounds=num_rounds,
        ),
        strategy=strategy,
    )

    print(
        "\n===== FedMed Federated Configuration ====="
    )
    print(
        f"Strategy: {strategy_name}"
    )
    print(
        f"Clients: {secagg_config.num_clients}"
    )
    print(
        f"Federated rounds: {num_rounds}"
    )

    if strategy_name in (
        "encrypted",
        "encrypted_fedavg",
        "fedavg_encrypted",
    ):
        print(
            "Selective CKKS encryption: ENABLED"
        )

    print(
        "===========================================\n"
    )

    # Existing SecAgg+ workflow remains enabled
    # for the normal FedAvg path.
    if strategy_name == "fedavg":
        workflow = DefaultWorkflow(
            fit_workflow=create_secagg_workflow(),
        )
    else:
        # Encrypted FedAvg, FedProx and SCAFFOLD
        # use the standard Flower workflow.
        workflow = DefaultWorkflow()

    workflow(
        grid,
        legacy_context,
    )


# ---------------------------------------------------------------------------
# Helper API
# ---------------------------------------------------------------------------


def build_server_app(
    strategy: Optional[
        fl.server.strategy.Strategy
    ] = None,
    num_rounds: int = 20,
) -> ServerApp:
    """Build a Flower ServerApp."""

    if strategy is None:
        strategy = create_strategy()

    config = create_server_config(
        num_rounds=num_rounds,
    )

    return ServerApp(
        strategy=strategy,
        config=config,
    )
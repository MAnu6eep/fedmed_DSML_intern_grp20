"""
experiments/scaffold_convergence_comparison.py

Compare FedAvg, FedProx, and SCAFFOLD on the existing
Dirichlet Non-IID hospital partition.

The experiment records:
    - train loss
    - global/validation loss
    - validation Dice
    - validation IoU
    - global model parameter change
    - participating clients
    - failures
    - SCAFFOLD status
    - SCAFFOLD participating clients
    - SCAFFOLD round

The experiment runner also supports the common
ExperimentConfig configuration system.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fedmed.config.experiment import (
    DatasetConfig,
    ExperimentConfig,
)
from fedmed.core.model import get_model
from fedmed.data.partitioner import partition_dirichlet
from fedmed.federation.client import FedMedClient
from fedmed.federation.server import (
    build_server_app,
    create_server_strategy,
)
from fedmed.metrics.benchmark_framework import SystemMonitor, record_and_export_benchmark


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

NUM_CLIENTS = 3
NUM_ROUNDS = 3
LOCAL_EPOCHS = 1
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
PROXIMAL_MU = 0.01

DIRICHLET_ALPHA = 0.5
SEED = 42


# ============================================================
# SYNTHETIC HOSPITAL DATASET
# ============================================================


class SyntheticHospitalDataset(Dataset):
    """Deterministic synthetic 3D MRI-like hospital dataset."""

    def __init__(self, volume_ids):
        self.volume_ids = volume_ids

    def __len__(self):
        return len(self.volume_ids)

    def __getitem__(self, index):
        volume_id = self.volume_ids[index]

        volume_number = int(
            volume_id.split("_")[-1]
        )

        image_value = (
            (volume_number % 3) + 1
        ) / 3.0

        image = torch.full(
            (4, 32, 32, 16),
            image_value,
            dtype=torch.float32,
        )

        label = torch.zeros(
            (1, 32, 32, 16),
            dtype=torch.float32,
        )

        label[
            :,
            8:24,
            8:24,
            4:12,
        ] = 1.0

        return {
            "image": image,
            "label": label,
        }


# ============================================================
# CLIENT CREATION
# ============================================================


def create_client(
    context,
    partitions,
    batch_size=1,
):
    """Create one simulated hospital client."""

    partition_id = int(
        context.node_config["partition-id"]
    )

    client_id = (
        f"client_{partition_id + 1}"
    )

    hospital_volumes = partitions[
        client_id
    ]

    dataset = SyntheticHospitalDataset(
        hospital_volumes
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    torch.manual_seed(SEED)

    model = get_model(
        in_channels=4,
        out_channels=1,
    )

    client = FedMedClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
    )

    return client.to_client()


# ============================================================
# CENTRALIZED EXPERIMENT
# ============================================================


def run_centralized_experiment(
    config: ExperimentConfig,
    partitions,
):
    """
    Run a centralized experiment using all hospital data.

    This keeps the same model/training configuration as the
    federated experiments but performs local training on the
    combined dataset.
    """

    all_volumes = []

    for hospital_volumes in partitions.values():
        all_volumes.extend(hospital_volumes)

    dataset = SyntheticHospitalDataset(
        all_volumes
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    torch.manual_seed(
        config.dataset.seed
    )

    model = get_model(
        in_channels=4,
        out_channels=1,
    )

    from fedmed.core.training import run_local_training

    result = run_local_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.local_epochs,
        learning_rate=config.learning_rate,
    )

    train_loss = float(
        result.get("train_loss", 0.0)
    )

    val_loss = float(
        result.get("val_loss", 0.0)
    )

    val_dice = float(
        result.get("val_dice", 0.0)
    )

    val_iou = float(
        result.get("val_iou", 0.0)
    )

    centralized_entry = {
        "round": 1,
        "train_loss": train_loss,
        "global_loss": val_loss,
        "val_loss": val_loss,
        "val_dice": val_dice,
        "val_iou": val_iou,
        "parameter_delta": 0.0,
        "participants": 1,
        "failures": 0,
        "scaffold": False,
        "scaffold_clients": 0,
        "scaffold_round": 0,
    }

    record_and_export_benchmark(
        strategy_name="Centralized",
        metrics=centralized_entry,
        round_or_epoch=1,
    )

    return {
        "distribution": "Centralized",
        "strategy": "Centralized",
        "settings": {
            "num_clients": NUM_CLIENTS,
            "num_rounds": config.num_rounds,
            "local_epochs": config.local_epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "proximal_mu": None,
            "partition": config.dataset.partition,
            "dirichlet_alpha": (
                config.dataset.dirichlet_alpha
            ),
            "seed": config.dataset.seed,
        },
        "history": [centralized_entry],
    }


# ============================================================
# STRATEGY FIT CONFIGURATION
# ============================================================


def configure_strategy_fit(
    strategy,
    config: ExperimentConfig,
):
    """
    Inject ExperimentConfig training parameters into the
    Flower Strategy.configure_fit method.

    Existing strategy-specific configuration is preserved.

    This is particularly important for SCAFFOLD because its
    configure_fit method already adds the server control
    variate and other SCAFFOLD-specific fields.
    """

    original_configure_fit = (
        strategy.configure_fit
    )

    client_fit_config = (
        config.to_client_config()
    )

    def configure_fit(
        server_round,
        parameters,
        client_manager,
    ):
        configured_clients = (
            original_configure_fit(
                server_round,
                parameters,
                client_manager,
            )
        )

        updated_clients = []

        for client, fit_ins in configured_clients:

            merged_config = {
                **fit_ins.config,
                **client_fit_config,
            }

            updated_fit_ins = fl.common.FitIns(
                parameters=fit_ins.parameters,
                config=merged_config,
            )

            updated_clients.append(
                (
                    client,
                    updated_fit_ins,
                )
            )

        return updated_clients

    strategy.configure_fit = configure_fit

    return strategy


# ============================================================
# METRIC RECORDING
# ============================================================


def record_strategy_metrics(
    strategy,
    history,
    strategy_name,
):
    """Record round metrics and global model changes."""

    original_aggregate_fit = (
        strategy.aggregate_fit
    )

    previous_parameters = None

    def aggregate_fit(
        server_round,
        results,
        failures,
    ):
        nonlocal previous_parameters

        total_examples = sum(
            fit_res.num_examples
            for _, fit_res in results
        )

        weighted_train_loss = 0.0
        weighted_val_loss = 0.0
        weighted_val_dice = 0.0

        for _, fit_res in results:

            weight = fit_res.num_examples

            weighted_train_loss += (
                weight
                * float(
                    fit_res.metrics.get(
                        "train_loss",
                        0.0,
                    )
                )
            )

            weighted_val_loss += (
                weight
                * float(
                    fit_res.metrics.get(
                        "val_loss",
                        0.0,
                    )
                )
            )

            weighted_val_dice += (
                weight
                * float(
                    fit_res.metrics.get(
                        "val_dice",
                        0.0,
                    )
                )
            )

        aggregated_parameters, metrics = (
            original_aggregate_fit(
                server_round,
                results,
                failures,
            )
        )

        parameter_delta = 0.0

        if aggregated_parameters is not None:

            arrays = (
                fl.common.parameters_to_ndarrays(
                    aggregated_parameters
                )
            )

            if previous_parameters is not None:

                parameter_delta = float(
                    np.sqrt(
                        sum(
                            np.sum(
                                (
                                    current
                                    - previous
                                )
                                ** 2
                            )
                            for current, previous in zip(
                                arrays,
                                previous_parameters,
                            )
                        )
                    )
                )

            previous_parameters = [
                array.copy()
                for array in arrays
            ]

        if total_examples > 0:
            train_loss = weighted_train_loss / total_examples
            val_loss = weighted_val_loss / total_examples
            val_dice = weighted_val_dice / total_examples
        else:
            train_loss = 0.0
            val_loss = 0.0
            val_dice = 0.0

        is_scaffold = (strategy_name.upper() == "SCAFFOLD")

        entry = {
            "round": server_round,
            "train_loss": train_loss,
            "global_loss": val_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "parameter_delta": parameter_delta,
            "participants": len(results),
            "failures": len(failures),
            "scaffold": is_scaffold,
            "scaffold_clients": len(results) if is_scaffold else 0,
            "scaffold_round": server_round if is_scaffold else 0,
        }

        history.append(entry)

        # Automatically record to benchmark suite
        record_and_export_benchmark(
            strategy_name=strategy_name,
            metrics=entry,
            round_or_epoch=server_round,
        )

        return (
            aggregated_parameters,
            metrics,
        )

    strategy.aggregate_fit = (
        aggregate_fit
    )

    return strategy


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================


def run_experiment(
    config,
    partitions,
):
    """
    Run one experiment selected by ExperimentConfig.

    Supported strategies:
        centralized
        fedavg
        fedprox
        scaffold

    Backward compatibility:
        run_experiment("FedAvg", partitions)
        run_experiment("FedProx", partitions)
        run_experiment("SCAFFOLD", partitions)
    """

    # --------------------------------------------------------
    # Backward compatibility
    # --------------------------------------------------------

    if isinstance(config, str):

        config = ExperimentConfig(
            strategy=config,
            num_rounds=NUM_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            dataset=DatasetConfig(
                name="brats",
                partition="non_iid",
                dirichlet_alpha=DIRICHLET_ALPHA,
                seed=SEED,
            ),
            proximal_mu=PROXIMAL_MU,
        )

    elif not isinstance(
        config,
        ExperimentConfig,
    ):

        raise TypeError(
            "config must be an ExperimentConfig "
            "instance or a supported strategy string."
        )

    # --------------------------------------------------------
    # Centralized
    # --------------------------------------------------------

    if config.strategy == "centralized":

        return run_centralized_experiment(
            config=config,
            partitions=partitions,
        )

    # --------------------------------------------------------
    # Federated strategies
    # --------------------------------------------------------

    history = []

    strategy_name = config.strategy

    display_name = {
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "scaffold": "SCAFFOLD",
    }.get(strategy_name.lower(), strategy_name)

    # --------------------------------------------------------
    # Build initial model.
    # --------------------------------------------------------

    torch.manual_seed(
        config.dataset.seed
    )

    initial_model = get_model(
        in_channels=4,
        out_channels=1,
    )

    initial_parameters = (
        fl.common.ndarrays_to_parameters(
            [
                value.detach()
                .cpu()
                .numpy()
                .copy()
                for value in (
                    initial_model
                    .state_dict()
                    .values()
                )
            ]
        )
    )

    # --------------------------------------------------------
    # Create Flower strategy.
    # --------------------------------------------------------

    strategy = create_server_strategy(
        strategy_name=strategy_name,
        proximal_mu=config.proximal_mu,
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
    )

    # --------------------------------------------------------
    # Inject configured client training parameters.
    # --------------------------------------------------------

    strategy = configure_strategy_fit(
        strategy,
        config,
    )

    # --------------------------------------------------------
    # Record metrics without changing aggregation.
    # --------------------------------------------------------

    strategy = record_strategy_metrics(
        strategy,
        history,
        display_name,
    )

    client_app = fl.client.ClientApp(
        client_fn=lambda context: create_client(
            context,
            partitions,
            batch_size=config.batch_size,
        )
    )

    server_app = build_server_app(
        strategy=strategy,
        num_rounds=config.num_rounds,
    )

    with SystemMonitor() as monitor:
        fl.simulation.run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=NUM_CLIENTS,
            backend_config={
                "client_resources": {
                    "num_cpus": 1,
                    "num_gpus": 0.0,
                }
            },
        )

    # Attach timing & VRAM info to history
    for entry in history:
        entry["execution_time_seconds"] = round(monitor.elapsed_time / max(len(history), 1), 4)
        entry["vram_peak_mb"] = monitor.vram_peak_mb
        entry["vram_current_mb"] = monitor.vram_current_mb
        entry["cuda_available"] = monitor.cuda_available

    return {
        "distribution": "Non-IID",
        "strategy": display_name,
        "settings": {
            "num_clients": NUM_CLIENTS,
            "num_rounds": config.num_rounds,
            "local_epochs": config.local_epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "proximal_mu": (
                config.proximal_mu
                if strategy_name == "fedprox"
                else None
            ),
            "partition": (
                config.dataset.partition
            ),
            "dirichlet_alpha": (
                config.dataset.dirichlet_alpha
            ),
            "seed": config.dataset.seed,
        },
        "history": history,
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================


def main():
    """Run FedAvg, FedProx and SCAFFOLD comparison."""

    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    # --------------------------------------------------------
    # Dataset configuration
    # --------------------------------------------------------

    dataset_config = DatasetConfig(
        name="brats",
        partition="non_iid",
        dirichlet_alpha=DIRICHLET_ALPHA,
        seed=SEED,
    )

    volumes = [
        f"volume_{i:02d}"
        for i in range(12)
    ]

    labels = [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
    ]

    partitions = partition_dirichlet(
        volumes,
        labels,
        num_clients=NUM_CLIENTS,
        alpha=dataset_config.dirichlet_alpha,
        seed=dataset_config.seed,
    )

    print(
        "\n===============================================\n"
        "       NON-IID HOSPITAL PARTITIONS\n"
        "===============================================\n"
    )

    for client_id, hospital_volumes in partitions.items():
        print(
            f"{client_id}: {len(hospital_volumes)} volumes -> {hospital_volumes}"
        )

    # --------------------------------------------------------
    # Create experiment configurations.
    # --------------------------------------------------------

    experiment_configs = [
        ExperimentConfig(
            strategy="fedavg",
            num_rounds=NUM_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            dataset=dataset_config,
            proximal_mu=PROXIMAL_MU,
        ),
        ExperimentConfig(
            strategy="fedprox",
            num_rounds=NUM_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            dataset=dataset_config,
            proximal_mu=PROXIMAL_MU,
        ),
        ExperimentConfig(
            strategy="scaffold",
            num_rounds=NUM_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            dataset=dataset_config,
            proximal_mu=PROXIMAL_MU,
        ),
    ]

    # --------------------------------------------------------
    # Run all strategies on the same partition.
    # --------------------------------------------------------

    results = []

    for config in experiment_configs:

        display_name = {
            "fedavg": "FedAvg",
            "fedprox": "FedProx",
            "scaffold": "SCAFFOLD",
        }[config.strategy]

        print(
            f"\n===============================================\n"
            f"              Running {display_name}\n"
            f"===============================================\n"
        )

        result = run_experiment(
            config,
            partitions,
        )

        for metrics in result["history"]:
            if metrics["failures"] != 0:
                raise RuntimeError(
                    f"{display_name} failed in "
                    f"round {metrics['round']}: "
                    f"{metrics['failures']} failures."
                )

        if len(result["history"]) != config.num_rounds:
            raise RuntimeError(
                f"{display_name} produced "
                f"{len(result['history'])} rounds; "
                f"expected {config.num_rounds}."
            )

        results.append(result)

        for metrics in result["history"]:
            print(
                f"Round {metrics['round']}: "
                f"global_loss={metrics['global_loss']:.6f}, "
                f"val_dice={metrics['val_dice']:.6f}, "
                f"parameter_delta={metrics['parameter_delta']:.6f}"
            )

    output_dir = PROJECT_ROOT / "experiments" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "scaffold_convergence_comparison.json"

    report: Dict[str, Any] = {
        "title": (
            "FedAvg vs FedProx vs SCAFFOLD "
            "Non-IID Convergence Comparison"
        ),
        "status": "COMPLETED",
        "distribution": "Non-IID",
        "partition_method": "Dirichlet",
        "dirichlet_alpha": (
            dataset_config.dirichlet_alpha
        ),
        "strategies": [
            "FedAvg",
            "FedProx",
            "SCAFFOLD",
        ],
        "experiment_configuration": {
            "num_clients": NUM_CLIENTS,
            "num_rounds": NUM_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "proximal_mu": PROXIMAL_MU,
            "dataset": dataset_config.name,
            "partition": dataset_config.partition,
            "dirichlet_alpha": (
                dataset_config.dirichlet_alpha
            ),
            "seed": dataset_config.seed,
        },
        "experiments": results,
    }

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(
        "\n================================================\n"
        "[SUCCESS] SCAFFOLD convergence comparison completed successfully.\n"
        f"Results saved to:\n{output_file}\n"
        "================================================\n"
    )


if __name__ == "__main__":
    main()
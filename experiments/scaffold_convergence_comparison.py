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

from fedmed.core.model import get_model
from fedmed.data.partitioner import partition_dirichlet
from fedmed.federation.client import FedMedClient
from fedmed.federation.server import (
    build_server_app,
    create_fedprox_strategy,
    create_scaffold_strategy,
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
        batch_size=1,
        shuffle=False,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=1,
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

        is_scaffold = (strategy_name == "SCAFFOLD")

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
# RUN ONE STRATEGY
# ============================================================


def run_experiment(
    strategy_name: str,
    partitions,
):
    """Run one FL strategy on the same Non-IID partition."""

    history = []

    torch.manual_seed(SEED)

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

    if strategy_name == "FedAvg":
        strategy = create_server_strategy(
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
        )

    elif strategy_name == "FedProx":
        strategy = create_fedprox_strategy(
            proximal_mu=PROXIMAL_MU,
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
        )

    elif strategy_name == "SCAFFOLD":
        strategy = create_scaffold_strategy(
            initial_parameters=initial_parameters,
        )

    else:
        raise ValueError(
            f"Unknown strategy: {strategy_name}"
        )

    strategy = record_strategy_metrics(
        strategy,
        history,
        strategy_name,
    )

    client_app = fl.client.ClientApp(
        client_fn=lambda context: create_client(
            context,
            partitions,
        )
    )

    server_app = build_server_app(
        strategy=strategy,
        num_rounds=NUM_ROUNDS,
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
        "strategy": strategy_name,
        "settings": {
            "num_clients": NUM_CLIENTS,
            "num_rounds": NUM_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "proximal_mu": (
                PROXIMAL_MU
                if strategy_name == "FedProx"
                else None
            ),
            "partition": "Dirichlet",
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "seed": SEED,
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
        alpha=DIRICHLET_ALPHA,
        seed=SEED,
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

    results = []

    for strategy_name in [
        "FedAvg",
        "FedProx",
        "SCAFFOLD",
    ]:
        print(
            f"\n===============================================\n"
            f"              Running {strategy_name}\n"
            f"===============================================\n"
        )

        result = run_experiment(
            strategy_name,
            partitions,
        )

        for metrics in result["history"]:
            if metrics["failures"] != 0:
                raise RuntimeError(
                    f"{strategy_name} failed in "
                    f"round {metrics['round']}: "
                    f"{metrics['failures']} failures."
                )

        if len(result["history"]) != NUM_ROUNDS:
            raise RuntimeError(
                f"{strategy_name} produced "
                f"{len(result['history'])} rounds; "
                f"expected {NUM_ROUNDS}."
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
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "strategies": [
            "FedAvg",
            "FedProx",
            "SCAFFOLD",
        ],
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
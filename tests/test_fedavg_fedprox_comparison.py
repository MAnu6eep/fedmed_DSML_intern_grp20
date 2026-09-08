import json
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fedmed.core.training import run_local_training
from fedmed.data.partitioner import (
    partition_dirichlet,
    partition_iid,
)
from fedmed.federation.client import FedMedClient
from fedmed.federation.server import (
    create_fedprox_strategy,
    create_server_strategy,
    build_server_app,
)


NUM_CLIENTS = 3
NUM_ROUNDS = 3
PROXIMAL_MU = 0.01


class SyntheticHospitalDataset(Dataset):
    """Small deterministic dataset for FL comparison."""

    def __init__(self, volume_ids):
        self.volume_ids = volume_ids

    def __len__(self):
        return len(self.volume_ids)

    def __getitem__(self, index):
        volume_id = self.volume_ids[index]
        volume_number = int(volume_id.split("_")[-1])

        # Different image intensity represents different
        # hospital/data distributions.
        image_value = (volume_number % 3 + 1) / 3.0

        image = torch.full(
            (1, 4, 4, 4),
            image_value,
            dtype=torch.float32,
        )

        # Same valid segmentation target for all samples.
        label = torch.zeros(
            (1, 4, 4, 4),
            dtype=torch.float32,
        )
        label[:, 1:3, 1:3, 1:3] = 1.0

        return {
            "image": image,
            "label": label,
        }


def create_client(
    context,
    partitions,
):
    partition_id = int(
        context.node_config["partition-id"]
    )

    client_id = f"client_{partition_id + 1}"

    hospital_volumes = partitions[client_id]

    dataset = SyntheticHospitalDataset(
        hospital_volumes
    )

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    torch.manual_seed(42)

    model = torch.nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=1,
    )

    client = FedMedClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
    )

    return client.to_client()


def record_strategy_metrics(
    strategy,
    history,
):
    original_aggregate_fit = strategy.aggregate_fit

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
            arrays = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            if previous_parameters is not None:
                parameter_delta = float(
                    np.sqrt(
                        sum(
                            np.sum(
                                (current - previous) ** 2
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
            train_loss = (
                weighted_train_loss
                / total_examples
            )

            val_loss = (
                weighted_val_loss
                / total_examples
            )

            val_dice = (
                weighted_val_dice
                / total_examples
            )
        else:
            train_loss = 0.0
            val_loss = 0.0
            val_dice = 0.0

        history.append(
            {
                "round": server_round,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "parameter_delta": parameter_delta,
                "participants": len(results),
                "failures": len(failures),
            }
        )

        return aggregated_parameters, metrics

    strategy.aggregate_fit = aggregate_fit

    return strategy


def run_experiment(
    distribution_name,
    partitions,
    strategy_name,
):
    history = []

    if strategy_name == "FedAvg":
        strategy = create_server_strategy(
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
        )
    else:
        strategy = create_fedprox_strategy(
            proximal_mu=PROXIMAL_MU,
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
        )

    strategy = record_strategy_metrics(
        strategy,
        history,
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

    return {
        "distribution": distribution_name,
        "strategy": strategy_name,
        "history": history,
    }


def test_fedavg_vs_fedprox_iid_and_non_iid():
    torch.set_num_threads(1)
    torch.manual_seed(42)

    volumes = [
        f"volume_{i:02d}"
        for i in range(12)
    ]

    labels = [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
    ]

    iid_partitions = partition_iid(
        volumes,
        num_clients=NUM_CLIENTS,
        seed=42,
    )

    non_iid_partitions = partition_dirichlet(
        volumes,
        labels,
        num_clients=NUM_CLIENTS,
        alpha=0.5,
        seed=42,
    )

    experiments = []

    for distribution_name, partitions in [
        ("IID", iid_partitions),
        ("Non-IID", non_iid_partitions),
    ]:
        for strategy_name in [
            "FedAvg",
            "FedProx",
        ]:
            result = run_experiment(
                distribution_name,
                partitions,
                strategy_name,
            )

            experiments.append(result)

    # Verify all four experiments completed.
    assert len(experiments) == 4

    for result in experiments:
        history = result["history"]

        assert len(history) == NUM_ROUNDS

        for round_metrics in history:
            assert round_metrics["participants"] == NUM_CLIENTS
            assert round_metrics["failures"] == 0
            assert np.isfinite(
                round_metrics["train_loss"]
            )
            assert np.isfinite(
                round_metrics["val_loss"]
            )
            assert np.isfinite(
                round_metrics["val_dice"]
            )

    # Print round-wise comparison metrics.
    print("\n===== FedAvg vs FedProx Comparison =====")

    for result in experiments:
        print(
            f"\n{result['distribution']} - "
            f"{result['strategy']}"
        )

        for metrics in result["history"]:
            print(
                f"Round {metrics['round']}: "
                f"train_loss="
                f"{metrics['train_loss']:.6f}, "
                f"val_loss="
                f"{metrics['val_loss']:.6f}, "
                f"val_dice="
                f"{metrics['val_dice']:.6f}, "
                f"parameter_delta="
                f"{metrics['parameter_delta']:.6f}"
            )

    # Save round-wise experiment results.
    output_dir = Path("tests") / "results"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "fedavg_fedprox_comparison.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiments,
            file,
            indent=2,
        )

    assert output_file.exists()
import json
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from flwr.server.strategy import FedAvg

from fedmed.core.model import FedMedUNet3D
from fedmed.data.partitioner import partition_iid
from fedmed.federation.client import FedMedClient
from fedmed.federation.server import (
    build_server_app,
    create_fedprox_strategy,
    get_initial_parameters,
)


NUM_CLIENTS = 3
NUM_ROUNDS = 3
PROXIMAL_MU = 0.01


class SyntheticHospitalDataset(Dataset):
    """
    Small deterministic synthetic 3D segmentation dataset.

    The project currently uses simulated hospital data rather
    than real MRI volumes.
    """

    def __init__(self, volume_ids):
        self.volume_ids = list(volume_ids)

    def __len__(self):
        return len(self.volume_ids)

    def __getitem__(self, index):
        volume_id = self.volume_ids[index]
        volume_number = int(volume_id.split("_")[-1])

        # Deterministic image content for reproducibility.
        image_value = ((volume_number % 3) + 1) / 3.0

        image = torch.full(
            (4, 16, 16, 16),
            image_value,
            dtype=torch.float32,
        )

        # Simple foreground segmentation target.
        label = torch.zeros(
            (1, 16, 16, 16),
            dtype=torch.float32,
        )

        label[
            :,
            6:10,
            6:10,
            6:10,
        ] = 1.0

        return {
            "image": image,
            "label": label,
        }


def create_model():
    """
    Create the actual FedMed 3D U-Net architecture
    with a smaller configuration for the end-to-end test.
    """
    return FedMedUNet3D(
        in_channels=4,
        out_channels=1,
        channels=(4, 8, 16),
        strides=(2, 2),
        num_res_units=1,
        dropout=0.0,
    )


def create_client(context, partitions):
    """
    Create one federated hospital client.

    Flower's partition-id identifies the simulated hospital.
    """

    partition_id = int(
        context.node_config["partition-id"]
    )

    client_id = f"client_{partition_id + 1}"

    hospital_volumes = partitions[client_id]

    train_dataset = SyntheticHospitalDataset(
        hospital_volumes
    )

    val_dataset = SyntheticHospitalDataset(
        hospital_volumes
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
    )

    torch.manual_seed(42)

    model = create_model()

    client = FedMedClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
    )

    return client.to_client()


def attach_recording(strategy, history):
    """
    Wrap Flower's aggregate_fit so that every round records:

    - number of participating hospitals
    - failures
    - weighted training loss
    - weighted validation loss
    - global parameter change
    """

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

        for _, fit_res in results:
            examples = fit_res.num_examples

            weighted_train_loss += (
                examples
                * float(
                    fit_res.metrics.get(
                        "train_loss",
                        0.0,
                    )
                )
            )

            weighted_val_loss += (
                examples
                * float(
                    fit_res.metrics.get(
                        "val_loss",
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
            current_parameters = (
                fl.common.parameters_to_ndarrays(
                    aggregated_parameters
                )
            )

            if previous_parameters is not None:
                squared_difference = 0.0

                for current, previous in zip(
                    current_parameters,
                    previous_parameters,
                ):
                    # Only calculate the numerical delta
                    # for floating-point parameters.
                    if np.issubdtype(
                        current.dtype,
                        np.floating,
                    ):
                        current_float = (
                            current.astype(
                                np.float64
                            )
                        )

                        previous_float = (
                            previous.astype(
                                np.float64
                            )
                        )

                        squared_difference += np.sum(
                            (
                                current_float
                                - previous_float
                            )
                            ** 2
                        )

                parameter_delta = float(
                    np.sqrt(
                        squared_difference
                    )
                )

            previous_parameters = [
                parameter.copy()
                for parameter in current_parameters
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
        else:
            train_loss = 0.0
            val_loss = 0.0

        round_metrics = {
            "round": server_round,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "parameter_delta": float(
                parameter_delta
            ),
            "participants": len(results),
            "failures": len(failures),
        }

        history.append(round_metrics)

        return (
            aggregated_parameters,
            metrics,
        )

    strategy.aggregate_fit = aggregate_fit

    return strategy


def run_experiment(
    strategy_name,
    partitions,
    initial_parameters,
):
    """
    Run one complete multi-round federated experiment.
    """

    history = []

    if strategy_name == "FedAvg":

        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
            initial_parameters=initial_parameters,
        )

    elif strategy_name == "FedProx":

        strategy = create_fedprox_strategy(
            proximal_mu=PROXIMAL_MU,
            initial_parameters=initial_parameters,
            fraction_fit=1.0,
            min_fit_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
        )

    else:
        raise ValueError(
            f"Unknown strategy: {strategy_name}"
        )

    strategy = attach_recording(
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
        "strategy": strategy_name,
        "history": history,
    }


def test_end_to_end_multi_round_federated_training():
    """
    End-to-end verification of:

    Hospital A + Hospital B + Hospital C
            ↓
       Local training
            ↓
       FedAvg / FedProx
            ↓
     Global aggregation
            ↓
       Multiple rounds
    """

    torch.set_num_threads(1)
    torch.manual_seed(42)

    # ---------------------------------------------------------
    # 1. Create simulated hospital volumes
    # ---------------------------------------------------------

    volumes = [
        f"volume_{i:02d}"
        for i in range(12)
    ]

    partitions = partition_iid(
        volumes,
        num_clients=NUM_CLIENTS,
        seed=42,
    )

    print("\nHospital partitions:")

    for client_id, hospital_volumes in partitions.items():
        print(
            f"{client_id}: "
            f"{hospital_volumes}"
        )

    # ---------------------------------------------------------
    # 2. Create identical initial global model parameters
    # ---------------------------------------------------------

    torch.manual_seed(42)

    initial_model = create_model()

    initial_parameters = (
        get_initial_parameters(
            initial_model
        )
    )

    # ---------------------------------------------------------
    # 3. Run FedAvg
    # ---------------------------------------------------------

    print("\nStarting FedAvg experiment...")

    fedavg_result = run_experiment(
        strategy_name="FedAvg",
        partitions=partitions,
        initial_parameters=initial_parameters,
    )

    # ---------------------------------------------------------
    # 4. Run FedProx
    # ---------------------------------------------------------

    print("\nStarting FedProx experiment...")

    fedprox_result = run_experiment(
        strategy_name="FedProx",
        partitions=partitions,
        initial_parameters=initial_parameters,
    )

    results = [
        fedavg_result,
        fedprox_result,
    ]

    # ---------------------------------------------------------
    # 5. Verify all rounds completed
    # ---------------------------------------------------------

    for result in results:

        history = result["history"]

        assert len(history) == NUM_ROUNDS, (
            f"{result['strategy']} did not "
            f"complete {NUM_ROUNDS} rounds."
        )

        # -----------------------------------------------------
        # Every round must have all 3 hospitals.
        # -----------------------------------------------------

        for round_metrics in history:

            assert (
                round_metrics["participants"]
                == NUM_CLIENTS
            ), (
                f"{result['strategy']} round "
                f"{round_metrics['round']} did not "
                f"receive updates from all hospitals."
            )

            assert (
                round_metrics["failures"] == 0
            ), (
                f"{result['strategy']} round "
                f"{round_metrics['round']} had failures."
            )

            assert np.isfinite(
                round_metrics["train_loss"]
            )

            assert np.isfinite(
                round_metrics["val_loss"]
            )

    # ---------------------------------------------------------
    # 6. Verify parameter exchange and aggregation
    # ---------------------------------------------------------

    for result in results:

        history = result["history"]

        # Round 1 establishes the first global model.
        # Later rounds must change it.
        later_round_deltas = [
            metrics["parameter_delta"]
            for metrics in history[1:]
        ]

        assert any(
            delta > 0.0
            for delta in later_round_deltas
        ), (
            f"{result['strategy']} global parameters "
            "did not change after aggregation."
        )

    # ---------------------------------------------------------
    # 7. Verify training improvement
    # ---------------------------------------------------------

    for result in results:

        history = result["history"]

        initial_loss = history[0]["train_loss"]
        final_loss = history[-1]["train_loss"]

        assert final_loss < initial_loss, (
            f"{result['strategy']} training loss "
            f"did not improve: "
            f"{initial_loss:.6f} -> "
            f"{final_loss:.6f}"
        )

    # ---------------------------------------------------------
    # 8. Print round-wise convergence results
    # ---------------------------------------------------------

    print(
        "\n=============================================="
    )
    print(
        " End-to-End Federated Training Results"
    )
    print(
        "=============================================="
    )

    for result in results:

        print(
            f"\n{result['strategy']}"
        )

        for metrics in result["history"]:

            print(
                f"Round {metrics['round']}: "
                f"train_loss="
                f"{metrics['train_loss']:.6f}, "
                f"val_loss="
                f"{metrics['val_loss']:.6f}, "
                f"parameter_delta="
                f"{metrics['parameter_delta']:.6f}, "
                f"participants="
                f"{metrics['participants']}, "
                f"failures="
                f"{metrics['failures']}"
            )

    # ---------------------------------------------------------
    # 9. Save round-wise results
    # ---------------------------------------------------------

    output_dir = (
        Path("tests") / "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "end_to_end_federated_training.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    assert output_file.exists()
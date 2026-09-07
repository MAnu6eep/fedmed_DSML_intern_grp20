"""
tests/test_fedprox_non_iid.py

Test FedProx training with the existing Dirichlet Non-IID
hospital partitioning.
"""

from collections import OrderedDict

import flwr as fl
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fedmed.data.partitioner import partition_dirichlet
from fedmed.federation.client import FedMedClient
from fedmed.federation.server import (
    create_fedprox_strategy,
    build_server_app,
)

from flwr.clientapp import ClientApp


# ---------------------------------------------------------------------
# Deterministic settings
# ---------------------------------------------------------------------

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------
# Synthetic hospital dataset
# ---------------------------------------------------------------------

class MockHospitalDataset(Dataset):
    """Small synthetic dataset representing one hospital partition."""

    def __init__(self, volume_ids):
        self.volume_ids = volume_ids

    def __len__(self):
        return len(self.volume_ids)

    def __getitem__(self, index):
        volume_id = self.volume_ids[index]

        # Deterministic value based on volume ID
        volume_number = int(
            volume_id.split("_")[-1]
        )

        generator = torch.Generator()
        generator.manual_seed(volume_number)

        image = torch.randn(
            1,
            8,
            8,
            8,
            generator=generator,
        )

        # Binary segmentation target
        label = (
            image > 0
        ).float()

        return {
            "image": image,
            "label": label,
        }


def make_mock_dataloader(
    volume_ids,
    batch_size=1,
):
    """Create a DataLoader for one hospital partition."""

    dataset = MockHospitalDataset(
        volume_ids
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


# ---------------------------------------------------------------------
# Small model for fast federated testing
# ---------------------------------------------------------------------

class MockSegmentationModel(nn.Module):
    """Small 3D segmentation model used only for the test."""

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------

def get_parameters(model):
    """Extract model parameters as NumPy arrays."""

    return [
        value.detach().cpu().numpy()
        for value in model.state_dict().values()
    ]


def set_parameters(model, parameters):
    """Load NumPy parameters into a model."""

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


# ---------------------------------------------------------------------
# FedProx Non-IID test
# ---------------------------------------------------------------------

def test_fedprox_with_non_iid_partitions():
    """
    Verify that FedProx can train successfully on
    Dirichlet Non-IID hospital partitions.
    """

    # -------------------------------------------------------------
    # 1. Create synthetic volume identifiers and labels
    # -------------------------------------------------------------

    volumes = [
        f"volume_{i:02d}"
        for i in range(12)
    ]

    labels = [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
    ]

    # -------------------------------------------------------------
    # 2. Create existing Dirichlet Non-IID partitions
    # -------------------------------------------------------------

    partitions = partition_dirichlet(
        volumes=volumes,
        labels=labels,
        num_clients=3,
        alpha=0.5,
        seed=42,
    )

    print("\n===== DIRICHLET NON-IID PARTITIONS =====")

    for client_id, client_volumes in partitions.items():
        print(
            f"{client_id}: "
            f"{len(client_volumes)} volumes -> "
            f"{client_volumes}"
        )

    # Make sure every volume was assigned exactly once
    assigned_volumes = []

    for client_volumes in partitions.values():
        assigned_volumes.extend(client_volumes)

    assert len(assigned_volumes) == len(volumes)
    assert set(assigned_volumes) == set(volumes)

    # -------------------------------------------------------------
    # 3. Create a FedProx strategy
    # -------------------------------------------------------------

    proximal_mu = 0.01

    strategy = create_fedprox_strategy(
        proximal_mu=proximal_mu,
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
    )

    # -------------------------------------------------------------
    # 4. Recording variables
    # -------------------------------------------------------------

    round_participants = []
    round_failures = []
    round_train_losses = []
    round_val_dice = []
    round_proximal_mu = []

    # Keep the original aggregate_fit method
    original_aggregate_fit = strategy.aggregate_fit

    def recording_aggregate_fit(
        server_round,
        results,
        failures,
    ):
        """Record FedProx training behavior before aggregation."""

        round_participants.append(
            len(results)
        )

        round_failures.append(
            len(failures)
        )

        # Record client-side training metrics
        train_losses = []
        val_dice_scores = []
        mu_values = []

        for _, fit_res in results:

            if "train_loss" in fit_res.metrics:
                train_losses.append(
                    float(
                        fit_res.metrics["train_loss"]
                    )
                )

            if "val_dice" in fit_res.metrics:
                val_dice_scores.append(
                    float(
                        fit_res.metrics["val_dice"]
                    )
                )

            if "proximal_mu" in fit_res.metrics:
                mu_values.append(
                    float(
                        fit_res.metrics["proximal_mu"]
                    )
                )

        if train_losses:
            round_train_losses.append(
                sum(train_losses)
                / len(train_losses)
            )
        if val_dice_scores:
            round_val_dice.append(
                sum(val_dice_scores)
                / len(val_dice_scores)
            )

        round_proximal_mu.extend(
            mu_values
        )

        # Perform the real FedProx aggregation
        return original_aggregate_fit(
            server_round,
            results,
            failures,
        )

    strategy.aggregate_fit = recording_aggregate_fit

    # -------------------------------------------------------------
    # 5. Create Flower client function
    # -------------------------------------------------------------

    def client_fn(context):
        """
        Create one simulated hospital client.

        IMPORTANT:
        Flower's node_id is a unique identifier and is NOT the
        partition index. The partition-id must be used to map
        simulated clients to the existing Non-IID partitions.
        """

        partition_id = int(
            context.node_config[
                "partition-id"
            ]
        )

        client_id = (
            f"client_{partition_id + 1}"
        )

        hospital_volumes = partitions[
            client_id
        ]

        # Training and validation loaders
        train_loader = make_mock_dataloader(
            hospital_volumes,
            batch_size=1,
        )

        val_loader = make_mock_dataloader(
            hospital_volumes,
            batch_size=1,
        )

        model = MockSegmentationModel()

        client = FedMedClient(
            client_id=client_id,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
        )

        return client.to_client()

    # -------------------------------------------------------------
    # 6. Run federated training
    # -------------------------------------------------------------

    print("\n===== STARTING FEDPROX SIMULATION =====")

    client_app = ClientApp(
        client_fn=client_fn
    )

    server_app = build_server_app(
        strategy=strategy,
        num_rounds=3,
    )

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=3,
    )

    # -------------------------------------------------------------
    # 7. Verify federated training completed
    # -------------------------------------------------------------

    print("\n===== FEDPROX TRAINING RESULTS =====")

    print(
        f"Rounds completed: "
        f"{len(round_participants)}"
    )

    print(
        f"Participants per round: "
        f"{round_participants}"
    )

    print(
        f"Failures per round: "
        f"{round_failures}"
    )

    print(
        f"Training losses: "
        f"{round_train_losses}"
    )

    print(
        f"Validation Dice: "
        f"{round_val_dice}"
    )

    print(
        f"FedProx mu values: "
        f"{round_proximal_mu}"
    )

    # Three rounds should complete
    assert len(round_participants) == 3

    # All three hospital clients should participate
    assert round_participants == [3, 3, 3]

    # No client/server aggregation failures
    assert round_failures == [0, 0, 0]

    # Every client should receive the configured FedProx value
    assert len(round_proximal_mu) == 9

    assert all(
        mu == pytest.approx(proximal_mu)
        for mu in round_proximal_mu
    )

    # Training metrics should have been recorded
    assert len(round_train_losses) == 3

    # Validation metrics should have been recorded
    assert len(round_val_dice) == 3

    print(
        "\nFedProx Non-IID training "
        "completed successfully."
    )


if __name__ == "__main__":
    test_fedprox_with_non_iid_partitions()
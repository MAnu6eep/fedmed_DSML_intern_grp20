"""
Multi-round, three-client Flower FedAvg simulation test.
"""

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import parameters_to_ndarrays

from fedmed.federation.client import FedMedClient


class MockSegmentationModel(nn.Module):
    """Small 3D model used for the federated simulation test."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x):
        return self.conv(x)


class RecordingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that records each round."""

    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
        )

        self.round_metrics = {}
        self.global_parameters = {}

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):
        """Aggregate updates and record the global model."""

        aggregated = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        parameters, _ = aggregated

        if parameters is not None:
            self.global_parameters[server_round] = (
                parameters_to_ndarrays(parameters)
            )

        self.round_metrics[server_round] = {
            "participants": len(results),
            "failures": len(failures),
        }

        print(
            f"Round {server_round}: "
            f"{len(results)} clients participated, "
            f"{len(failures)} failures"
        )

        return aggregated


def make_mock_dataloader(client_index):
    """Create deterministic mock data for one hospital."""

    torch.manual_seed(100 + client_index)

    images = torch.randn(
        2,
        1,
        4,
        4,
        4,
    )

    labels = torch.sigmoid(images * 0.5)

    base_dataset = TensorDataset(
        images,
        labels,
    )

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            image, label = self.dataset[index]

            return {
                "image": image,
                "label": label,
            }

    return DataLoader(
        DictDataset(base_dataset),
        batch_size=1,
        shuffle=False,
    )


def client_fn(context):
    """Create one isolated hospital client."""

    client_index = int(context.node_id)

    model = MockSegmentationModel()

    train_loader = make_mock_dataloader(client_index)
    val_loader = make_mock_dataloader(client_index)

    client = FedMedClient(
        client_id=f"hospital_{client_index}",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
    )

    return client.to_client()


def test_three_client_multi_round_fedavg():
    """Verify multi-round FedAvg across three hospital clients."""

    strategy = RecordingFedAvg()

    server_app = fl.server.ServerApp(
        strategy=strategy,
        config=fl.server.ServerConfig(
            num_rounds=3,
        ),
    )

    client_app = fl.client.ClientApp(
        client_fn=client_fn
    )

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=3,
        backend_config={
            "client_resources": {
                "num_cpus": 1,
                "num_gpus": 0,
            }
        },
    )

    # Verify all three rounds completed.
    assert set(strategy.round_metrics.keys()) == {1, 2, 3}

    # Verify all three hospitals participated every round.
    for round_number in range(1, 4):
        assert (
            strategy.round_metrics[round_number]["participants"]
            == 3
        )

        assert (
            strategy.round_metrics[round_number]["failures"]
            == 0
        )

    # Verify global parameters were produced every round.
    assert set(strategy.global_parameters.keys()) == {1, 2, 3}

    # Get global parameters from consecutive rounds.
    round_1 = strategy.global_parameters[1]
    round_2 = strategy.global_parameters[2]
    round_3 = strategy.global_parameters[3]

    # Verify the global model changed after round 1.
    assert any(
        not torch.equal(
            torch.tensor(param_1),
            torch.tensor(param_2),
        )
        for param_1, param_2 in zip(round_1, round_2)
    )

    # Verify the global model changed after round 2.
    assert any(
        not torch.equal(
            torch.tensor(param_2),
            torch.tensor(param_3),
        )
        for param_2, param_3 in zip(round_2, round_3)
    )

    print(
        "\n3-client, 3-round FedAvg simulation PASSED"
    )
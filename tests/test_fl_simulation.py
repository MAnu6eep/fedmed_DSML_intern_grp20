"""Day 6: Three-client, one-round Flower FedAvg simulation test."""

from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fedmed.core.training import run_local_training
from fedmed.federation.client import FedMedClient
from fedmed.federation.server import create_server_strategy


class MockSegmentationModel(nn.Module):
    """Small 3D model used only for the federated simulation test."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class RecordingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that records round-level participation and metrics."""

    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
        )
        self.fit_clients = []
        self.round_metrics = {}

    def aggregate_fit(self, server_round, results, failures):
        self.fit_clients = [
            client_proxy.cid for client_proxy, _ in results
        ]

        aggregated = super().aggregate_fit(
            server_round,
            results,
            failures,
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
    """Create deterministic mock tensors for one hospital."""
    torch.manual_seed(100 + client_index)

    images = torch.randn(2, 1, 4, 4, 4)
    labels = torch.sigmoid(images * 0.5)

    dataset = TensorDataset(images, labels)

    # Convert TensorDataset output into the dictionary format expected
    # by fedmed/core/training.py.
    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, index):
            image, label = self.base_dataset[index]
            return {"image": image, "label": label}

    return DataLoader(
        DictDataset(dataset),
        batch_size=1,
        shuffle=False,
    )


def client_fn(context):
    """Create one isolated mock hospital client."""
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


def test_three_client_one_round_fedavg():
    """Verify one complete three-client Flower FedAvg round."""

    strategy = RecordingFedAvg()

    server_app = fl.server.ServerApp(
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=1),
    )

    client_app = fl.client.ClientApp(client_fn=client_fn)

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

    assert 1 in strategy.round_metrics
    assert strategy.round_metrics[1]["participants"] == 3
    assert strategy.round_metrics[1]["failures"] == 0

    assert len(strategy.fit_clients) == 3

    print("3-client, 1-round FedAvg simulation PASSED")
    print(f"Participating clients: {strategy.fit_clients}")
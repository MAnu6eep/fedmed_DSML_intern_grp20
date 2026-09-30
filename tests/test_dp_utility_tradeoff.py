import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fedmed.core.training import train_one_epoch
from fedmed.privacy.differential_privacy import (
    DifferentialPrivacyConfig,
)


class SmallTrainingDataset(Dataset):
    """Deterministic dataset for DP utility comparison."""

    def __init__(self):
        generator = torch.Generator().manual_seed(42)

        self.images = torch.randn(
            8,
            4,
            generator=generator,
        )

        self.labels = torch.randn(
            8,
            2,
            generator=generator,
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            "image": self.images[index],
            "label": self.labels[index],
        }


class MSELossWrapper(torch.nn.Module):
    """Simple loss compatible with the local training loop."""

    def forward(self, outputs, targets):
        return torch.nn.functional.mse_loss(
            outputs,
            targets,
        )


def create_loader():
    return DataLoader(
        SmallTrainingDataset(),
        batch_size=2,
        shuffle=False,
    )


def run_experiment(
    name: str,
    dp_config: DifferentialPrivacyConfig,
) -> dict:
    """Run one training configuration and record utility metrics."""

    torch.manual_seed(42)
    np.random.seed(42)

    model = torch.nn.Linear(4, 2)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    loss_fn = MSELossWrapper()

    loss = train_one_epoch(
        model=model,
        dataloader=create_loader(),
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=torch.device("cpu"),
        dp_config=dp_config,
    )

    parameters_valid = all(
        torch.isfinite(parameter).all().item()
        for parameter in model.parameters()
    )

    return {
        "configuration": name,
        "dp_enabled": dp_config.enabled,
        "max_grad_norm": dp_config.max_grad_norm,
        "noise_multiplier": (
            dp_config.noise_multiplier
            if dp_config.enabled
            else None
        ),
        "epsilon": dp_config.epsilon,
        "delta": dp_config.delta,
        "train_loss": round(float(loss), 6),
        "parameters_valid": parameters_valid,
    }


def test_dp_utility_tradeoff():
    """Evaluate utility across baseline and DP noise settings."""

    configurations = [
        (
            "Baseline",
            DifferentialPrivacyConfig(
                enabled=False,
            ),
        ),
        (
            "DP Low Noise",
            DifferentialPrivacyConfig(
                enabled=True,
                max_grad_norm=1.0,
                noise_multiplier=0.1,
            ),
        ),
        (
            "DP Medium Noise",
            DifferentialPrivacyConfig(
                enabled=True,
                max_grad_norm=1.0,
                noise_multiplier=0.5,
            ),
        ),
        (
            "DP High Noise",
            DifferentialPrivacyConfig(
                enabled=True,
                max_grad_norm=1.0,
                noise_multiplier=1.0,
            ),
        ),
    ]

    results = [
        run_experiment(name, config)
        for name, config in configurations
    ]

    assert len(results) == 4

    for result in results:
        assert result["parameters_valid"]
        assert np.isfinite(result["train_loss"])

    output_dir = Path("experiments/outputs")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir / "dp_utility_tradeoff.json"
    )

    output_file.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nDP Utility Tradeoff Results:")
    for result in results:
        print(result)

    assert output_file.exists()
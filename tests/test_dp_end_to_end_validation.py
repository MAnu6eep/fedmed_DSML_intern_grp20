import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fedmed.config.experiment import ExperimentConfig
from fedmed.core.training import train_one_epoch
from fedmed.privacy.differential_privacy import (
    DifferentialPrivacyConfig,
)


class SmallTrainingDataset(Dataset):
    """Deterministic dataset for end-to-end DP validation."""

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


def create_model():
    torch.manual_seed(42)
    return torch.nn.Linear(4, 2)


def model_parameters_are_valid(model):
    return all(
        torch.isfinite(parameter).all().item()
        for parameter in model.parameters()
    )


def run_training(dp_config):
    model = create_model()

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

    return model, float(loss)


def test_dp_configuration_is_propagated():
    """Configured DP parameters must reach the client configuration."""

    config = ExperimentConfig(
        dp=DifferentialPrivacyConfig(
            enabled=True,
            max_grad_norm=1.0,
            noise_multiplier=0.5,
            epsilon=8.0,
            delta=1e-5,
        )
    )

    client_config = config.to_client_config()

    assert client_config["dp_enabled"] is True
    assert client_config["dp_max_grad_norm"] == 1.0
    assert client_config["dp_noise_multiplier"] == 0.5
    assert client_config["dp_epsilon"] == 8.0
    assert client_config["dp_delta"] == 1e-5


def test_end_to_end_without_dp():
    """Training must complete successfully without DP."""

    config = DifferentialPrivacyConfig(
        enabled=False,
    )

    model, loss = run_training(config)

    assert np.isfinite(loss)
    assert model_parameters_are_valid(model)


def test_end_to_end_with_dp():
    """Training must complete successfully with DP enabled."""

    config = DifferentialPrivacyConfig(
        enabled=True,
        max_grad_norm=1.0,
        noise_multiplier=0.5,
        epsilon=8.0,
        delta=1e-5,
    )

    model, loss = run_training(config)

    assert np.isfinite(loss)
    assert model_parameters_are_valid(model)


def test_dp_changes_training_result():
    """Enabled DP should alter the training result relative to baseline."""

    baseline_config = DifferentialPrivacyConfig(
        enabled=False,
    )

    dp_config = DifferentialPrivacyConfig(
        enabled=True,
        max_grad_norm=1.0,
        noise_multiplier=0.5,
        epsilon=8.0,
        delta=1e-5,
    )

    torch.manual_seed(42)
    np.random.seed(42)
    _, baseline_loss = run_training(baseline_config)

    torch.manual_seed(42)
    np.random.seed(42)
    _, dp_loss = run_training(dp_config)

    assert np.isfinite(baseline_loss)
    assert np.isfinite(dp_loss)
    assert baseline_loss != dp_loss


def test_privacy_parameters_are_recorded():
    """Configured epsilon and delta must be preserved."""

    config = DifferentialPrivacyConfig(
        enabled=True,
        max_grad_norm=1.0,
        noise_multiplier=0.5,
        epsilon=8.0,
        delta=1e-5,
    )

    config.validate()

    result = {
        "dp_enabled": config.enabled,
        "noise_multiplier": config.noise_multiplier,
        "max_grad_norm": config.max_grad_norm,
        "epsilon": config.epsilon,
        "delta": config.delta,
    }

    assert result["dp_enabled"] is True
    assert result["epsilon"] == 8.0
    assert result["delta"] == 1e-5

    output_dir = Path("experiments/outputs")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir / "dp_end_to_end_validation.json"
    )

    output_file.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    assert output_file.exists()
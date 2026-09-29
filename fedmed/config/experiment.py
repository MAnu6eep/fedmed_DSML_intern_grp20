"""Experiment configuration for centralized and federated training."""

from dataclasses import dataclass, field
from typing import Any, Dict


SUPPORTED_STRATEGIES = {
    "centralized",
    "fedavg",
    "fedprox",
    "scaffold",
}


@dataclass
class DatasetConfig:
    """Dataset and partition configuration."""

    name: str = "brats"
    partition: str = "non_iid"
    dirichlet_alpha: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Dataset name must not be empty.")

        if not self.partition.strip():
            raise ValueError("Partition name must not be empty.")

        if self.dirichlet_alpha <= 0:
            raise ValueError(
                "dirichlet_alpha must be greater than 0."
            )


@dataclass
class ExperimentConfig:
    """Configuration shared by all experiment strategies."""

    strategy: str = "fedavg"

    num_rounds: int = 3
    local_epochs: int = 1
    learning_rate: float = 1e-4
    batch_size: int = 1

    dataset: DatasetConfig = field(
        default_factory=DatasetConfig
    )

    # FedProx-specific parameter
    proximal_mu: float = 0.01

    def __post_init__(self) -> None:
        self.strategy = self.strategy.strip().lower()

        if self.strategy not in SUPPORTED_STRATEGIES:
            supported = ", ".join(
                sorted(SUPPORTED_STRATEGIES)
            )
            raise ValueError(
                f"Unsupported strategy: '{self.strategy}'. "
                f"Expected one of: {supported}."
            )

        if self.num_rounds <= 0:
            raise ValueError(
                "num_rounds must be greater than 0."
            )

        if self.local_epochs <= 0:
            raise ValueError(
                "local_epochs must be greater than 0."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than 0."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0."
            )

        if self.proximal_mu < 0:
            raise ValueError(
                "proximal_mu must be non-negative."
            )

    @property
    def is_federated(self) -> bool:
        """Return whether this configuration uses Flower."""
        return self.strategy != "centralized"

    def to_client_config(self) -> Dict[str, Any]:
        """Build the Flower fit configuration."""

        config: Dict[str, Any] = {
            "local_epochs": self.local_epochs,
            "learning_rate": self.learning_rate,
        }

        if self.strategy == "fedprox":
            config["proximal_mu"] = self.proximal_mu

        if self.strategy == "scaffold":
            config["scaffold"] = True

        return config

    def to_server_config(self) -> Dict[str, Any]:
        """Build the Flower server configuration."""

        return {
            "strategy": self.strategy,
            "num-server-rounds": self.num_rounds,
        }
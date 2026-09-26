"""Differential privacy scaffolding for FedMed."""

from dataclasses import dataclass


@dataclass
class DifferentialPrivacyConfig:
    """Configuration for differential privacy."""

    enabled: bool = False
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    epsilon: float | None = None
    delta: float | None = None

    def validate(self) -> None:
        """Validate differential privacy configuration."""

        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be greater than 0.")

        if self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be non-negative.")

        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError("epsilon must be greater than 0.")

        if self.delta is not None and not 0 < self.delta < 1:
            raise ValueError("delta must be between 0 and 1.")


def apply_dp(config: DifferentialPrivacyConfig, gradients):
    """Apply differential privacy to gradients.

    This is a placeholder interface for future DP integration.
    Actual gradient clipping, Gaussian noise addition, and privacy
    accounting will be implemented when DP is integrated into training.
    """
    if not config.enabled:
        return gradients

    raise NotImplementedError(
        "Differential privacy training integration is not implemented yet."
    )
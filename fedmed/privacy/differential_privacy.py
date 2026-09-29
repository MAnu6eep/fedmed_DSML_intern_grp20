"""Differential privacy utilities for FedMed."""

from dataclasses import dataclass

import numpy as np


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


class DifferentialPrivacy:
    """Apply gradient clipping and Gaussian noise."""

    def __init__(self, config: DifferentialPrivacyConfig):
        self.config = config
        self.config.validate()

    def clip_gradients(self, gradients: np.ndarray) -> np.ndarray:
        """Clip gradients to the configured maximum L2 norm."""

        gradients = np.asarray(gradients, dtype=np.float64)

        norm = np.linalg.norm(gradients)

        if norm == 0:
            return gradients.copy()

        if norm <= self.config.max_grad_norm:
            return gradients.copy()

        scale = self.config.max_grad_norm / norm

        return gradients * scale

    def add_gaussian_noise(self, gradients: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to gradients."""

        gradients = np.asarray(gradients, dtype=np.float64)

        if self.config.noise_multiplier == 0:
            return gradients.copy()

        noise = np.random.normal(
            loc=0.0,
            scale=self.config.noise_multiplier,
            size=gradients.shape,
        )

        return gradients + noise

    def apply(self, gradients: np.ndarray) -> np.ndarray:
        """Apply differential privacy to gradients."""

        gradients = np.asarray(gradients, dtype=np.float64)

        if not self.config.enabled:
            return gradients.copy()

        clipped_gradients = self.clip_gradients(gradients)

        return self.add_gaussian_noise(clipped_gradients)


def apply_dp(
    config: DifferentialPrivacyConfig,
    gradients: np.ndarray,
) -> np.ndarray:
    """Apply configured differential privacy to gradients."""

    dp = DifferentialPrivacy(config)

    return dp.apply(gradients)
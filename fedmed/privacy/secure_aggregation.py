"""Secure aggregation integration helpers for FedMed.

This module connects SecAgg+ configuration with the federated
training workflow. It validates client participation and prepares
model updates for future secure aggregation.
"""

from typing import Dict, List, Sequence

import numpy as np

from fedmed.privacy.secagg_config import SecAggPlusConfig


class SecureAggregationManager:
    """Validate SecAgg+ participation and prepare model updates."""

    def __init__(self, config: SecAggPlusConfig) -> None:
        """Initialize the secure aggregation manager."""
        self.config = config

    def validate_participation(
        self,
        participating_clients: int,
    ) -> bool:
        """Validate that enough clients participate in a round.

        Raises:
            ValueError: If the participant count is invalid or below
                the configured SecAgg+ threshold.
        """
        if participating_clients < 0:
            raise ValueError(
                "Participating client count cannot be negative."
            )

        if participating_clients > self.config.num_clients:
            raise ValueError(
                "Participating clients cannot exceed the configured "
                "number of clients."
            )

        if participating_clients < self.config.threshold:
            raise ValueError(
                "Insufficient participating clients for SecAgg+ "
                f"reconstruction: received {participating_clients}, "
                f"required at least {self.config.threshold}."
            )

        return True

    def prepare_model_updates(
        self,
        model_updates: Sequence[Sequence[np.ndarray]],
    ) -> List[List[np.ndarray]]:
        """Validate and prepare client model updates for aggregation.

        Each client must provide the same number of parameter arrays.
        The method currently performs validation and copying only.
        Actual cryptographic secure aggregation will be integrated
        in a later implementation.
        """
        num_participants = len(model_updates)

        self.validate_participation(num_participants)

        if not model_updates:
            raise ValueError("No model updates were provided.")

        expected_parameter_count = len(model_updates[0])

        prepared_updates: List[List[np.ndarray]] = []

        for client_index, client_update in enumerate(
            model_updates
        ):
            if len(client_update) != expected_parameter_count:
                raise ValueError(
                    "Client "
                    f"{client_index} provided "
                    f"{len(client_update)} parameters, but expected "
                    f"{expected_parameter_count}."
                )

            prepared_updates.append(
                [
                    np.asarray(parameter).copy()
                    for parameter in client_update
                ]
            )

        return prepared_updates

    def get_round_requirements(self) -> Dict[str, int]:
        """Return client participation requirements for a round."""
        return {
            "num_clients": self.config.num_clients,
            "threshold": self.config.threshold,
        }
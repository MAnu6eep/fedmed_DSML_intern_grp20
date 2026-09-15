"""SCAFFOLD strategy foundation for federated learning.

This module defines the configuration and control-variate structures
required for a future SCAFFOLD implementation.

The actual SCAFFOLD training/update equations are intentionally not
implemented yet.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


NDArrayList = List[np.ndarray]


@dataclass
class SCAFFOLDConfig:
    """Configuration for SCAFFOLD federated learning experiments."""

    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 3
    min_evaluate_clients: int = 3
    min_available_clients: int = 3

    local_epochs: int = 1
    learning_rate: float = 1e-4

    server_learning_rate: float = 1.0
    client_learning_rate: float = 1.0

    def __post_init__(self) -> None:
        """Validate the SCAFFOLD configuration."""
        if not 0.0 < self.fraction_fit <= 1.0:
            raise ValueError("fraction_fit must be in (0, 1].")

        if not 0.0 < self.fraction_evaluate <= 1.0:
            raise ValueError("fraction_evaluate must be in (0, 1].")

        if self.min_fit_clients < 1:
            raise ValueError("min_fit_clients must be positive.")

        if self.min_evaluate_clients < 1:
            raise ValueError("min_evaluate_clients must be positive.")

        if self.min_available_clients < 1:
            raise ValueError("min_available_clients must be positive.")

        if self.local_epochs < 1:
            raise ValueError("local_epochs must be positive.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if self.server_learning_rate <= 0:
            raise ValueError("server_learning_rate must be positive.")

        if self.client_learning_rate <= 0:
            raise ValueError("client_learning_rate must be positive.")


@dataclass
class ServerControlVariate:
    """Server-side SCAFFOLD control variate.

    The control variate has one NumPy array for each trainable model
    parameter. It is initialized lazily once the model parameters are
    available.
    """

    values: Optional[NDArrayList] = field(default=None)

    def initialize(self, parameters: NDArrayList) -> None:
        """Initialize the server control variate to zero."""
        self.values = [
            np.zeros_like(parameter)
            for parameter in parameters
        ]

    def is_initialized(self) -> bool:
        """Return whether the control variate has been initialized."""
        return self.values is not None

    def get_values(self) -> NDArrayList:
        """Return the current server control variate."""
        if self.values is None:
            raise RuntimeError(
                "Server control variate has not been initialized."
            )

        return [
            value.copy()
            for value in self.values
        ]


@dataclass
class ClientControlVariate:
    """Client-side SCAFFOLD control variate.

    Each client maintains its own control variate independently.
    """

    client_id: str
    values: Optional[NDArrayList] = field(default=None)

    def initialize(self, parameters: NDArrayList) -> None:
        """Initialize the client control variate to zero."""
        self.values = [
            np.zeros_like(parameter)
            for parameter in parameters
        ]

    def is_initialized(self) -> bool:
        """Return whether the control variate has been initialized."""
        return self.values is not None

    def get_values(self) -> NDArrayList:
        """Return the current client control variate."""
        if self.values is None:
            raise RuntimeError(
                f"Client control variate for '{self.client_id}' "
                "has not been initialized."
            )

        return [
            value.copy()
            for value in self.values
        ]


@dataclass
class ControlVariateState:
    """Container for server and client SCAFFOLD control variates."""

    server: ServerControlVariate = field(
        default_factory=ServerControlVariate
    )
    clients: Dict[str, ClientControlVariate] = field(
        default_factory=dict
    )

    def register_client(self, client_id: str) -> ClientControlVariate:
        """Register a client and create its control variate."""
        if client_id not in self.clients:
            self.clients[client_id] = ClientControlVariate(
                client_id=client_id
            )

        return self.clients[client_id]

    def initialize(
        self,
        parameters: NDArrayList,
        client_ids: Optional[List[str]] = None,
    ) -> None:
        """Initialize server and optional client control variates."""
        self.server.initialize(parameters)

        if client_ids is not None:
            for client_id in client_ids:
                client = self.register_client(client_id)
                client.initialize(parameters)


class SCAFFOLDStrategy:
    """Foundation for a future SCAFFOLD federated strategy.

    This class deliberately focuses on strategy configuration and
    control-variate state management. Actual SCAFFOLD training logic
    will be added in a later implementation.
    """

    name = "scaffold"

    def __init__(
        self,
        config: Optional[SCAFFOLDConfig] = None,
    ) -> None:
        self.config = config or SCAFFOLDConfig()
        self.control_variates = ControlVariateState()

    def initialize_control_variates(
        self,
        parameters: NDArrayList,
        client_ids: Optional[List[str]] = None,
    ) -> None:
        """Initialize control variates from model parameters."""
        self.control_variates.initialize(
            parameters=parameters,
            client_ids=client_ids,
        )

    def register_client(
        self,
        client_id: str,
    ) -> ClientControlVariate:
        """Register a client for SCAFFOLD state management."""
        return self.control_variates.register_client(client_id)
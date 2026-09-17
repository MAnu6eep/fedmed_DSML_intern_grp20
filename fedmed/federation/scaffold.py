"""SCAFFOLD strategy and control-variate utilities."""

import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


NDArrayList = List[np.ndarray]


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SCAFFOLDConfig:
    """Configuration for SCAFFOLD federated learning."""

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
        """Validate SCAFFOLD configuration."""

        if not 0.0 < self.fraction_fit <= 1.0:
            raise ValueError(
                "fraction_fit must be in (0, 1]."
            )

        if not 0.0 < self.fraction_evaluate <= 1.0:
            raise ValueError(
                "fraction_evaluate must be in (0, 1]."
            )

        if self.min_fit_clients < 1:
            raise ValueError(
                "min_fit_clients must be positive."
            )

        if self.min_evaluate_clients < 1:
            raise ValueError(
                "min_evaluate_clients must be positive."
            )

        if self.min_available_clients < 1:
            raise ValueError(
                "min_available_clients must be positive."
            )

        if self.local_epochs < 1:
            raise ValueError(
                "local_epochs must be positive."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.server_learning_rate <= 0:
            raise ValueError(
                "server_learning_rate must be positive."
            )

        if self.client_learning_rate <= 0:
            raise ValueError(
                "client_learning_rate must be positive."
            )


# ============================================================================
# Server Control Variate
# ============================================================================


@dataclass
class ServerControlVariate:
    """Global SCAFFOLD server control variate c."""

    values: Optional[NDArrayList] = field(
        default=None
    )

    def initialize(
        self,
        parameters: NDArrayList,
    ) -> None:
        """Initialize server control variate to zero."""

        self.values = [
            np.zeros_like(parameter)
            for parameter in parameters
        ]

    def is_initialized(self) -> bool:
        """Return whether the control variate is initialized."""

        return self.values is not None

    def get_values(self) -> NDArrayList:
        """Return a copy of the server control variate."""

        if self.values is None:
            raise RuntimeError(
                "Server control variate has not been initialized."
            )

        return [
            value.copy()
            for value in self.values
        ]

    def update(
        self,
        client_control_variates: List[NDArrayList],
        learning_rate: float = 1.0,
    ) -> None:
        """Update server control variate.

        c <- c + lr * mean(c_i - c)
        """

        if self.values is None:
            raise RuntimeError(
                "Server control variate has not been initialized."
            )

        if not client_control_variates:
            return

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        expected_length = len(self.values)

        for client_cv in client_control_variates:
            if len(client_cv) != expected_length:
                raise ValueError(
                    "Client control variate does not match "
                    "server control variate."
                )

        for parameter_index in range(
            expected_length
        ):
            mean_delta = np.mean(
                [
                    client_cv[parameter_index]
                    - self.values[parameter_index]
                    for client_cv in client_control_variates
                ],
                axis=0,
            )

            self.values[parameter_index] += (
                learning_rate * mean_delta
            )


# ============================================================================
# Client Control Variate
# ============================================================================


@dataclass
class ClientControlVariate:
    """Local SCAFFOLD client control variate c_i."""

    client_id: str

    values: Optional[NDArrayList] = field(
        default=None
    )

    def initialize(
        self,
        parameters: NDArrayList,
    ) -> None:
        """Initialize client control variate to zero."""

        self.values = [
            np.zeros_like(parameter)
            for parameter in parameters
        ]

    def is_initialized(self) -> bool:
        """Return whether the client control variate is initialized."""

        return self.values is not None

    def get_values(self) -> NDArrayList:
        """Return a copy of the client control variate."""

        if self.values is None:
            raise RuntimeError(
                f"Client control variate for "
                f"'{self.client_id}' has not been initialized."
            )

        return [
            value.copy()
            for value in self.values
        ]

    def update(
        self,
        global_parameters: NDArrayList,
        local_parameters: NDArrayList,
        server_control_variate: NDArrayList,
        learning_rate: float,
        local_steps: int,
    ) -> None:
        """Update c_i after local SCAFFOLD training.

        c_i(new) = c_i(old) - c
                   - (w_local - w_global) / (K * eta)

        Equivalent form:

        c_i(new) = c_i(old) - c
                   + (w_global - w_local) / (K * eta)
        """

        if self.values is None:
            self.initialize(
                global_parameters
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if local_steps <= 0:
            raise ValueError(
                "local_steps must be positive."
            )

        if not (
            len(global_parameters)
            == len(local_parameters)
            == len(server_control_variate)
            == len(self.values)
        ):
            raise ValueError(
                "Control variates and model parameters "
                "must have the same length."
            )

        denominator = float(
            local_steps * learning_rate
        )

        updated_values: NDArrayList = []

        for (
            old_cv,
            global_param,
            local_param,
            server_cv,
        ) in zip(
            self.values,
            global_parameters,
            local_parameters,
            server_control_variate,
        ):
            new_cv = (
                old_cv
                - server_cv
                - (
                    local_param - global_param
                )
                / denominator
            )

            updated_values.append(
                np.asarray(
                    new_cv,
                    dtype=old_cv.dtype,
                )
            )

        self.values = updated_values


# ============================================================================
# SCAFFOLD Gradient Correction
# ============================================================================


def scaffold_gradient(
    gradient: np.ndarray,
    client_control: np.ndarray,
    server_control: np.ndarray,
) -> np.ndarray:
    """Apply the SCAFFOLD gradient correction.

    g_corrected = g - c_i + c
    """

    return (
        gradient
        - client_control
        + server_control
    )


# ============================================================================
# Serialization
# ============================================================================


def serialize_control_variate(
    control_variate: NDArrayList,
) -> bytes:
    """Serialize a control variate for Flower transmission."""

    if not isinstance(
        control_variate,
        list,
    ):
        raise TypeError(
            "Control variate must be a list of NumPy arrays."
        )

    return pickle.dumps(
        [
            np.asarray(value)
            for value in control_variate
        ],
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def deserialize_control_variate(
    data: bytes,
) -> NDArrayList:
    """Deserialize a Flower-transmitted control variate."""

    if not isinstance(data, bytes):
        raise TypeError(
            "Serialized control variate must be bytes."
        )

    values = pickle.loads(data)

    if not isinstance(values, list):
        raise ValueError(
            "Serialized control variate must contain a list."
        )

    return [
        np.asarray(value)
        for value in values
    ]


# ============================================================================
# Control Variate State
# ============================================================================


@dataclass
class ControlVariateState:
    """Container for server and hospital client control variates."""

    server: ServerControlVariate = field(
        default_factory=ServerControlVariate
    )

    clients: Dict[
        str,
        ClientControlVariate,
    ] = field(
        default_factory=dict
    )

    def register_client(
        self,
        client_id: str,
    ) -> ClientControlVariate:
        """Register a hospital client."""

        if client_id not in self.clients:
            self.clients[client_id] = (
                ClientControlVariate(
                    client_id=client_id
                )
            )

        return self.clients[client_id]

    def initialize(
        self,
        parameters: NDArrayList,
        client_ids: Optional[
            List[str]
        ] = None,
    ) -> None:
        """Initialize server and optional client variates."""

        self.server.initialize(
            parameters
        )

        if client_ids is not None:
            for client_id in client_ids:
                client = self.register_client(
                    client_id
                )

                client.initialize(
                    parameters
                )


# ============================================================================
# SCAFFOLD Strategy
# ============================================================================


class SCAFFOLDStrategy:
    """Core SCAFFOLD control-variate state and operations."""

    name = "scaffold"

    def __init__(
        self,
        config: Optional[
            SCAFFOLDConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or SCAFFOLDConfig()
        )

        self.control_variates = (
            ControlVariateState()
        )

    def initialize_control_variates(
        self,
        parameters: NDArrayList,
        client_ids: Optional[
            List[str]
        ] = None,
    ) -> None:
        """Initialize server/client control variates."""

        self.control_variates.initialize(
            parameters=parameters,
            client_ids=client_ids,
        )

    def register_client(
        self,
        client_id: str,
    ) -> ClientControlVariate:
        """Register a hospital client."""

        return (
            self.control_variates
            .register_client(client_id)
        )

    def get_server_control_variate(
        self,
    ) -> NDArrayList:
        """Return server c for transmission."""

        return (
            self.control_variates
            .server
            .get_values()
        )

    def get_client_control_variate(
        self,
        client_id: str,
    ) -> NDArrayList:
        """Return client c_i."""

        client = self.register_client(
            client_id
        )

        if not client.is_initialized():
            if not (
                self.control_variates
                .server
                .is_initialized()
            ):
                raise RuntimeError(
                    "Control variates have not been initialized."
                )

            client.initialize(
                self.control_variates
                .server
                .get_values()
            )

        return client.get_values()

    def update_client_control_variate(
        self,
        client_id: str,
        global_parameters: NDArrayList,
        local_parameters: NDArrayList,
        local_steps: int,
    ) -> None:
        """Update client c_i after local training."""

        client = self.register_client(
            client_id
        )

        if not client.is_initialized():
            client.initialize(
                global_parameters
            )

        client.update(
            global_parameters=(
                global_parameters
            ),
            local_parameters=(
                local_parameters
            ),
            server_control_variate=(
                self.get_server_control_variate()
            ),
            learning_rate=(
                self.config.client_learning_rate
            ),
            local_steps=local_steps,
        )

    def update_server_control_variate(
        self,
        client_ids: List[str],
    ) -> None:
        """Aggregate participating client control variates."""

        client_variates = [
            self.get_client_control_variate(
                client_id
            )
            for client_id in client_ids
        ]

        self.control_variates.server.update(
            client_control_variates=(
                client_variates
            ),
            learning_rate=(
                self.config.server_learning_rate
            ),
        )
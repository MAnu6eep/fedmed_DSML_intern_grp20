import numpy as np
import pytest

from fedmed.federation.scaffold import (
    ClientControlVariate,
    SCAFFOLDConfig,
    SCAFFOLDStrategy,
    ServerControlVariate,
)


def test_scaffold_config_defaults():
    config = SCAFFOLDConfig()

    assert config.fraction_fit == 1.0
    assert config.min_fit_clients == 3
    assert config.local_epochs == 1
    assert config.learning_rate == 1e-4


def test_scaffold_config_rejects_invalid_learning_rate():
    with pytest.raises(ValueError):
        SCAFFOLDConfig(learning_rate=0.0)


def test_server_control_variate_initializes_to_zero():
    parameters = [
        np.ones((2, 2), dtype=np.float32),
        np.ones(3, dtype=np.float32),
    ]

    control_variate = ServerControlVariate()
    control_variate.initialize(parameters)

    assert control_variate.is_initialized()
    assert len(control_variate.values) == 2
    assert np.array_equal(
        control_variate.values[0],
        np.zeros((2, 2), dtype=np.float32),
    )
    assert np.array_equal(
        control_variate.values[1],
        np.zeros(3, dtype=np.float32),
    )


def test_client_control_variate_keeps_client_identity():
    control_variate = ClientControlVariate(
        client_id="hospital-1"
    )

    assert control_variate.client_id == "hospital-1"
    assert not control_variate.is_initialized()


def test_scaffold_strategy_initializes_server_and_clients():
    parameters = [
        np.ones(4, dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
    ]

    strategy = SCAFFOLDStrategy()

    strategy.initialize_control_variates(
        parameters,
        client_ids=["hospital-1", "hospital-2"],
    )

    assert strategy.name == "scaffold"
    assert strategy.control_variates.server.is_initialized()
    assert len(strategy.control_variates.clients) == 2

    assert strategy.control_variates.clients[
        "hospital-1"
    ].is_initialized()

    assert strategy.control_variates.clients[
        "hospital-2"
    ].is_initialized()


def test_register_client_is_idempotent():
    strategy = SCAFFOLDStrategy()

    first = strategy.register_client("hospital-1")
    second = strategy.register_client("hospital-1")

    assert first is second
    assert len(strategy.control_variates.clients) == 1


def test_uninitialized_control_variate_raises():
    control_variate = ServerControlVariate()

    with pytest.raises(RuntimeError):
        control_variate.get_values()
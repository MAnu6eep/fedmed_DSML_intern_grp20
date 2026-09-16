import numpy as np
import pytest

from fedmed.federation.scaffold import (
    ClientControlVariate,
    SCAFFOLDConfig,
    SCAFFOLDStrategy,
    ServerControlVariate,
    scaffold_gradient,
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


def test_scaffold_gradient_correction():
    gradient = np.array([5.0, 6.0], dtype=np.float32)
    client_control = np.array([2.0, 1.0], dtype=np.float32)
    server_control = np.array([1.0, 3.0], dtype=np.float32)

    corrected = scaffold_gradient(
        gradient,
        client_control,
        server_control,
    )

    expected = np.array([4.0, 8.0], dtype=np.float32)

    assert np.allclose(corrected, expected)


def test_client_control_variate_update():
    global_parameters = [
        np.array([1.0, 2.0], dtype=np.float32),
    ]

    local_parameters = [
        np.array([0.8, 2.4], dtype=np.float32),
    ]

    server_control = [
        np.array([0.1, 0.2], dtype=np.float32),
    ]

    client = ClientControlVariate(
        client_id="hospital-1"
    )

    client.initialize(global_parameters)

    client.update(
        global_parameters=global_parameters,
        local_parameters=local_parameters,
        server_control_variate=server_control,
        learning_rate=0.1,
        local_steps=2,
    )

    expected = np.array(
        [0.9, -2.2],
        dtype=np.float32,
    )

    assert np.allclose(
        client.values[0],
        expected,
    )


def test_server_control_variate_update():
    server = ServerControlVariate()

    parameters = [
        np.zeros(2, dtype=np.float32),
    ]

    server.initialize(parameters)

    clients = [
        [np.array([1.0, 2.0], dtype=np.float32)],
        [np.array([3.0, 4.0], dtype=np.float32)],
    ]

    server.update(clients)

    expected = np.array(
        [2.0, 3.0],
        dtype=np.float32,
    )

    assert np.allclose(
        server.values[0],
        expected,
    )


def test_strategy_exposes_server_and_client_control_variates():
    parameters = [
        np.zeros(2, dtype=np.float32),
    ]

    strategy = SCAFFOLDStrategy()

    strategy.initialize_control_variates(
        parameters,
        client_ids=["hospital-1"],
    )

    server_cv = strategy.get_server_control_variate()
    client_cv = strategy.get_client_control_variate(
        "hospital-1"
    )

    assert np.allclose(
        server_cv[0],
        np.zeros(2),
    )

    assert np.allclose(
        client_cv[0],
        np.zeros(2),
    )
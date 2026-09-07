
"""Tests for SecAgg+ client participation and threshold behavior."""

import numpy as np
import pytest

from fedmed.privacy.secagg_config import SecAggPlusConfig
from fedmed.privacy.secure_aggregation import SecureAggregationManager


@pytest.fixture
def secagg_manager() -> SecureAggregationManager:
    """Create a SecAgg+ manager with a 2-of-3 configuration."""
    config = SecAggPlusConfig(
        num_clients=3,
        threshold=2,
    )
    return SecureAggregationManager(config)


def test_all_three_clients_participate(
    secagg_manager: SecureAggregationManager,
) -> None:
    """Verify aggregation requirements are satisfied with 3/3 clients."""
    assert secagg_manager.validate_participation(3) is True


def test_two_of_three_clients_meet_threshold(
    secagg_manager: SecureAggregationManager,
) -> None:
    """Verify the configured 2-of-3 threshold succeeds."""
    assert secagg_manager.validate_participation(2) is True


def test_one_of_three_clients_fails_threshold(
    secagg_manager: SecureAggregationManager,
) -> None:
    """Verify aggregation is rejected when only 1/3 clients participate."""
    with pytest.raises(
        ValueError,
        match="Insufficient participating clients",
    ):
        secagg_manager.validate_participation(1)


def test_zero_clients_fail_threshold(
    secagg_manager: SecureAggregationManager,
) -> None:
    """Verify aggregation is rejected when no clients participate."""
    with pytest.raises(
        ValueError,
        match="Insufficient participating clients",
    ):
        secagg_manager.validate_participation(0)


def test_participating_clients_cannot_exceed_configured_clients(
    secagg_manager: SecureAggregationManager,
) -> None:
    """Verify participation cannot exceed the configured client count."""
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        secagg_manager.validate_participation(4)


def test_prepare_model_updates_with_three_clients(
    secagg_manager: SecureAggregationManager,
) -> None:
    """Verify model updates can be prepared with all 3 clients."""
    model_updates = [
        [np.array([1.0, 2.0, 3.0])],
        [np.array([4.0, 5.0, 6.0])],
        [np.array([7.0, 8.0, 9.0])],
    ]

    prepared = secagg_manager.prepare_model_updates(model_updates)

    assert len(prepared) == 3
    assert len(prepared[0]) == 1
    np.testing.assert_array_equal(
        prepared[0][0],
        np.array([1.0, 2.0, 3.0]),
    )


def test_prepare_model_updates_with_two_clients(
    secagg_manager: SecureAggregationManager,
) -> None:
    """Verify model updates can be prepared at the 2-client threshold."""
    model_updates = [
        [np.array([1.0, 2.0, 3.0])],
        [np.array([4.0, 5.0, 6.0])],
    ]

    prepared = secagg_manager.prepare_model_updates(model_updates)

    assert len(prepared) == 2


def test_prepare_model_updates_with_one_client_fails(
    secagg_manager: SecureAggregationManager,
) -> None:
    """Verify model preparation fails below the 2-client threshold."""
    model_updates = [
        [np.array([1.0, 2.0, 3.0])],
    ]

    with pytest.raises(
        ValueError,
        match="Insufficient participating clients",
    ):
        secagg_manager.prepare_model_updates(model_updates)


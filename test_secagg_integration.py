"""Tests for SecAgg+ federation integration."""

import numpy as np

from fedmed.privacy.secagg_config import SecAggPlusConfig
from fedmed.privacy.secure_aggregation import (
    SecureAggregationManager,
)


def run_tests() -> None:
    """Run basic SecAgg+ integration tests."""

    config = SecAggPlusConfig(
        num_clients=3,
        threshold=2,
    )

    manager = SecureAggregationManager(config)

    print("=== SecAgg+ Integration Test ===")

    # Test 1: Three clients participate
    assert manager.validate_participation(3)
    print("✓ Test 1 passed: 3 participating clients accepted.")

    # Test 2: Threshold number of clients participate
    assert manager.validate_participation(2)
    print("✓ Test 2 passed: threshold of 2 clients accepted.")

    # Test 3: Below threshold
    try:
        manager.validate_participation(1)
    except ValueError as error:
        print(
            "✓ Test 3 passed: below-threshold participation "
            f"rejected ({error})"
        )
    else:
        raise AssertionError(
            "Expected below-threshold participation to fail."
        )

    # Test 4: Prepare model updates from three clients
    model_updates = [
        [
            np.array([1.0, 2.0]),
            np.array([0.1, 0.2]),
        ],
        [
            np.array([1.5, 2.5]),
            np.array([0.3, 0.4]),
        ],
        [
            np.array([2.0, 3.0]),
            np.array([0.5, 0.6]),
        ],
    ]

    prepared_updates = manager.prepare_model_updates(
        model_updates
    )

    assert len(prepared_updates) == 3
    assert len(prepared_updates[0]) == 2

    print(
        "✓ Test 4 passed: model updates from 3 clients "
        "prepared successfully."
    )

    print("\n✓ All SecAgg+ integration tests passed.")


if __name__ == "__main__":
    run_tests()
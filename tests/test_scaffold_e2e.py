import numpy as np

from experiments.scaffold_convergence_comparison import (
    DIRICHLET_ALPHA,
    NUM_CLIENTS,
    NUM_ROUNDS,
    SEED,
    partition_dirichlet,
    run_experiment,
)


def test_scaffold_end_to_end_non_iid_federated_training():
    volumes = [
        f"volume_{i:02d}"
        for i in range(12)
    ]

    labels = [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
    ]

    partitions = partition_dirichlet(
        volumes,
        labels,
        num_clients=NUM_CLIENTS,
        alpha=DIRICHLET_ALPHA,
        seed=SEED,
    )

    # Existing Non-IID partition contains all configured clients.
    assert len(partitions) == NUM_CLIENTS
    assert all(
        len(client_volumes) > 0
        for client_volumes in partitions.values()
    )

    result = run_experiment(
        "SCAFFOLD",
        partitions,
    )

    assert result["strategy"] == "SCAFFOLD"
    assert result["distribution"] == "Non-IID"

    history = result["history"]

    # All configured federated rounds completed.
    assert len(history) == NUM_ROUNDS

    # No federation failures occurred.
    assert all(
        metrics["failures"] == 0
        for metrics in history
    )

    # SCAFFOLD-specific metrics are recorded in the history.
    scaffold_clients = [
        metrics["scaffold_clients"]
        for metrics in history
    ]

    scaffold_rounds = [
        metrics["scaffold_round"]
        for metrics in history
    ]

    scaffold_enabled = [
        metrics["scaffold"]
        for metrics in history
    ]

    assert scaffold_clients == [NUM_CLIENTS] * NUM_ROUNDS
    assert scaffold_rounds == list(range(1, NUM_ROUNDS + 1))
    assert scaffold_enabled == [True] * NUM_ROUNDS

    # Global model parameters changed after rounds.
    parameter_deltas = [
        metrics["parameter_delta"]
        for metrics in history
    ]

    assert parameter_deltas[0] == 0.0
    assert all(
        delta > 0.0
        for delta in parameter_deltas[1:]
    )

    # Convergence metrics were recorded.
    for metrics in history:
        assert np.isfinite(metrics["global_loss"])
        assert np.isfinite(metrics["val_dice"])
        assert np.isfinite(metrics["parameter_delta"])
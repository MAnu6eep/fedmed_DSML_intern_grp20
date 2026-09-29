import pytest

from fedmed.config.experiment import (
    DatasetConfig,
    ExperimentConfig,
    SUPPORTED_STRATEGIES,
)


def test_supported_strategies():
    assert SUPPORTED_STRATEGIES == {
        "centralized",
        "fedavg",
        "fedprox",
        "scaffold",
    }


@pytest.mark.parametrize(
    "strategy",
    ["centralized", "fedavg", "fedprox", "scaffold"],
)
def test_valid_strategies(strategy):
    config = ExperimentConfig(strategy=strategy)

    assert config.strategy == strategy


def test_strategy_is_normalized():
    config = ExperimentConfig(strategy="  FedProx ")

    assert config.strategy == "fedprox"


def test_invalid_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unsupported strategy"):
        ExperimentConfig(strategy="unknown")


def test_common_parameters():
    config = ExperimentConfig(
        strategy="fedavg",
        num_rounds=5,
        local_epochs=2,
        learning_rate=0.001,
        batch_size=4,
    )

    assert config.num_rounds == 5
    assert config.local_epochs == 2
    assert config.learning_rate == 0.001
    assert config.batch_size == 4


def test_dataset_configuration():
    config = ExperimentConfig(
        dataset=DatasetConfig(
            name="brats",
            partition="non_iid",
            dirichlet_alpha=0.5,
            seed=42,
        )
    )

    assert config.dataset.name == "brats"
    assert config.dataset.partition == "non_iid"
    assert config.dataset.dirichlet_alpha == 0.5
    assert config.dataset.seed == 42


def test_fedprox_configuration():
    config = ExperimentConfig(
        strategy="fedprox",
        proximal_mu=0.05,
    )

    assert config.proximal_mu == 0.05

    client_config = config.to_client_config()

    assert client_config["proximal_mu"] == 0.05


def test_scaffold_configuration():
    config = ExperimentConfig(strategy="scaffold")

    client_config = config.to_client_config()

    assert client_config["scaffold"] is True


def test_fedavg_client_configuration():
    config = ExperimentConfig(
        strategy="fedavg",
        local_epochs=2,
        learning_rate=0.0005,
    )

    assert config.to_client_config() == {
        "local_epochs": 2,
        "learning_rate": 0.0005,
    }


def test_centralized_is_not_federated():
    config = ExperimentConfig(strategy="centralized")

    assert config.is_federated is False


def test_federated_strategies_are_federated():
    for strategy in ["fedavg", "fedprox", "scaffold"]:
        config = ExperimentConfig(strategy=strategy)
        assert config.is_federated is True


def test_invalid_training_parameters():
    with pytest.raises(ValueError):
        ExperimentConfig(num_rounds=0)

    with pytest.raises(ValueError):
        ExperimentConfig(local_epochs=0)

    with pytest.raises(ValueError):
        ExperimentConfig(learning_rate=0)

    with pytest.raises(ValueError):
        ExperimentConfig(batch_size=0)


def test_invalid_fedprox_mu():
    with pytest.raises(ValueError):
        ExperimentConfig(
            strategy="fedprox",
            proximal_mu=-0.1,
        )

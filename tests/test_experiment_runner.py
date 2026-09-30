"""Tests for the common experiment runner."""

from pathlib import Path

import pytest

from run_experiment import (
    build_experiment_config,
    create_partitions,
    load_yaml_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_CONFIG = {
    "strategy": "fedavg",
    "num_rounds": 3,
    "local_epochs": 1,
    "learning_rate": 0.0001,
    "batch_size": 1,
    "dataset": {
        "name": "brats",
        "partition": "non_iid",
        "dirichlet_alpha": 0.5,
        "seed": 42,
    },
}


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def test_load_valid_yaml(tmp_path):
    config_file = tmp_path / "fedavg.yaml"

    config_file.write_text(
        """
strategy: fedavg
num_rounds: 3
local_epochs: 1
learning_rate: 0.0001
batch_size: 1

dataset:
  name: brats
  partition: non_iid
  dirichlet_alpha: 0.5
  seed: 42
""",
        encoding="utf-8",
    )

    config = load_yaml_config(config_file)

    assert config["strategy"] == "fedavg"
    assert config["num_rounds"] == 3
    assert config["dataset"]["seed"] == 42


def test_missing_yaml_file():
    with pytest.raises(FileNotFoundError):
        load_yaml_config(
            Path("does_not_exist.yaml")
        )


def test_rejects_unknown_config_field(tmp_path):
    config_file = tmp_path / "invalid.yaml"

    config_file.write_text(
        """
strategy: fedavg
unknown_field: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported configuration"):
        load_yaml_config(config_file)


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

def test_build_fedavg_config():
    config = build_experiment_config(
        VALID_CONFIG
    )

    assert config.strategy == "fedavg"
    assert config.num_rounds == 3
    assert config.local_epochs == 1
    assert config.learning_rate == 0.0001
    assert config.batch_size == 1
    assert config.dataset.name == "brats"
    assert config.dataset.partition == "non_iid"
    assert config.dataset.dirichlet_alpha == 0.5
    assert config.dataset.seed == 42


@pytest.mark.parametrize(
    "strategy",
    [
        "fedavg",
        "fedprox",
        "scaffold",
        "centralized",
    ],
)
def test_supported_strategies(strategy):
    config_data = dict(VALID_CONFIG)
    config_data["strategy"] = strategy

    config = build_experiment_config(
        config_data
    )

    assert config.strategy == strategy


def test_invalid_strategy():
    config_data = dict(VALID_CONFIG)
    config_data["strategy"] = "invalid_strategy"

    with pytest.raises(
        ValueError,
        match="Unsupported strategy",
    ):
        build_experiment_config(config_data)


def test_invalid_num_rounds():
    config_data = dict(VALID_CONFIG)
    config_data["num_rounds"] = 0

    with pytest.raises(
        ValueError,
        match="num_rounds",
    ):
        build_experiment_config(config_data)


def test_invalid_learning_rate():
    config_data = dict(VALID_CONFIG)
    config_data["learning_rate"] = 0

    with pytest.raises(
        ValueError,
        match="learning_rate",
    ):
        build_experiment_config(config_data)


def test_invalid_dataset_partition():
    config_data = dict(VALID_CONFIG)
    config_data["dataset"] = {
        "name": "brats",
        "partition": "",
        "dirichlet_alpha": 0.5,
        "seed": 42,
    }

    with pytest.raises(
        ValueError,
        match="Partition name",
    ):
        build_experiment_config(config_data)


# ---------------------------------------------------------------------------
# Partition reproducibility
# ---------------------------------------------------------------------------

def test_partition_is_reproducible():
    config = build_experiment_config(
        VALID_CONFIG
    )

    first = create_partitions(config)
    second = create_partitions(config)

    assert first == second


def test_different_seed_changes_partition():
    config_one = build_experiment_config(
        VALID_CONFIG
    )

    config_data = dict(VALID_CONFIG)
    config_data["dataset"] = dict(
        VALID_CONFIG["dataset"]
    )
    config_data["dataset"]["seed"] = 123

    config_two = build_experiment_config(
        config_data
    )

    first = create_partitions(config_one)
    second = create_partitions(config_two)

    assert first != second


# ---------------------------------------------------------------------------
# Standardized benchmark configurations
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS = {
    "centralized.yaml": {
        "strategy": "centralized",
        "proximal_mu": None,
    },
    "fedavg.yaml": {
        "strategy": "fedavg",
        "proximal_mu": None,
    },
    "fedprox.yaml": {
        "strategy": "fedprox",
        "proximal_mu": 0.01,
    },
    "scaffold.yaml": {
        "strategy": "scaffold",
        "proximal_mu": None,
    },
}

@pytest.mark.parametrize(
    "config_name, expected",
    BENCHMARK_CONFIGS.items(),
)
def test_standardized_benchmark_config(config_name, expected):
    """Verify every benchmark YAML maps to the expected experiment config."""
    config_path = Path("experiments") / config_name

    raw_config = load_yaml_config(config_path)
    config = build_experiment_config(raw_config)

    assert config.strategy == expected["strategy"]
    assert config.num_rounds == 3
    assert config.local_epochs == 1
    assert config.learning_rate == 0.0001
    assert config.batch_size == 1

    assert config.dataset.name == "brats"
    assert config.dataset.partition == "non_iid"
    assert config.dataset.dirichlet_alpha == 0.5
    assert config.dataset.seed == 42

    if expected["strategy"] == "fedprox":
        assert config.proximal_mu == expected["proximal_mu"]
"""Tests for the common experiment runner."""

import json
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


def test_standardized_configs_use_reproducible_partitioning():
    """Verify every benchmark configuration produces the same partition twice."""
    for config_name in BENCHMARK_CONFIGS:
        config_path = Path("experiments") / config_name

        config = build_experiment_config(
            load_yaml_config(config_path)
        )

        first = create_partitions(config)
        second = create_partitions(config)

        assert first == second


# ---------------------------------------------------------------------------
# Executed benchmark result validation
# ---------------------------------------------------------------------------

RESULT_CONFIGS = {
    "fedavg_result.json": {
        "strategy": "FedAvg",
        "config_strategy": "fedavg",
        "proximal_mu": None,
    },
    "fedprox_result.json": {
        "strategy": "FedProx",
        "config_strategy": "fedprox",
        "proximal_mu": 0.01,
    },
    "scaffold_result.json": {
        "strategy": "SCAFFOLD",
        "config_strategy": "scaffold",
        "proximal_mu": None,
    },
}


@pytest.mark.parametrize(
    "result_name, expected",
    RESULT_CONFIGS.items(),
)
def test_federated_benchmark_result(result_name, expected):
    """Verify completed federated benchmark results are valid."""
    result_path = Path("experiments") / "outputs" / result_name

    assert result_path.exists()

    with result_path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    assert result["strategy"] == expected["strategy"]

    settings = result["settings"]

    assert settings["num_clients"] == 3
    assert settings["num_rounds"] == 3
    assert settings["local_epochs"] == 1
    assert settings["learning_rate"] == 0.0001
    assert settings["batch_size"] == 1
    assert settings["proximal_mu"] == expected["proximal_mu"]

    assert settings["partition"] == "non_iid"
    assert settings["dirichlet_alpha"] == 0.5
    assert settings["seed"] == 42

    assert result["config"]["strategy"] == expected["config_strategy"]

    history = result["history"]

    assert len(history) == 3
    assert [entry["round"] for entry in history] == [1, 2, 3]

    for entry in history:
        assert entry["participants"] == 3
        assert entry["failures"] == 0


def test_scaffold_result_contains_control_variate_execution():
    """Verify the SCAFFOLD result records control-variate participation."""
    result_path = (
        Path("experiments")
        / "outputs"
        / "scaffold_result.json"
    )

    with result_path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    for entry in result["history"]:
        assert entry["scaffold"] is True
        assert entry["scaffold_clients"] == 3

    assert [entry["scaffold_round"] for entry in result["history"]] == [
        1,
        2,
        3,
    ]
"""
FedMed common experiment runner.

Run any configured experiment with:

    python run_experiment.py --config experiments/fedavg.yaml
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# FedMed imports
# ---------------------------------------------------------------------------

from fedmed.config.experiment import (
    DatasetConfig,
    ExperimentConfig,
)

from fedmed.data.partitioner import partition_dirichlet

from fedmed.privacy.differential_privacy import (
    DifferentialPrivacyConfig,
)

from experiments.scaffold_convergence_comparison import (
    NUM_CLIENTS,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Supported configuration fields
# ---------------------------------------------------------------------------

SUPPORTED_CONFIG_KEYS = {
    "strategy",
    "num_rounds",
    "local_epochs",
    "learning_rate",
    "batch_size",
    "dataset",
    "proximal_mu",
    "dp",
}

SUPPORTED_DATASET_KEYS = {
    "name",
    "partition",
    "dirichlet_alpha",
    "seed",
}

SUPPORTED_DP_KEYS = {
    "enabled",
    "max_grad_norm",
    "noise_multiplier",
    "epsilon",
    "delta",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a configured FedMed experiment."
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the experiment YAML configuration.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load and validate the YAML configuration file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Configuration path is not a file: {path}"
        )

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            "Configuration file must use .yaml or .yml extension."
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML configuration: {exc}"
        ) from exc

    if data is None:
        raise ValueError(
            "Configuration file is empty."
        )

    if not isinstance(data, dict):
        raise ValueError(
            "Configuration root must be a YAML mapping."
        )

    unknown_keys = set(data) - SUPPORTED_CONFIG_KEYS

    if unknown_keys:
        raise ValueError(
            "Unsupported configuration field(s): "
            + ", ".join(sorted(unknown_keys))
        )

    return data


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

def build_experiment_config(
    raw_config: Dict[str, Any],
) -> ExperimentConfig:
    """Convert YAML data into ExperimentConfig."""

    dataset_data = raw_config.get("dataset", {})

    if dataset_data is None:
        dataset_data = {}

    if not isinstance(dataset_data, dict):
        raise ValueError(
            "'dataset' must be a mapping."
        )

    unknown_dataset_keys = (
        set(dataset_data) - SUPPORTED_DATASET_KEYS
    )

    if unknown_dataset_keys:
        raise ValueError(
            "Unsupported dataset field(s): "
            + ", ".join(sorted(unknown_dataset_keys))
        )

    dataset = DatasetConfig(
        **dataset_data
    )

    # --------------------------------------------------------------
    # Differential Privacy configuration
    # --------------------------------------------------------------

    dp_data = raw_config.get("dp", {})

    if dp_data is None:
        dp_data = {}

    if not isinstance(dp_data, dict):
        raise ValueError(
            "'dp' must be a mapping."
        )

    unknown_dp_keys = (
        set(dp_data) - SUPPORTED_DP_KEYS
    )

    if unknown_dp_keys:
        raise ValueError(
            "Unsupported DP field(s): "
            + ", ".join(sorted(unknown_dp_keys))
        )

    dp_config = DifferentialPrivacyConfig(
        **dp_data
    )

    # --------------------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------------------

    experiment_fields = {
        key: value
        for key, value in raw_config.items()
        if key not in {
            "dataset",
            "dp",
        }
    }

    return ExperimentConfig(
        dataset=dataset,
        dp=dp_config,
        **experiment_fields,
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_reproducibility(seed: int) -> None:
    """Set deterministic random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Dataset partitioning
# ---------------------------------------------------------------------------

def create_partitions(
    config: ExperimentConfig,
) -> Dict[str, list]:
    """
    Create the deterministic hospital partition used by
    the current FedMed experiment setup.
    """

    if config.dataset.partition != "non_iid":
        raise ValueError(
            "The common runner currently supports "
            "the 'non_iid' partition."
        )

    # Current synthetic FedMed experiment dataset.
    volumes = [
        f"volume_{index:02d}"
        for index in range(12)
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
        alpha=config.dataset.dirichlet_alpha,
        seed=config.dataset.seed,
    )

    return partitions


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_from_config(
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """Launch the configured experiment."""

    set_reproducibility(
        config.dataset.seed
    )

    partitions = create_partitions(
        config
    )

    result = run_experiment(
        config,
        partitions,
    )

    if result is None:
        result = {}

    if not isinstance(result, dict):
        result = {
            "result": result
        }

    # Store the exact configuration used.
    result["config"] = {
        "strategy": config.strategy,
        "num_rounds": config.num_rounds,
        "local_epochs": config.local_epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "proximal_mu": (
            config.proximal_mu
            if config.strategy == "fedprox"
            else None
        ),
        "dataset": {
            "name": config.dataset.name,
            "partition": config.dataset.partition,
            "dirichlet_alpha": (
                config.dataset.dirichlet_alpha
            ),
            "seed": config.dataset.seed,
        },
        "dp": {
            "enabled": config.dp.enabled,
            "max_grad_norm": config.dp.max_grad_norm,
            "noise_multiplier": config.dp.noise_multiplier,
            "epsilon": config.dp.epsilon,
            "delta": config.dp.delta,
        },
    }

    # Store the exact partition used so the experiment
    # can be reproduced later.
    result["partitions"] = partitions

    return result


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_result(
    result: Dict[str, Any],
    config_path: Path,
) -> Path:
    """Save experiment result as JSON."""

    output_dir = (
        PROJECT_ROOT
        / "experiments"
        / "outputs"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{config_path.stem}_result.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            default=str,
        )

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    try:
        raw_config = load_yaml_config(
            args.config
        )

        config = build_experiment_config(
            raw_config
        )

        print("=" * 60)
        print("FedMed Common Experiment Runner")
        print("=" * 60)

        print(
            f"Config   : {args.config}"
        )

        print(
            f"Strategy : {config.strategy}"
        )

        print(
            f"Rounds   : {config.num_rounds}"
        )

        print(
            f"Epochs   : {config.local_epochs}"
        )

        print(
            f"LR       : {config.learning_rate}"
        )

        print(
            f"Batch    : {config.batch_size}"
        )

        print(
            f"Dataset  : {config.dataset.name}"
        )

        print(
            f"Partition: {config.dataset.partition}"
        )

        print(
            f"Seed     : {config.dataset.seed}"
        )

        if config.strategy == "fedprox":
            print(
                f"Mu       : {config.proximal_mu}"
            )

        print(
            f"DP       : {config.dp.enabled}"
        )

        print(
            f"DP norm  : {config.dp.max_grad_norm}"
        )

        print(
            f"DP noise : {config.dp.noise_multiplier}"
        )

        print(
            f"Epsilon  : {config.dp.epsilon}"
        )

        print(
            f"Delta    : {config.dp.delta}"
        )

        print("=" * 60)

        result = run_from_config(
            config
        )

        output_path = save_result(
            result,
            args.config,
        )

        print()
        print(
            "[SUCCESS] Experiment completed."
        )

        print(
            f"Strategy : {config.strategy}"
        )

        print(
            f"DP       : {config.dp.enabled}"
        )

        print(
            f"Result   : {output_path}"
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
        KeyError,
        ImportError,
    ) as exc:

        print(
            f"[ERROR] {exc}",
            file=sys.stderr,
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
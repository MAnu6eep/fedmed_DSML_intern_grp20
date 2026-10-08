"""Common experiment runner.

Launch a configured centralized or federated experiment from YAML.

Example:
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

from fedmed.data.partitioner import partition_dirichlet, partition_iid

from experiments.scaffold_convergence_comparison import (
    NUM_CLIENTS,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Supported configuration fields and strategies
# ---------------------------------------------------------------------------

SUPPORTED_CONFIG_KEYS = {
    "strategy",
    "rounds",
    "num_rounds",
    "local_epochs",
    "learning_rate",
    "batch_size",
    "partition",
    "dataset",
    "mu",
    "proximal_mu",
    "dp",
}

SUPPORTED_STRATEGIES = [
    "centralized",
    "fedavg",
    "fedprox",
    "scaffold",
]

SUPPORTED_DATASET_KEYS = {
    "name",
    "partition",
    "dirichlet_alpha",
    "seed",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

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
# YAML loading & normalization
# ---------------------------------------------------------------------------

def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load, normalize, and validate the YAML configuration."""

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

    # 1. Strategy validation
    strategy_raw = data.get("strategy")
    if not strategy_raw:
        raise ValueError("Configuration missing required field: 'strategy'.")

    strategy = str(strategy_raw).strip().lower()
    if strategy not in SUPPORTED_STRATEGIES:
        supported_list = "\n".join(f"- {s}" for s in SUPPORTED_STRATEGIES)
        raise ValueError(
            f"Unsupported strategy: {strategy_raw}\n"
            f"Supported strategies:\n{supported_list}"
        )
    data["strategy"] = strategy

    # 2. Check field name aliases
    if "rounds" in data and "num_rounds" not in data:
        data["num_rounds"] = data.pop("rounds")

    if "mu" in data and "proximal_mu" not in data:
        data["proximal_mu"] = data.pop("mu")

    # Validate FedProx mu parameter
    if strategy == "fedprox":
        mu = data.get("proximal_mu", 0.01)
        if mu is None or mu < 0:
            raise ValueError("FedProx requires a non-negative 'mu' parameter (e.g., mu: 0.01).")
        data["proximal_mu"] = mu

    # Normalize partition mapping to dataset config structure
    if "partition" in data and "dataset" not in data:
        partition_val = data.pop("partition")
        if isinstance(partition_val, dict):
            p_type = partition_val.get("type", "non_iid")
            p_alpha = partition_val.get("alpha", 0.5)
        else:
            p_type = str(partition_val)
            p_alpha = 0.5
        data["dataset"] = {
            "name": "brats",
            "partition": p_type,
            "dirichlet_alpha": p_alpha,
            "seed": 42,
        }

    unknown_keys = set(data) - SUPPORTED_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(
            "Unsupported configuration field(s): "
            + ", ".join(sorted(unknown_keys))
        )

    return data


# ---------------------------------------------------------------------------
# Experiment configuration builder
# ---------------------------------------------------------------------------

def build_experiment_config(
    raw_config: Dict[str, Any],
) -> ExperimentConfig:
    """Convert YAML data into a validated ExperimentConfig."""

    dataset_data = raw_config.get(
        "dataset",
        {},
    )

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

    experiment_fields = {
        key: value
        for key, value in raw_config.items()
        if key not in {"dataset", "dp"}
    }

    return ExperimentConfig(
        dataset=dataset,
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

    # Keep CPU execution deterministic for the current experiment runner.
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Dataset partitioning
# ---------------------------------------------------------------------------

def create_partitions(
    config: ExperimentConfig,
) -> Dict[str, list]:
    """Create the hospital partition (IID or Non-IID Dirichlet)."""

    volumes = [
        f"volume_{index:02d}"
        for index in range(12)
    ]

    labels = [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
    ]

    p_type = config.dataset.partition.lower()

    if p_type in {"iid", "uniform"}:
        return partition_iid(
            volumes,
            num_clients=NUM_CLIENTS,
            seed=config.dataset.seed,
        )
    elif p_type in {"non_iid", "dirichlet"}:
        return partition_dirichlet(
            volumes,
            labels,
            num_clients=NUM_CLIENTS,
            alpha=config.dataset.dirichlet_alpha,
            seed=config.dataset.seed,
        )
    else:
        raise ValueError(
            f"Unsupported partition type: '{config.dataset.partition}'. "
            "Supported partition types: 'iid', 'non_iid'."
        )


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

def run_from_config(
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """Initialize and run the configured experiment."""

    set_reproducibility(config.dataset.seed)
    partitions = create_partitions(config)

    if config.strategy == "centralized":
        from fedmed.core.model import get_model
        from fedmed.core.training import run_local_training
        from fedmed.metrics.benchmark_framework import SystemMonitor, record_and_export_benchmark
        from torch.utils.data import DataLoader, Dataset

        class SyntheticCentralizedDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                image = torch.full((4, 32, 32, 16), 0.5, dtype=torch.float32)
                label = torch.zeros((1, 32, 32, 16), dtype=torch.float32)
                label[:, 4:12, 4:12, 2:8] = 1.0
                return {"image": image, "label": label}

        model = get_model(in_channels=4, out_channels=1)
        loader = DataLoader(SyntheticCentralizedDataset(), batch_size=config.batch_size)

        with SystemMonitor() as monitor:
            metrics = run_local_training(
                model=model,
                train_loader=loader,
                val_loader=loader,
                epochs=config.local_epochs,
                learning_rate=config.learning_rate,
                device=torch.device("cpu"),
            )

        combined_metrics = {
            "dice": metrics.get("val_dice", 0.0),
            "hd95": metrics.get("val_hd95", 20.0),
            "execution_time_seconds": monitor.elapsed_time,
            "vram_peak_mb": monitor.vram_peak_mb,
            "vram_current_mb": monitor.vram_current_mb,
            "cuda_available": monitor.cuda_available,
        }

        record_and_export_benchmark(
            strategy_name="Centralized",
            metrics=combined_metrics,
            round_or_epoch=config.local_epochs,
        )

        history_entry = {
            "round": 1,
            "train_loss": metrics["train_loss"],
            "global_loss": metrics["val_loss"],
            "val_loss": metrics["val_loss"],
            "val_dice": metrics["val_dice"],
            "execution_time_seconds": monitor.elapsed_time,
            "vram_peak_mb": monitor.vram_peak_mb,
            "failures": 0,
        }

        result = {
            "distribution": config.dataset.partition,
            "strategy": "Centralized",
            "history": [history_entry],
        }
    else:
        result = run_experiment(config, partitions)

    if result is None:
        result = {}

    if not isinstance(result, dict):
        result = {"result": result}

    # Store exact configuration used.
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
            "dirichlet_alpha": config.dataset.dirichlet_alpha,
            "seed": config.dataset.seed,
        },
    }

    result["partitions"] = partitions
    return result


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_result(
    result: Dict[str, Any],
    config_path: Path,
) -> Path:
    """Save the experiment result as JSON."""

    output_dir = PROJECT_ROOT / "experiments" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{config_path.stem}_result.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, default=str)

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """CLI entry point."""

    args = parse_args()

    try:
        raw_config = load_yaml_config(args.config)
        config = build_experiment_config(raw_config)

        print("=" * 60)
        print("FedMed Common Experiment Runner")
        print("=" * 60)
        print(f"Config   : {args.config}")
        print(f"Strategy : {config.strategy}")
        print(f"Rounds   : {config.num_rounds}")
        print(f"Epochs   : {config.local_epochs}")
        print(f"LR       : {config.learning_rate}")
        print(f"Batch    : {config.batch_size}")
        print(f"Dataset  : {config.dataset.name}")
        print(f"Partition: {config.dataset.partition}")
        print(f"Seed     : {config.dataset.seed}")

        if config.strategy == "fedprox":
            print(f"Mu       : {config.proximal_mu}")

        print("=" * 60)

        result = run_from_config(config)
        output_path = save_result(result, args.config)

        print()
        print("[SUCCESS] Experiment completed.")
        print(f"Strategy : {config.strategy}")
        print(f"Result   : {output_path}")

        return 0

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
        KeyError,
        ImportError,
    ) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

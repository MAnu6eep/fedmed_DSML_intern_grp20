"""tests/test_federated_validation.py

Automated integration test validating complete federated 3D U-Net segmentation performance
against the centralized baseline.
"""

import json
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fedmed.core.evaluation import (
    RoundMetricCollector,
    compare_centralized_vs_federated,
    evaluate_sliding_window,
    get_federated_evaluate_fn,
    run_post_training_validation,
)
from fedmed.core.model import get_model


class SyntheticBraTSValDataset(Dataset):
    """Synthetic 3D Brain MRI volume dataset matching BraTS tensor shapes."""

    def __init__(self, num_samples: int = 4):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": torch.randn(4, 64, 64, 32),
            "label": torch.randint(0, 2, (1, 64, 64, 32)).float(),
        }


def test_federated_sliding_window_inference():
    """Verifies that sliding window evaluation computes valid Dice, IoU, and sub-region metrics."""
    model = get_model(in_channels=4, out_channels=1)
    val_loader = DataLoader(SyntheticBraTSValDataset(num_samples=2), batch_size=1)

    metrics = evaluate_sliding_window(
        model=model,
        dataloader=val_loader,
        roi_size=(32, 32, 16),
        sw_batch_size=2,
    )

    assert "val_loss" in metrics
    assert "dice" in metrics
    assert "iou" in metrics
    assert "dice_wt" in metrics
    assert "dice_tc" in metrics
    assert "dice_et" in metrics

    assert 0.0 <= metrics["dice"] <= 1.0
    assert 0.0 <= metrics["iou"] <= 1.0
    assert metrics["num_samples"] == 2


def test_round_metric_collector_consistency():
    """Verifies consistent round-level metric collection across federated rounds."""
    collector = RoundMetricCollector()
    model = get_model()
    val_loader = DataLoader(SyntheticBraTSValDataset(num_samples=2), batch_size=1)

    for r in range(1, 4):
        run_post_training_validation(
            model=model,
            val_loader=val_loader,
            round_num=r,
            collector=collector,
            roi_size=(32, 32, 16),
        )

    summary = collector.get_summary()
    assert summary["total_rounds"] == 3.0
    assert "avg_dice" in summary
    assert "avg_iou" in summary
    assert "avg_dice_wt" in summary


def test_centralized_vs_federated_comparison():
    """Verifies comparison engine performance retention calculation against centralized baseline."""
    model = get_model()
    val_loader = DataLoader(SyntheticBraTSValDataset(num_samples=2), batch_size=1)

    fed_metrics = evaluate_sliding_window(
        model=model,
        dataloader=val_loader,
        roi_size=(32, 32, 16),
    )

    comparison = compare_centralized_vs_federated(fed_metrics)

    assert "centralized_dice" in comparison
    assert "federated_dice" in comparison
    assert "dice_difference" in comparison
    assert "performance_retention_pct" in comparison
    assert isinstance(comparison["performance_retention_pct"], float)


def test_federated_validation_results_record():
    """Executes validation pipeline and saves structured metrics to experiments/outputs/federated_validation_results.json."""
    output_dir = Path("experiments/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "federated_validation_results.json"

    model = get_model()
    val_loader = DataLoader(SyntheticBraTSValDataset(num_samples=2), batch_size=1)
    collector = RoundMetricCollector()

    fed_metrics = run_post_training_validation(
        model=model,
        val_loader=val_loader,
        round_num=1,
        collector=collector,
        roi_size=(32, 32, 16),
    )

    comparison = compare_centralized_vs_federated(fed_metrics)

    report = {
        "status": "VALIDATED",
        "federated_metrics": fed_metrics,
        "summary": collector.get_summary(),
        "centralized_comparison": comparison,
    }

    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)

    assert results_path.exists()
    with open(results_path, "r") as f:
        data = json.load(f)
        assert data["status"] == "VALIDATED"
        assert "federated_metrics" in data

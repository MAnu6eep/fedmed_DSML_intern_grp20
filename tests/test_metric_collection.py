"""
tests.test_metric_collection
=============================
Unit tests verifying automated benchmark metric collection and resource monitoring integration.
"""

import os
import json
import pytest
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from fedmed.core.evaluation import (
    evaluate_sliding_window,
    RoundMetricCollector,
    get_federated_evaluate_fn,
)
from fedmed.core.training import run_local_training
from fedmed.core.model import get_model
from fedmed.metrics import (
    BenchmarkResult,
    BenchmarkSuite,
    create_benchmark_result,
    record_and_export_benchmark,
    SUPPORTED_STRATEGIES,
)


class MockDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        # 4 MRI channels, 3D volume matching UNet requirements
        img = torch.randn(4, 64, 64, 32)
        label = torch.zeros(1, 64, 64, 32)
        label[:, 8:24, 8:24, 4:12] = 1.0
        return {"image": img, "label": label}


def test_evaluate_sliding_window_metric_collection():
    """Verify evaluate_sliding_window computes Dice, IoU, HD95, execution time, and VRAM metrics."""
    model = get_model(in_channels=4, out_channels=1)
    loader = DataLoader(MockDataset(), batch_size=1)

    results = evaluate_sliding_window(
        model=model,
        dataloader=loader,
        roi_size=(64, 64, 32),
        device=torch.device("cpu"),
    )

    assert "dice" in results
    assert "iou" in results
    assert "hd95" in results
    assert "hausdorff_distance_95" in results
    assert "execution_time_seconds" in results
    assert "vram_peak_mb" in results
    assert "cuda_available" in results
    assert results["execution_time_seconds"] >= 0.0


def test_round_metric_collector_hd95_and_resources():
    """Verify RoundMetricCollector records HD95 and system resource metrics."""
    collector = RoundMetricCollector()
    entry = collector.record_round(
        round_num=1,
        val_loss=0.25,
        dice_score=0.88,
        iou_score=0.78,
        hd95_score=2.5,
        num_samples=4,
        execution_time_seconds=1.45,
        vram_peak_mb=512.0,
        region_metrics={"dice_wt": 0.90, "hd95_wt": 2.1}
    )

    assert entry["round"] == 1.0
    assert entry["hd95"] == 2.5
    assert entry["execution_time_seconds"] == 1.45
    assert entry["vram_peak_mb"] == 512.0

    summary = collector.get_summary()
    assert summary["avg_dice"] == 0.88
    assert summary["avg_hd95"] == 2.5
    assert summary["avg_dice_wt"] == 0.90


def test_run_local_training_metric_collection():
    """Verify run_local_training records timing and resource usage metrics."""
    model = get_model(in_channels=4, out_channels=1)
    train_loader = DataLoader(MockDataset(), batch_size=1)
    val_loader = DataLoader(MockDataset(), batch_size=1)

    metrics = run_local_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        device=torch.device("cpu"),
    )

    assert "train_loss" in metrics
    assert "val_loss" in metrics
    assert "val_dice" in metrics
    assert "val_hd95" in metrics
    assert "execution_time_seconds" in metrics
    assert "vram_peak_mb" in metrics
    assert "cuda_available" in metrics


def test_record_and_export_benchmark_all_strategies(tmp_path):
    """Verify record_and_export_benchmark records all 4 strategies and exports valid JSON."""
    suite = BenchmarkSuite("Test_Metric_Collection_Suite")
    output_file = tmp_path / "benchmark_results.json"

    for strategy in SUPPORTED_STRATEGIES:
        metrics = {
            "dice": 0.85,
            "hd95": 3.0,
            "iou": 0.75,
            "execution_time_seconds": 5.2,
            "vram_peak_mb": 1024.0,
            "cuda_available": False,
        }
        res = record_and_export_benchmark(
            strategy_name=strategy,
            metrics=metrics,
            round_or_epoch=1,
            suite=suite,
            output_path=output_file,
        )
        assert res.strategy_name == strategy
        assert res.dice_score == 0.85
        assert res.hausdorff_distance_95 == 3.0

    assert os.path.exists(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_records"] == 4
    strats_recorded = [r["strategy_name"] for r in data["detailed_results"]]
    for s in SUPPORTED_STRATEGIES:
        assert s in strats_recorded


def test_metric_collection_non_interference():
    """Verify metric collection handles unusual empty tensors without breaking execution."""
    empty_metrics = {
        "dice": "invalid_val",  # should fall back to 0.0 without exception
        "hd95": float('inf'),
    }
    res = create_benchmark_result("Centralized", empty_metrics)
    assert res.strategy_name == "Centralized"
    assert res.dice_score == 0.0
    assert res.hausdorff_distance_95 == float('inf')

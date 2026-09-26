"""
tests.test_benchmark_framework
===============================
Unit tests for fedmed.metrics.benchmark_framework module.
"""

import os
import json
import pytest
import torch
import numpy as np
from pathlib import Path

from fedmed.metrics import (
    SystemMonitor,
    compute_hausdorff_distance_95,
    BenchmarkResult,
    BenchmarkSuite,
    SUPPORTED_STRATEGIES,
)


def test_system_monitor_context_manager():
    """Verify SystemMonitor accurately tracks elapsed time and VRAM usage."""
    with SystemMonitor() as monitor:
        # Simulate work
        x = torch.zeros((100, 100))
        del x

    assert monitor.elapsed_time >= 0.0
    assert isinstance(monitor.vram_peak_mb, float)
    assert isinstance(monitor.vram_current_mb, float)
    assert isinstance(monitor.cuda_available, bool)


def test_system_monitor_manual_start_stop():
    """Verify manual start/stop workflow of SystemMonitor."""
    monitor = SystemMonitor()
    monitor.start()
    y = np.ones((50, 50))
    metrics = monitor.stop()

    assert "elapsed_time_seconds" in metrics
    assert "vram_peak_mb" in metrics
    assert "vram_current_mb" in metrics
    assert "cuda_available" in metrics
    assert metrics["elapsed_time_seconds"] >= 0.0


def test_compute_hausdorff_distance_95_perfect_match():
    """Verify HD95 returns zero (or minimal distance) for identical predictions and ground truth."""
    pred = torch.zeros((1, 1, 16, 16, 16))
    label = torch.zeros((1, 1, 16, 16, 16))
    pred[0, 0, 4:8, 4:8, 4:8] = 1.0
    label[0, 0, 4:8, 4:8, 4:8] = 1.0

    hd95 = compute_hausdorff_distance_95(pred, label)
    assert hd95 == 0.0


def test_compute_hausdorff_distance_95_empty_mask_handling():
    """Verify HD95 handles empty masks gracefully without crashing."""
    pred = torch.zeros((1, 1, 16, 16, 16))
    label = torch.zeros((1, 1, 16, 16, 16))

    # Both empty -> 0.0
    assert compute_hausdorff_distance_95(pred, label) == 0.0

    # One empty -> inf
    pred[0, 0, 2:4, 2:4, 2:4] = 1.0
    assert compute_hausdorff_distance_95(pred, label) == float('inf')


def test_compute_hausdorff_distance_95_numpy_input():
    """Verify numpy arrays are properly accepted by HD95 function."""
    pred = np.zeros((16, 16, 16), dtype=np.float32)
    label = np.zeros((16, 16, 16), dtype=np.float32)
    pred[2:6, 2:6, 2:6] = 1.0
    label[3:7, 3:7, 3:7] = 1.0

    hd95 = compute_hausdorff_distance_95(pred, label)
    assert isinstance(hd95, float)
    assert hd95 >= 0.0


def test_benchmark_result_creation_and_dict():
    """Verify BenchmarkResult captures required attributes for all strategies."""
    for strategy in SUPPORTED_STRATEGIES:
        res = BenchmarkResult(
            strategy_name=strategy,
            round_or_epoch=5,
            execution_time_seconds=12.5,
            dice_score=0.87,
            hausdorff_distance_95=3.2,
            iou_score=0.77,
            vram_peak_mb=1024.0,
            vram_current_mb=512.0,
            region_metrics={"dice_wt": 0.89, "dice_tc": 0.85, "dice_et": 0.81}
        )

        d = res.to_dict()
        assert d["strategy_name"] == strategy
        assert d["round_or_epoch"] == 5
        assert d["dice_score"] == 0.87
        assert d["hausdorff_distance_95"] == 3.2
        assert d["region_metrics"]["dice_wt"] == 0.89


def test_benchmark_suite_aggregation_and_export(tmp_path):
    """Verify BenchmarkSuite aggregates results, builds comparisons, and exports JSON."""
    suite = BenchmarkSuite(experiment_name="Unit_Test_Experiment")

    # Add results for Centralized, FedAvg, FedProx, SCAFFOLD
    strategies_data = {
        "Centralized": (0.91, 2.1, 15.0),
        "FedAvg": (0.88, 3.5, 20.0),
        "FedProx": (0.89, 3.1, 22.0),
        "SCAFFOLD": (0.90, 2.8, 25.0)
    }

    for strat_name, (dice, hd95, time_sec) in strategies_data.items():
        suite.record_result(BenchmarkResult(
            strategy_name=strat_name,
            dice_score=dice,
            hausdorff_distance_95=hd95,
            execution_time_seconds=time_sec,
            vram_peak_mb=2048.0
        ))

    # Check strategy summary
    fedavg_summary = suite.get_strategy_summary("FedAvg")
    assert fedavg_summary["strategy_name"] == "FedAvg"
    assert fedavg_summary["count"] == 1
    assert fedavg_summary["avg_dice_score"] == 0.88
    assert fedavg_summary["avg_hd95"] == 3.5

    # Check comparison table
    comparison = suite.compare_all_strategies()
    assert len(comparison) >= 4
    strat_names_in_comp = [c["strategy_name"] for c in comparison]
    for s in SUPPORTED_STRATEGIES:
        assert s in strat_names_in_comp

    # Check JSON export
    output_file = tmp_path / "test_benchmark_results.json"
    exported_path = suite.export_json(output_file)
    assert os.path.exists(exported_path)

    with open(exported_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["experiment_name"] == "Unit_Test_Experiment"
    assert data["total_records"] == 4
    assert len(data["detailed_results"]) == 4
    assert len(data["comparison_summary"]) >= 4

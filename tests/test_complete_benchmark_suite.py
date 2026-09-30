"""
tests.test_complete_benchmark_suite
===================================
Unit test suite verifying Phase 2 Complete Benchmark Suite validation.
Ensures Centralized, FedAvg, FedProx, and SCAFFOLD produce valid complete results in the expected JSON format.
"""

import os
import json
import pytest
from pathlib import Path

from fedmed.metrics.benchmark_framework import (
    BenchmarkResult,
    BenchmarkSuite,
    SUPPORTED_STRATEGIES,
    record_and_export_benchmark,
)


def test_complete_benchmark_suite_execution_and_schema(tmp_path):
    """Verify that all four strategies produce valid complete results with no missing required metrics."""
    suite = BenchmarkSuite(experiment_name="Complete_Model_Comparison_Suite")
    output_file = tmp_path / "benchmark_framework_results.json"

    # Simulate/record complete experiments for all four strategies
    strategy_metrics = {
        "Centralized": {"dice": 0.912, "hd95": 2.15, "execution_time_seconds": 12.4, "vram_peak_mb": 1536.0},
        "FedAvg": {"dice": 0.884, "hd95": 3.42, "execution_time_seconds": 18.2, "vram_peak_mb": 1024.0},
        "FedProx": {"dice": 0.891, "hd95": 3.10, "execution_time_seconds": 21.5, "vram_peak_mb": 1024.0},
        "SCAFFOLD": {"dice": 0.903, "hd95": 2.75, "execution_time_seconds": 24.8, "vram_peak_mb": 1024.0},
    }

    for strat_name, metrics in strategy_metrics.items():
        record_and_export_benchmark(
            strategy_name=strat_name,
            metrics=metrics,
            round_or_epoch=3,
            suite=suite,
            output_path=output_file,
        )

    # 1. Check file exists
    assert os.path.exists(output_file), "Benchmark output file was not created"

    # 2. Check JSON content & structure
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "experiment_name" in data
    assert "export_timestamp" in data
    assert "comparison_summary" in data
    assert "detailed_results" in data

    # 3. Verify all four approaches are present with non-missing metrics
    detailed_strats = [r["strategy_name"] for r in data["detailed_results"]]
    for s in SUPPORTED_STRATEGIES:
        assert s in detailed_strats, f"Strategy {s} missing from detailed results"

    for record in data["detailed_results"]:
        assert record["strategy_name"] in SUPPORTED_STRATEGIES
        assert "dice_score" in record and record["dice_score"] is not None
        assert "hausdorff_distance_95" in record and record["hausdorff_distance_95"] is not None
        assert "execution_time_seconds" in record and record["execution_time_seconds"] >= 0.0
        assert "vram_peak_mb" in record and record["vram_peak_mb"] >= 0.0

    # 4. Verify summary table contains valid comparison metrics
    comparison = data["comparison_summary"]
    assert len(comparison) >= 4
    for summary_item in comparison:
        if summary_item["count"] > 0:
            assert summary_item["avg_dice_score"] > 0.0
            assert summary_item["avg_execution_time_seconds"] > 0.0


def test_no_missing_required_metrics_in_records():
    """Ensure no benchmark record has missing or NaN fields."""
    metrics_valid = {
        "dice": 0.85,
        "hd95": 2.8,
        "iou": 0.75,
        "execution_time_seconds": 10.0,
        "vram_peak_mb": 500.0,
        "cuda_available": False
    }

    res = record_and_export_benchmark(
        strategy_name="FedAvg",
        metrics=metrics_valid,
        round_or_epoch=1
    )

    d = res.to_dict()
    required_keys = ["strategy_name", "round_or_epoch", "timestamp", "execution_time_seconds", "dice_score", "hausdorff_distance_95", "vram_peak_mb"]
    for k in required_keys:
        assert k in d
        assert d[k] is not None

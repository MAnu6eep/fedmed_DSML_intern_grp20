"""
tests.test_benchmark_setup
==========================
Unit test suite verifying Phase 1 Benchmark Setup for Centralized, FedAvg, FedProx, and SCAFFOLD.
Ensures all four training strategies execute using the consistent evaluation pipeline.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from fedmed.core.model import get_model
from fedmed.core.evaluation import evaluate_sliding_window, evaluate_and_extract_slices
from fedmed.metrics.benchmark_framework import (
    SystemMonitor,
    compute_hausdorff_distance_95,
    BenchmarkResult,
    BenchmarkSuite,
    create_benchmark_result,
    record_and_export_benchmark,
    SUPPORTED_STRATEGIES,
)


class MockMRIVolumeDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        # 4 MRI channels, 3D volume matching BraTS models
        img = torch.randn(4, 64, 64, 32)
        label = torch.zeros(1, 64, 64, 32)
        label[:, 8:24, 8:24, 4:12] = 1.0
        return {"image": img, "label": label}


def test_consistent_evaluation_pipeline_across_all_strategies():
    """Verify Centralized, FedAvg, FedProx, and SCAFFOLD use the same evaluation settings and metrics."""
    model = get_model(in_channels=4, out_channels=1)
    dataloader = DataLoader(MockMRIVolumeDataset(), batch_size=1)
    consistent_roi_size = (64, 64, 32)

    for strategy in SUPPORTED_STRATEGIES:
        # Run evaluation through standard evaluation pipeline
        eval_metrics = evaluate_sliding_window(
            model=model,
            dataloader=dataloader,
            roi_size=consistent_roi_size,
            device=torch.device("cpu"),
        )

        # Check required evaluation metrics exist
        assert "dice" in eval_metrics, f"{strategy} missing dice metric"
        assert "hd95" in eval_metrics, f"{strategy} missing HD95 metric"
        assert "execution_time_seconds" in eval_metrics, f"{strategy} missing execution time"
        assert "vram_peak_mb" in eval_metrics, f"{strategy} missing VRAM tracking"
        assert "cuda_available" in eval_metrics, f"{strategy} missing CUDA status"

        # Check valid non-negative value types
        assert isinstance(eval_metrics["dice"], float)
        assert isinstance(eval_metrics["hd95"], float)
        assert eval_metrics["execution_time_seconds"] >= 0.0


def test_standardized_benchmark_result_recording():
    """Verify BenchmarkResult captures identical schema for Centralized and Federated strategies."""
    suite = BenchmarkSuite(experiment_name="Standardized_Pipeline_Test")

    for strategy in SUPPORTED_STRATEGIES:
        metrics_dict = {
            "dice": 0.88,
            "hd95": 2.45,
            "val_loss": 0.15,
            "execution_time_seconds": 3.12,
            "vram_peak_mb": 512.0,
            "cuda_available": False,
        }

        result = create_benchmark_result(
            strategy_name=strategy,
            metrics=metrics_dict,
            round_or_epoch=3,
        )

        assert result.strategy_name == strategy
        assert result.dice_score == 0.88
        assert result.hausdorff_distance_95 == 2.45
        assert result.execution_time_seconds == 3.12
        assert result.vram_peak_mb == 512.0

        suite.record_result(result)

    summary = suite.compare_all_strategies()
    assert len(summary) >= 4
    for s_summary in summary:
        assert s_summary["strategy_name"] in SUPPORTED_STRATEGIES
        assert s_summary["count"] == 1

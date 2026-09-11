"""tests/test_week2_ml_validation.py

PyTest automated integration suite verifying Week 2 ML evaluation metrics consolidation
across Centralized Baseline, IID Federated, and Non-IID Federated (FedProx) experiments.
"""

import json
from pathlib import Path
import pytest
from experiments.week2_ml_validation import run_week2_validation_pipeline


def test_week2_validation_pipeline_execution():
    """Executes the full Week 2 validation pipeline and verifies report schema and metric bounds."""
    report = run_week2_validation_pipeline()

    assert report["status"] == "COMPLETED"
    assert "models" in report
    assert "comparisons" in report
    assert "summary" in report

    # Verify model evaluation entries contain overall and sub-region metrics
    models = report["models"]
    for setup in ["centralized_baseline", "iid_federated", "non_iid_federated_fedprox"]:
        assert setup in models
        metrics = models[setup]
        assert "val_loss" in metrics
        assert "dice" in metrics
        assert "iou" in metrics
        assert "dice_wt" in metrics
        assert "dice_tc" in metrics
        assert "dice_et" in metrics
        assert 0.0 <= metrics["dice"] <= 1.0
        assert 0.0 <= metrics["iou"] <= 1.0


def test_week2_validation_report_file_generation():
    """Verifies that experiments/outputs/week2_ml_validation_report.json is written to disk."""
    report_path = Path("experiments/outputs/week2_ml_validation_report.json")
    assert report_path.exists()

    with open(report_path, "r") as f:
        data = json.load(f)
        assert data["status"] == "COMPLETED"
        assert "summary" in data
        assert "iid_retention_pct" in data["summary"]
        assert "non_iid_retention_pct" in data["summary"]

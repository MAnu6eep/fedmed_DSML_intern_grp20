"""tests/test_integrated_ml_pipeline.py

Automated integration test suite verifying the complete Week 3 integrated ML workflow:
from MRI preprocessing to 3D U-Net inference, 2D slice extraction, and dashboard visualization.
"""

import json
from pathlib import Path
import pytest

from experiments.verify_integrated_ml_pipeline import (
    run_integrated_ml_pipeline_verification,
)


def test_run_integrated_ml_pipeline_verification_execution():
    """Verifies end-to-end execution of the integrated ML pipeline."""
    report = run_integrated_ml_pipeline_verification()

    assert report["status"] == "PASSED"
    assert "pipeline_stages" in report

    stages = report["pipeline_stages"]
    assert stages["preprocessing"]["input_channels"] == 4
    assert "inference" in stages
    assert stages["visualization_slicing"]["mri_mask_shape_alignment"] is True
    assert stages["visualization_slicing"]["base64_preview_generated"] is True
    assert stages["federated_integration"]["fl_round_artifact_exists"] is True


def test_integrated_ml_pipeline_report_file_generation():
    """Verifies that experiments/outputs/integrated_ml_pipeline_report.json is written to disk."""
    report_path = Path("experiments/outputs/integrated_ml_pipeline_report.json")
    assert report_path.exists()

    with open(report_path, "r") as f:
        data = json.load(f)
        assert data["status"] == "PASSED"
        assert "pipeline_stages" in data

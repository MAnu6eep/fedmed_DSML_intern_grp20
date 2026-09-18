"""tests/test_preprocessing_consistency.py

Automated test suite verifying experiment-level preprocessing configuration and
MRI intensity normalization and spatial resampling consistency across federated hospital nodes.
"""

import json
from pathlib import Path
import pytest

from experiments.verify_preprocessing_consistency import (
    PreprocessingConfig,
    verify_hospital_preprocessing_consistency,
)


def test_preprocessing_config_default_initialization():
    """Verifies standard experiment-level preprocessing configuration defaults."""
    config = PreprocessingConfig()
    assert config.roi_size == (64, 64, 32)
    assert config.pixdim == (1.0, 1.0, 1.0)
    assert config.axcodes == "RAS"
    assert config.nonzero_norm is True
    assert config.channel_wise_norm is True


def test_verify_hospital_preprocessing_consistency_execution():
    """Verifies that MRI data from all 3 hospital nodes is preprocessed uniformly."""
    report = verify_hospital_preprocessing_consistency(num_hospitals=3, samples_per_hospital=2)

    assert report["status"] == "PASSED"
    assert report["hospital_count"] == 3
    assert "consistency_checks" in report

    checks = report["consistency_checks"]
    assert checks["spatial_shape_uniformity"] is True
    assert checks["intensity_mean_zero_centered"] is True
    assert checks["overall_consistent"] is True

    # Verify hospital node metrics
    nodes = report["hospital_nodes"]
    for h_idx in range(1, 4):
        h_id = f"hospital_{h_idx}"
        assert h_id in nodes
        metrics = nodes[h_id]
        assert metrics["channels"] == 4
        assert "intensity_stats_nonzero" in metrics


def test_preprocessing_consistency_report_file_generation():
    """Verifies that experiments/outputs/preprocessing_consistency_report.json is written to disk."""
    report_path = Path("experiments/outputs/preprocessing_consistency_report.json")
    assert report_path.exists()

    with open(report_path, "r") as f:
        data = json.load(f)
        assert data["status"] == "PASSED"
        assert "hospital_nodes" in data

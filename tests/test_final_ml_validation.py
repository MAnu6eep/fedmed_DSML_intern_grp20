"""
tests.test_final_ml_validation
===============================
Unit test suite verifying Phase 3 Final ML Validation & Comparison Table Consolidation.
Ensures Centralized, FedAvg, FedProx, and SCAFFOLD metrics are correctly aggregated into
the final comparison table and report.
"""

import json
import pytest
from pathlib import Path

from scripts.finalize_benchmark_report import (
    build_final_comparison_table,
    generate_formatted_markdown_report,
    load_benchmark_data,
    main as run_finalize_main,
)
from fedmed.metrics.benchmark_framework import SUPPORTED_STRATEGIES


def test_final_comparison_table_structure_and_completeness():
    """Verify that all four training strategies are included in the consolidated comparison table with all required fields."""
    mock_data = {
        "export_timestamp": "2026-09-30T12:00:00Z",
        "comparison_summary": [
            {
                "strategy_name": "Centralized",
                "count": 1,
                "avg_dice_score": 0.9150,
                "avg_hd95": 2.10,
                "avg_execution_time_seconds": 12.5,
                "max_vram_peak_mb": 1536.0,
            },
            {
                "strategy_name": "FedAvg",
                "count": 1,
                "avg_dice_score": 0.8850,
                "avg_hd95": 3.40,
                "avg_execution_time_seconds": 18.0,
                "max_vram_peak_mb": 1024.0,
            },
            {
                "strategy_name": "FedProx",
                "count": 1,
                "avg_dice_score": 0.8920,
                "avg_hd95": 3.15,
                "avg_execution_time_seconds": 21.0,
                "max_vram_peak_mb": 1024.0,
            },
            {
                "strategy_name": "SCAFFOLD",
                "count": 1,
                "avg_dice_score": 0.9050,
                "avg_hd95": 2.70,
                "avg_execution_time_seconds": 24.0,
                "max_vram_peak_mb": 1024.0,
            },
        ],
    }

    table = build_final_comparison_table(mock_data)

    # Check length and strategy names
    assert len(table) == 4
    strategies_in_table = [row["Experiment"] for row in table]
    for strat in SUPPORTED_STRATEGIES:
        assert strat in strategies_in_table

    # Check fields in each row
    required_cols = ["Experiment", "Dice", "Hausdorff", "Time", "VRAM", "Status"]
    for row in table:
        for col in required_cols:
            assert col in row
        assert row["Status"] == "COMPLETED"
        assert row["Dice"] > 0.0
        assert row["Time"] > 0.0


def test_markdown_report_formatting():
    """Verify that the generated markdown report contains headers, table columns, and strategy rows."""
    sample_table = [
        {"Experiment": "Centralized", "Dice": 0.915, "Hausdorff": 2.1, "Time": 12.5, "VRAM": 1536.0, "Status": "COMPLETED"},
        {"Experiment": "FedAvg", "Dice": 0.885, "Hausdorff": 3.4, "Time": 18.0, "VRAM": 1024.0, "Status": "COMPLETED"},
        {"Experiment": "FedProx", "Dice": 0.892, "Hausdorff": 3.15, "Time": 21.0, "VRAM": 1024.0, "Status": "COMPLETED"},
        {"Experiment": "SCAFFOLD", "Dice": 0.905, "Hausdorff": 2.7, "Time": 24.0, "VRAM": 1024.0, "Status": "COMPLETED"},
    ]

    report = generate_formatted_markdown_report(sample_table)

    assert "# Final ML Benchmark Comparison Report" in report
    assert "| Experiment | Dice | Hausdorff (mm) | Time (s) | VRAM (MB) | Status |" in report
    for s in SUPPORTED_STRATEGIES:
        assert f"**{s}**" in report


def test_finalize_benchmark_report_execution():
    """Verify that running the report generator script produces valid JSON output payload."""
    exit_code = run_finalize_main()
    assert exit_code == 0

    report_path = Path(__file__).resolve().parent.parent / "experiments" / "outputs" / "final_benchmark_comparison_report.json"
    assert report_path.exists()

    with open(report_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    assert "table" in payload
    assert len(payload["table"]) == 4

"""tests/test_experiments_integration.py

Automated integration tests verifying end-to-end preprocessing, sliding-window evaluation,
and automatic 3D tumor slice visualization payload extraction.
"""

import json
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fedmed.core.evaluation import (
    RoundMetricCollector,
    evaluate_and_extract_slices,
    get_federated_evaluate_fn,
)
from fedmed.core.model import get_model
from fedmed.federation.server import get_parameters


class Synthetic3DMRIValDataset(Dataset):
    """Synthetic 3D Brain MRI volume dataset for integration testing."""

    def __init__(self, num_samples: int = 2):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": torch.randn(4, 64, 64, 32),
            "label": torch.randint(0, 2, (1, 64, 64, 32)).float(),
        }


def test_evaluate_and_extract_slices_integration():
    """Verifies that sliding window evaluation automatically produces 2D slice visualization payloads."""
    model = get_model(in_channels=4, out_channels=1)
    val_loader = DataLoader(Synthetic3DMRIValDataset(num_samples=2), batch_size=1)

    eval_res = evaluate_and_extract_slices(
        model=model,
        dataloader=val_loader,
        roi_size=(32, 32, 16),
        extract_slices=True,
        num_slices=2,
    )

    assert "val_loss" in eval_res
    assert "dice" in eval_res
    assert "iou" in eval_res
    assert "visualization_payload" in eval_res

    payload = eval_res["visualization_payload"]
    assert "slices_by_orientation" in payload
    by_orient = payload["slices_by_orientation"]
    assert "axial" in by_orient
    assert "coronal" in by_orient
    assert "sagittal" in by_orient


def test_empty_segmentation_output_resilience():
    """Verifies that empty or background-only predictions do not crash evaluation or slice extraction."""
    # Model returning zeros
    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4])

    zero_model = ZeroModel()
    val_loader = DataLoader(Synthetic3DMRIValDataset(num_samples=1), batch_size=1)

    eval_res = evaluate_and_extract_slices(
        model=zero_model,
        dataloader=val_loader,
        roi_size=(32, 32, 16),
        extract_slices=True,
    )

    assert "visualization_payload" in eval_res
    assert eval_res["visualization_payload"]["status"] in ["SUCCESS", "FALLBACK"]


def test_federated_evaluation_callback_saves_artifact():
    """Verifies that federated round evaluation callbacks automatically save round visualization artifacts."""
    output_dir = Path("experiments/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    round_artifact_file = output_dir / "federated_visualization_round_99.json"
    if round_artifact_file.exists():
        round_artifact_file.unlink()

    model = get_model()
    val_loader = DataLoader(Synthetic3DMRIValDataset(num_samples=1), batch_size=1)
    collector = RoundMetricCollector()

    eval_fn = get_federated_evaluate_fn(
        model=model,
        val_loader=val_loader,
        collector=collector,
        save_outputs=True,
    )

    params = get_parameters(model)
    loss, metrics = eval_fn(server_round=99, parameters=params, config={})

    assert loss >= 0.0
    assert "visualization_payload" in metrics
    assert round_artifact_file.exists()

    with open(round_artifact_file, "r") as f:
        saved_data = json.load(f)
        assert saved_data["round"] == 99.0
        assert "visualization_payload" in saved_data

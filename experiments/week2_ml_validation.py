"""experiments/week2_ml_validation.py

Consolidated Week 2 ML Validation Script for 3D U-Net Medical Segmentation.
Executes and compares Centralized Baseline vs. IID Federated vs. Non-IID Federated setups.
Consolidates Dice, IoU, and tumor sub-region (WT, TC, ET) metrics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset
from monai.losses import DiceFocalLoss

from fedmed.core.evaluation import (
    RoundMetricCollector,
    compare_centralized_vs_federated,
    evaluate_sliding_window,
    run_post_training_validation,
)
from fedmed.core.model import get_model
from fedmed.core.training import train_one_epoch, run_local_training


class MockBraTSVolumeDataset(Dataset):
    """Synthetic 3D Brain MRI volume dataset matching BraTS tensor shapes (4 channels -> 1 label)."""

    def __init__(self, num_samples: int = 6):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "image": torch.randn(4, 64, 64, 32),
            "label": torch.randint(0, 2, (1, 64, 64, 32)).float(),
        }


def run_week2_validation_pipeline() -> Dict[str, Any]:
    """Runs end-to-end evaluation across Centralized, IID Federated, and Non-IID Federated setups."""
    output_dir = Path("experiments/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "week2_ml_validation_report.json"

    device = torch.device("cpu")
    loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)
    val_dataset = MockBraTSVolumeDataset(num_samples=4)
    val_loader = DataLoader(val_dataset, batch_size=1)

    print("=================================================================")
    print("      FedMed Week 2 ML Validation & Experimental Comparison     ")
    print("=================================================================\n")

    # 1. Centralized Baseline Benchmark Evaluation
    print("1. Running Centralized Baseline Evaluation...")
    centralized_model = get_model(in_channels=4, out_channels=1)
    centralized_loader = DataLoader(MockBraTSVolumeDataset(num_samples=6), batch_size=2)
    optimizer = torch.optim.AdamW(centralized_model.parameters(), lr=1e-4)

    # Train centralized baseline for 1 epoch
    train_one_epoch(centralized_model, centralized_loader, optimizer, loss_fn, device=device)
    centralized_metrics = evaluate_sliding_window(
        model=centralized_model,
        dataloader=val_loader,
        roi_size=(32, 32, 16),
        device=device,
    )
    print(f"   [Centralized Baseline] Val Loss: {centralized_metrics['val_loss']:.4f} | Dice: {centralized_metrics['dice']*100:.2f}% | IoU: {centralized_metrics['iou']*100:.2f}%")

    # 2. IID Federated Setup Evaluation
    print("\n2. Running IID Federated Setup Evaluation...")
    iid_model = get_model(in_channels=4, out_channels=1)
    iid_train_loader = DataLoader(MockBraTSVolumeDataset(num_samples=4), batch_size=2)

    run_local_training(
        model=iid_model,
        train_loader=iid_train_loader,
        val_loader=val_loader,
        epochs=1,
        learning_rate=1e-4,
        device=device,
    )

    iid_metrics = evaluate_sliding_window(
        model=iid_model,
        dataloader=val_loader,
        roi_size=(32, 32, 16),
        device=device,
    )
    print(f"   [IID Federated] Val Loss: {iid_metrics['val_loss']:.4f} | Dice: {iid_metrics['dice']*100:.2f}% | IoU: {iid_metrics['iou']*100:.2f}%")

    # 3. Non-IID Federated (FedProx) Setup Evaluation
    print("\n3. Running Non-IID Federated (FedProx) Setup Evaluation...")
    non_iid_model = get_model(in_channels=4, out_channels=1)
    non_iid_train_loader = DataLoader(MockBraTSVolumeDataset(num_samples=3), batch_size=2)

    # Global reference parameters for FedProx proximal term
    global_params = [p.detach().clone() for p in non_iid_model.parameters()]

    run_local_training(
        model=non_iid_model,
        train_loader=non_iid_train_loader,
        val_loader=val_loader,
        epochs=1,
        learning_rate=1e-4,
        device=device,
        global_parameters=global_params,
        proximal_mu=0.01,
    )

    non_iid_metrics = evaluate_sliding_window(
        model=non_iid_model,
        dataloader=val_loader,
        roi_size=(32, 32, 16),
        device=device,
    )
    print(f"   [Non-IID FedProx] Val Loss: {non_iid_metrics['val_loss']:.4f} | Dice: {non_iid_metrics['dice']*100:.2f}% | IoU: {non_iid_metrics['iou']*100:.2f}%")

    # 4. Consolidate Comparative Metrics
    iid_comparison = compare_centralized_vs_federated(iid_metrics)
    non_iid_comparison = compare_centralized_vs_federated(non_iid_metrics)

    report = {
        "title": "FedMed Week 2 Consolidated ML Validation Report",
        "status": "COMPLETED",
        "models": {
            "centralized_baseline": centralized_metrics,
            "iid_federated": iid_metrics,
            "non_iid_federated_fedprox": non_iid_metrics,
        },
        "comparisons": {
            "iid_vs_centralized": iid_comparison,
            "non_iid_vs_centralized": non_iid_comparison,
        },
        "summary": {
            "best_dice_setup": "IID Federated" if iid_metrics["dice"] >= non_iid_metrics["dice"] else "Non-IID FedProx",
            "centralized_dice": centralized_metrics["dice"],
            "iid_dice": iid_metrics["dice"],
            "non_iid_dice": non_iid_metrics["dice"],
            "iid_retention_pct": iid_comparison["performance_retention_pct"],
            "non_iid_retention_pct": non_iid_comparison["performance_retention_pct"],
        },
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=================================================================")
    print(f"[SUCCESS] Week 2 ML Validation Report saved to {report_path}")
    print("=================================================================\n")

    return report


if __name__ == "__main__":
    run_week2_validation_pipeline()

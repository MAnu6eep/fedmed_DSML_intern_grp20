"""fedmed/core/evaluation.py

Local model validation engine, round-level metric collection, and 3D segmentation evaluation.
Computes Dice Similarity Coefficient and Intersection over Union (IoU) metrics.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, MeanIoU
from torch.utils.data import DataLoader


class RoundMetricCollector:
    """Structures and tracks validation metrics across federated learning rounds."""

    def __init__(self):
        self.history: List[Dict[str, float]] = []

    def record_round(
        self,
        round_num: int,
        val_loss: float,
        dice_score: float,
        iou_score: float,
        num_samples: int,
    ) -> Dict[str, float]:
        """Record metrics for a specific FL round."""
        round_entry = {
            "round": float(round_num),
            "val_loss": float(val_loss),
            "dice": float(dice_score),
            "iou": float(iou_score),
            "num_samples": float(num_samples),
        }
        self.history.append(round_entry)
        return round_entry

    def get_summary(self) -> Dict[str, float]:
        """Returns summary statistics across recorded rounds."""
        if not self.history:
            return {"avg_dice": 0.0, "avg_iou": 0.0, "avg_loss": 0.0, "best_dice": 0.0}

        dice_scores = [entry["dice"] for entry in self.history]
        iou_scores = [entry["iou"] for entry in self.history]
        loss_scores = [entry["val_loss"] for entry in self.history]

        return {
            "avg_dice": float(sum(dice_scores) / len(dice_scores)),
            "best_dice": float(max(dice_scores)),
            "avg_iou": float(sum(iou_scores) / len(iou_scores)),
            "best_iou": float(max(iou_scores)),
            "avg_loss": float(sum(loss_scores) / len(loss_scores)),
            "total_rounds": float(len(self.history)),
        }


def evaluate_model_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Runs validation evaluation over 3D MRI volumes and computes Dice and IoU metrics.
    
    Args:
        model: 3D U-Net PyTorch model
        dataloader: Validation data loader with MONAI dictionary transforms
        loss_fn: Optional loss function (defaults to DiceFocalLoss)
        device: PyTorch device (CPU or CUDA)
        
    Returns:
        Dictionary containing 'val_loss', 'dice', 'iou', and 'num_samples'.
    """
    model.eval()
    model.to(device)

    if loss_fn is None:
        loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)

    running_loss = 0.0
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")

    total_samples = 0
    total_batches = len(dataloader)

    if total_batches == 0:
        return {"val_loss": 0.0, "dice": 0.0, "iou": 0.0, "num_samples": 0}

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            total_samples += images.size(0)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            # Binarize predictions using sigmoid thresholding (> 0.5)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Update MONAI metrics
            dice_metric(y_pred=preds, y=labels)
            iou_metric(y_pred=preds, y=labels)

    avg_loss = running_loss / total_batches
    avg_dice = float(dice_metric.aggregate().item())
    avg_iou = float(iou_metric.aggregate().item())

    dice_metric.reset()
    iou_metric.reset()

    return {
        "val_loss": round(avg_loss, 6),
        "dice": round(avg_dice, 6),
        "iou": round(avg_iou, 6),
        "num_samples": total_samples,
    }


def run_post_training_validation(
    model: nn.Module,
    val_loader: DataLoader,
    round_num: int = 1,
    collector: Optional[RoundMetricCollector] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Separate post-training validation pipeline executed after local training rounds.
    
    Args:
        model: Trained 3D U-Net model
        val_loader: MONAI validation dataloader
        round_num: Current FL round index
        collector: Optional RoundMetricCollector instance
        device: Execution device
        
    Returns:
        Structured round metrics dictionary.
    """
    metrics = evaluate_model_metrics(model, val_loader, device=device)

    if collector is not None:
        collector.record_round(
            round_num=round_num,
            val_loss=metrics["val_loss"],
            dice_score=metrics["dice"],
            iou_score=metrics["iou"],
            num_samples=metrics["num_samples"],
        )

    return metrics


if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from fedmed.core.model import get_model

    # Smoke test validation pipeline
    class MockValDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, idx):
            return {
                "image": torch.randn(4, 32, 32, 16),
                "label": torch.randint(0, 2, (1, 32, 32, 16)).float()
            }

    model = get_model()
    val_loader = DataLoader(MockValDataset(), batch_size=2)
    collector = RoundMetricCollector()

    res = run_post_training_validation(model, val_loader, round_num=1, collector=collector)
    print("[OK] Post-training validation pipeline output:", res)
    print("[OK] Metric collector summary:", collector.get_summary())

"""fedmed/core/evaluation.py

Local model validation engine, sliding-window inference, round-level metric collection,
and 3D MRI segmentation evaluation (Dice & IoU).
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
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


def evaluate_sliding_window(
    model: nn.Module,
    dataloader: DataLoader,
    roi_size: Tuple[int, int, int] = (64, 64, 32),
    sw_batch_size: int = 2,
    overlap: float = 0.25,
    loss_fn: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Evaluates 3D U-Net over volumetric MRI scans using sliding-window inference.

    Args:
        model: 3D U-Net PyTorch model
        dataloader: Validation dataloader supplying MONAI dictionary batches
        roi_size: Spatial ROI patch size for sliding window (H, W, D)
        sw_batch_size: Number of sliding window patches processed per step
        overlap: Amount of overlap between consecutive sliding window patches
        loss_fn: Optional loss function (defaults to DiceFocalLoss)
        device: Execution device (CPU or CUDA)

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

            # Process large 3D volume using sliding-window inference and reconstruct complete prediction
            outputs = sliding_window_inference(
                inputs=images,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )

            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            # Binarize prediction for segmentation metric scoring (sigmoid > 0.5)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Accumulate Dice and IoU metrics
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


def evaluate_model_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Standard evaluation wrapper calling sliding-window inference."""
    return evaluate_sliding_window(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        device=device,
    )


def run_post_training_validation(
    model: nn.Module,
    val_loader: DataLoader,
    round_num: int = 1,
    collector: Optional[RoundMetricCollector] = None,
    roi_size: Tuple[int, int, int] = (64, 64, 32),
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Runs post-training sliding-window validation and records round metrics."""
    metrics = evaluate_sliding_window(
        model=model,
        dataloader=val_loader,
        roi_size=roi_size,
        device=device,
    )

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

    # Smoke test sliding-window evaluation over mock 3D MRI volume
    class Mock3DMRIBatch(torch.utils.data.Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, idx):
            return {
                "image": torch.randn(4, 64, 64, 32),
                "label": torch.randint(0, 2, (1, 64, 64, 32)).float(),
            }

    model = get_model()
    val_loader = DataLoader(Mock3DMRIBatch(), batch_size=1)
    collector = RoundMetricCollector()

    res = evaluate_sliding_window(model, val_loader, roi_size=(32, 32, 16))
    collector.record_round(
        round_num=1,
        val_loss=res["val_loss"],
        dice_score=res["dice"],
        iou_score=res["iou"],
        num_samples=res["num_samples"],
    )

    print("[OK] Sliding-window evaluation output:", res)
    print("[OK] Round Metric Collector summary:", collector.get_summary())

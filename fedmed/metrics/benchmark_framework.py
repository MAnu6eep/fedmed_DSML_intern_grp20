"""
fedmed.metrics.benchmark_framework
==================================
Benchmarking and System-Monitoring framework for comparing FL and Centralized training strategies.
Supports metrics collection (Dice, Hausdorff Distance 95, IoU), execution timing, and GPU VRAM tracking.
"""

import os
import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import torch
import numpy as np

try:
    from monai.metrics import HausdorffDistanceMetric
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False


SUPPORTED_STRATEGIES = ["Centralized", "FedAvg", "FedProx", "SCAFFOLD"]


class SystemMonitor:
    """
    Monitors execution time and GPU VRAM resource usage.
    Supports both manual start/stop and context manager usage (`with SystemMonitor(): ...`).
    """
    def __init__(self, device: Optional[Union[torch.device, str]] = None):
        self.device = device
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: float = 0.0
        self.cuda_available: bool = torch.cuda.is_available()
        self.vram_peak_mb: float = 0.0
        self.vram_current_mb: float = 0.0

    def start(self):
        """Start tracking time and GPU memory."""
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass
        self.start_time = time.perf_counter()
        self.end_time = None
        return self

    def stop(self):
        """Stop tracking and compute metrics."""
        self.end_time = time.perf_counter()
        if self.start_time is not None:
            self.elapsed_time = self.end_time - self.start_time
        else:
            self.elapsed_time = 0.0

        if self.cuda_available:
            try:
                # Bytes to MB (1 MB = 1024 * 1024 bytes)
                self.vram_peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                self.vram_current_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            except Exception:
                self.vram_peak_mb = 0.0
                self.vram_current_mb = 0.0
        else:
            self.vram_peak_mb = 0.0
            self.vram_current_mb = 0.0

        return {
            "elapsed_time_seconds": self.elapsed_time,
            "vram_peak_mb": self.vram_peak_mb,
            "vram_current_mb": self.vram_current_mb,
            "cuda_available": self.cuda_available
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def compute_hausdorff_distance_95(
    preds: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    percentile: float = 95.0,
    include_background: bool = False
) -> float:
    """
    Computes the 95th percentile Hausdorff Distance (HD95) between 3D predictions and labels.
    
    Args:
        preds: Binary or class prediction tensor/array (shape: B, C, H, W, D or C, H, W, D or H, W, D).
        labels: Ground truth tensor/array (same shape format as preds).
        percentile: Distance percentile threshold (default: 95.0).
        include_background: Whether to include background channel in metric computation.
        
    Returns:
        float: HD95 distance metric value in mm (or voxels). Returns inf or fallback float on error.
    """
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Standardize input dimensions to (B, C, H, W, D) or (B, C, H, W)
    if preds.ndim == 3:
        preds = preds.unsqueeze(0).unsqueeze(0)
        labels = labels.unsqueeze(0).unsqueeze(0)
    elif preds.ndim == 4:
        preds = preds.unsqueeze(0)
        labels = labels.unsqueeze(0)

    # Ensure float / bool tensors
    preds = (preds > 0.5).float()
    labels = (labels > 0.5).float()

    # Check if either ground truth or prediction is completely empty
    if labels.sum() == 0 or preds.sum() == 0:
        return float('inf') if labels.sum() != preds.sum() else 0.0

    if MONAI_AVAILABLE:
        try:
            hd_metric = HausdorffDistanceMetric(
                include_background=include_background,
                percentile=percentile,
                reduction="mean"
            )
            val = hd_metric(preds, labels)
            if isinstance(val, torch.Tensor):
                val_item = val.item()
            else:
                val_item = float(val)
            
            if np.isnan(val_item) or np.isinf(val_item):
                return float('inf')
            return float(val_item)
        except Exception:
            pass

    # Simplified Euclidean distance metric fallback if MONAI calculation returns nan or fails
    try:
        pts_p = torch.nonzero(preds[0, 0], as_tuple=False).float()
        pts_l = torch.nonzero(labels[0, 0], as_tuple=False).float()
        if len(pts_p) == 0 or len(pts_l) == 0:
            return float('inf')
        
        # Subsample if point sets are too large for simple pairwise computation
        if len(pts_p) > 1000:
            idx = torch.randperm(len(pts_p))[:1000]
            pts_p = pts_p[idx]
        if len(pts_l) > 1000:
            idx = torch.randperm(len(pts_l))[:1000]
            pts_l = pts_l[idx]

        dists_p2l = torch.cdist(pts_p, pts_l).min(dim=1)[0]
        dists_l2p = torch.cdist(pts_l, pts_p).min(dim=1)[0]
        all_dists = torch.cat([dists_p2l, dists_l2p])
        
        hd95_val = torch.quantile(all_dists, percentile / 100.0).item()
        return float(hd95_val)
    except Exception:
        return float('inf')


@dataclass
class BenchmarkResult:
    """
    Standard dataclass capturing benchmark evaluation outputs for a given training approach.
    """
    strategy_name: str
    round_or_epoch: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    execution_time_seconds: float = 0.0
    dice_score: float = 0.0
    hausdorff_distance_95: float = 0.0
    iou_score: float = 0.0
    vram_peak_mb: float = 0.0
    vram_current_mb: float = 0.0
    cuda_available: bool = False
    region_metrics: Dict[str, float] = field(default_factory=dict)
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.strategy_name not in SUPPORTED_STRATEGIES and not self.strategy_name.startswith("Custom"):
            # Allow custom strategy names if needed, but validate baseline ones
            pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return asdict(self)


class BenchmarkSuite:
    """
    Master suite for accumulating, comparing, and exporting benchmark results across strategies.
    """
    def __init__(self, experiment_name: str = "FL_vs_Centralized_Benchmark"):
        self.experiment_name = experiment_name
        self.results: List[BenchmarkResult] = []

    def record_result(self, result: BenchmarkResult) -> None:
        """Record a single benchmark evaluation result."""
        self.results.append(result)

    def get_strategy_results(self, strategy_name: str) -> List[BenchmarkResult]:
        """Get all recorded benchmark results for a specific strategy."""
        return [r for r in self.results if r.strategy_name == strategy_name]

    def get_strategy_summary(self, strategy_name: str) -> Dict[str, Any]:
        """
        Compute summary metrics (mean/max) for a given strategy.
        """
        strat_results = self.get_strategy_results(strategy_name)
        if not strat_results:
            return {
                "strategy_name": strategy_name,
                "count": 0,
                "avg_dice_score": 0.0,
                "avg_hd95": 0.0,
                "avg_execution_time_seconds": 0.0,
                "max_vram_peak_mb": 0.0
            }

        dice_vals = [r.dice_score for r in strat_results]
        hd95_vals = [r.hausdorff_distance_95 for r in strat_results if not np.isinf(r.hausdorff_distance_95)]
        time_vals = [r.execution_time_seconds for r in strat_results]
        vram_vals = [r.vram_peak_mb for r in strat_results]

        return {
            "strategy_name": strategy_name,
            "count": len(strat_results),
            "avg_dice_score": float(np.mean(dice_vals)) if dice_vals else 0.0,
            "best_dice_score": float(np.max(dice_vals)) if dice_vals else 0.0,
            "avg_hd95": float(np.mean(hd95_vals)) if hd95_vals else 0.0,
            "avg_execution_time_seconds": float(np.mean(time_vals)) if time_vals else 0.0,
            "total_execution_time_seconds": float(np.sum(time_vals)) if time_vals else 0.0,
            "max_vram_peak_mb": float(np.max(vram_vals)) if vram_vals else 0.0,
        }

    def compare_all_strategies(self) -> List[Dict[str, Any]]:
        """
        Generates a comparative summary across all supported training strategies.
        """
        comparison = []
        strategies = SUPPORTED_STRATEGIES
        # Also include any other strategy present in results
        existing_strats = set(r.strategy_name for r in self.results)
        all_strats = list(dict.fromkeys(strategies + list(existing_strats)))

        for strat in all_strats:
            comparison.append(self.get_strategy_summary(strat))

        return comparison

    def export_json(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Exports recorded benchmark results and strategy comparisons to a JSON file.
        """
        if output_path is None:
            output_dir = Path("experiments/outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "benchmark_framework_results.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "experiment_name": self.experiment_name,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_records": len(self.results),
            "comparison_summary": self.compare_all_strategies(),
            "detailed_results": [r.to_dict() for r in self.results]
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return str(output_path)

"""experiments/verify_preprocessing_consistency.py

Experiment-level preprocessing configuration and verification suite.
Validates that MRI data from different hospital partitions (hospital_1, hospital_2, hospital_3)
passes through a consistent, reproducible MONAI normalization and spatial resampling pipeline.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedmed.data.loader import create_brats_dataloader
from fedmed.data.partitioner import partition_dirichlet, partition_iid
from scripts.setup_data import generate_mock_brats_data

# Configure log paths
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "preprocessing_consistency_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("preprocessing_consistency")


@dataclass
class PreprocessingConfig:
    """Standardized experiment-level preprocessing parameters."""

    roi_size: Tuple[int, int, int] = (64, 64, 32)
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    axcodes: str = "RAS"
    nonzero_norm: bool = True
    channel_wise_norm: bool = True
    resample_mode: Tuple[str, str] = ("bilinear", "nearest")
    scale_min: Optional[float] = None
    scale_max: Optional[float] = None


def verify_hospital_preprocessing_consistency(
    config: Optional[PreprocessingConfig] = None,
    num_hospitals: int = 3,
    samples_per_hospital: int = 2,
) -> Dict[str, Any]:
    """Tests MRI preprocessing across hospital nodes and records validation metrics.

    Args:
        config: PreprocessingConfig instance. Defaults to standard BraTS config.
        num_hospitals: Number of hospital partitions to evaluate (default: 3).
        samples_per_hospital: Mock sample volume count per hospital.

    Returns:
        Structured consistency report dictionary.
    """
    if config is None:
        config = PreprocessingConfig()

    logger.info("=" * 60)
    logger.info("Verifying Preprocessing Consistency Across Hospital Nodes")
    logger.info("=" * 60)
    logger.info(f"Configuration: {asdict(config)}")

    # 1. Ensure synthetic BraTS dataset is available
    raw_data_dir = PROJECT_ROOT / "data" / "raw" / "brats"
    images_dir = raw_data_dir / "imagesTr"
    labels_dir = raw_data_dir / "labelsTr"

    total_needed = num_hospitals * samples_per_hospital
    if not images_dir.exists() or len(list(images_dir.glob("*.nii.gz"))) < total_needed:
        logger.info(f"Generating synthetic BraTS dataset at {raw_data_dir}...")
        generate_mock_brats_data(raw_data_dir, num_samples=total_needed)

    image_files = sorted(list(images_dir.glob("*.nii.gz")))[:total_needed]
    label_files = sorted(list(labels_dir.glob("*.nii.gz")))[:total_needed]

    data_dicts = [
        {"image": str(img), "label": str(lbl)}
        for img, lbl in zip(image_files, label_files)
    ]

    # Partition volumes across 3 hospital nodes
    volume_ids = [f"case_{i:03d}" for i in range(len(data_dicts))]
    partitions = partition_iid(volume_ids, num_clients=num_hospitals, seed=42)

    hospital_metrics: Dict[str, Any] = {}
    shapes_list: List[Tuple[int, ...]] = []
    means_list: List[float] = []
    stds_list: List[float] = []

    for h_idx in range(1, num_hospitals + 1):
        hospital_id = f"hospital_{h_idx}"
        assigned_indices = [
            int(vid.split("_")[1]) for vid in partitions.get(f"client_{h_idx}", [])
        ]
        h_data_dicts = [data_dicts[i] for i in assigned_indices if i < len(data_dicts)]

        if not h_data_dicts:
            h_data_dicts = [data_dicts[h_idx - 1]]

        # Build MONAI dataloader with experiment preprocessing config
        dataloader = create_brats_dataloader(
            data_dicts=h_data_dicts,
            batch_size=1,
            is_train=False,
            use_cache=False,
            roi_size=config.roi_size,
            pixdim=config.pixdim,
            axcodes=config.axcodes,
            nonzero_norm=config.nonzero_norm,
            channel_wise_norm=config.channel_wise_norm,
            resample_mode=config.resample_mode,
            scale_min=config.scale_min,
            scale_max=config.scale_max,
        )

        # Inspect preprocessed batch
        first_batch = next(iter(dataloader))
        img_tensor = first_batch["image"]  # (B, C, H, W, D)
        lbl_tensor = first_batch["label"]

        spatial_shape = tuple(img_tensor.shape[1:])
        label_shape = tuple(lbl_tensor.shape[1:])
        shapes_list.append(spatial_shape)

        # Calculate non-zero voxel intensity statistics
        nonzero_mask = img_tensor != 0
        if nonzero_mask.any():
            nz_voxels = img_tensor[nonzero_mask].float()
            mean_val = float(nz_voxels.mean().item())
            std_val = float(nz_voxels.std().item())
            min_val = float(nz_voxels.min().item())
            max_val = float(nz_voxels.max().item())
        else:
            mean_val, std_val, min_val, max_val = 0.0, 1.0, 0.0, 0.0

        means_list.append(mean_val)
        stds_list.append(std_val)

        hospital_metrics[hospital_id] = {
            "samples_processed": len(h_data_dicts),
            "image_tensor_shape": list(img_tensor.shape),
            "spatial_shape": list(spatial_shape),
            "label_shape": list(label_shape),
            "channels": int(img_tensor.shape[1]),
            "intensity_stats_nonzero": {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
            },
        }

        logger.info(
            f"[{hospital_id}] Processed shape: {spatial_shape} | Nonzero mean: {mean_val:.4f}, std: {std_val:.4f}"
        )

    # Validate shape and intensity consistency across hospital nodes
    shapes_consistent = len(set(shapes_list)) == 1
    means_consistent = max(abs(m) for m in means_list) < 1.0  # Zero-centered mean check
    stds_consistent = all(0.5 <= s <= 2.0 for s in stds_list)  # Unit variance check

    is_overall_consistent = shapes_consistent and means_consistent and stds_consistent

    report = {
        "status": "PASSED" if is_overall_consistent else "FAILED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preprocessing_config": asdict(config),
        "hospital_count": num_hospitals,
        "consistency_checks": {
            "spatial_shape_uniformity": shapes_consistent,
            "intensity_mean_zero_centered": means_consistent,
            "intensity_variance_normalized": stds_consistent,
            "overall_consistent": is_overall_consistent,
        },
        "hospital_nodes": hospital_metrics,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("-" * 60)
    logger.info(f"Overall Preprocessing Consistency: {report['status']}")
    logger.info(f"Report saved to: {REPORT_PATH}")
    logger.info("=" * 60)

    return report


if __name__ == "__main__":
    verify_hospital_preprocessing_consistency()

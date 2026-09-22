"""experiments/verify_integrated_ml_pipeline.py

End-to-end integration and verification script for the complete Week 3 ML pipeline.
Verifies seamless integration from MRI preprocessing -> 3D U-Net sliding window inference ->
2D tumor slice extraction -> alignment verification -> federated runner execution ->
dashboard visualization payload generation.
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedmed.core.evaluation import (
    RoundMetricCollector,
    evaluate_and_extract_slices,
    get_federated_evaluate_fn,
)
from fedmed.core.model import get_model
from fedmed.data.loader import create_brats_dataloader
from fedmed.data.slice_extraction import (
    convert_slice_to_visualization_format,
    extract_tumor_slices,
    extract_visualization_payload,
    find_best_tumor_slices,
)
from scripts.setup_data import generate_mock_brats_data

# Configure artifact output directory
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "integrated_ml_pipeline_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("integrated_ml_pipeline")


def run_integrated_ml_pipeline_verification() -> Dict[str, Any]:
    """Runs end-to-end verification of the integrated ML pipeline."""
    logger.info("=" * 70)
    logger.info("  Starting Integrated ML Preprocessing, Inference & Visualization Verification")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[1/5] Hardware device: {device}")

    # 1. Ensure synthetic BraTS dataset exists
    raw_data_dir = PROJECT_ROOT / "data" / "raw" / "brats"
    images_dir = raw_data_dir / "imagesTr"
    labels_dir = raw_data_dir / "labelsTr"

    if not images_dir.exists() or len(list(images_dir.glob("*.nii.gz"))) < 4:
        logger.info("Generating mock 3D BraTS NIfTI dataset...")
        generate_mock_brats_data(raw_data_dir, num_samples=4)

    image_files = sorted(list(images_dir.glob("*.nii.gz")))[:4]
    label_files = sorted(list(labels_dir.glob("*.nii.gz")))[:4]
    data_dicts = [
        {"image": str(img), "label": str(lbl)}
        for img, lbl in zip(image_files, label_files)
    ]

    # 2. Build MONAI Preprocessing DataLoader
    logger.info("[2/5] Initializing MONAI dataloader with intensity normalization & spatial resampling...")
    val_loader = create_brats_dataloader(
        data_dicts,
        batch_size=1,
        is_train=False,
        use_cache=False,
        roi_size=(64, 64, 32),
        pixdim=(1.0, 1.0, 1.0),
        axcodes="RAS",
        nonzero_norm=True,
        channel_wise_norm=True,
    )

    first_batch = next(iter(val_loader))
    img_tensor = first_batch["image"]
    lbl_tensor = first_batch["label"]

    logger.info(f"   • Preprocessed Image Shape : {tuple(img_tensor.shape)}")
    logger.info(f"   • Preprocessed Label Shape : {tuple(lbl_tensor.shape)}")

    # 3. Model Instantiation & 3D U-Net Inference
    logger.info("[3/5] Instantiating 3D U-Net model and executing sliding window inference...")
    model = get_model(in_channels=4, out_channels=1).to(device)

    eval_results = evaluate_and_extract_slices(
        model=model,
        dataloader=val_loader,
        roi_size=(32, 32, 16),
        extract_slices=True,
        num_slices=3,
        device=device,
    )

    logger.info(f"   • Validation Dice Score : {eval_results['dice'] * 100:.2f}%")
    logger.info(f"   • Validation Loss       : {eval_results['val_loss']:.4f}")

    # 4. 2D Slice Extraction & Alignment Verification
    logger.info("[4/5] Verifying 2D tumor slice extraction and MRI-mask alignment...")
    vis_payload = eval_results.get("visualization_payload", {})
    by_orient = vis_payload.get("slices_by_orientation", {})

    alignment_checks = []
    base64_checks = []

    for orient_name, slice_list in by_orient.items():
        for s_info in slice_list:
            shape = s_info.get("spatial_shape", [])
            mri_arr = s_info.get("mri_slice", [])
            mask_arr = s_info.get("mask_slice", [])

            # Shape alignment assertion
            is_aligned = (
                len(shape) == 2
                and len(mri_arr) == shape[0]
                and len(mri_arr[0]) == shape[1]
                and len(mask_arr) == shape[0]
                and len(mask_arr[0]) == shape[1]
            )
            alignment_checks.append(is_aligned)

            # Base64 preview assertion
            has_b64 = s_info.get("mri_base64") is not None and s_info.get("mask_base64") is not None
            base64_checks.append(has_b64)

    all_aligned = len(alignment_checks) > 0 and all(alignment_checks)
    all_b64_valid = len(base64_checks) > 0 and all(base64_checks)

    logger.info(f"   • 2D MRI-Mask Shape Alignment : {'PASSED' if all_aligned else 'FAILED'}")
    logger.info(f"   • Base64 Image Preview Generation : {'PASSED' if all_b64_valid else 'FAILED'}")

    # 5. Federated Strategy Callback Integration
    logger.info("[5/5] Testing Flower federated evaluation callback and artifact export...")
    collector = RoundMetricCollector()
    eval_fn = get_federated_evaluate_fn(
        model=model,
        val_loader=val_loader,
        collector=collector,
        save_outputs=True,
    )

    params = [p.detach().cpu().numpy() for p in model.parameters()]
    fl_loss, fl_metrics = eval_fn(server_round=1, parameters=params, config={})

    fl_artifact = OUTPUT_DIR / "federated_visualization_round_1.json"
    fl_artifact_exists = fl_artifact.exists()

    logger.info(f"   • FL Evaluation Callback Output : Loss {fl_loss:.4f}, Dice {fl_metrics['dice']*100:.2f}%")
    logger.info(f"   • Federated Visualization Artifact Saved : {fl_artifact_exists}")

    # Consolidate Verification Summary
    is_fully_functional = (
        eval_results["num_samples"] > 0
        and all_aligned
        and all_b64_valid
        and fl_artifact_exists
    )

    report = {
        "status": "PASSED" if is_fully_functional else "FAILED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_stages": {
            "preprocessing": {
                "input_channels": 4,
                "spatial_shape": list(img_tensor.shape[1:]),
                "orientation": "RAS",
            },
            "inference": {
                "val_loss": eval_results["val_loss"],
                "val_dice": eval_results["dice"],
                "val_iou": eval_results["iou"],
                "samples_evaluated": eval_results["num_samples"],
            },
            "visualization_slicing": {
                "payload_status": vis_payload.get("status"),
                "orientations_extracted": list(by_orient.keys()),
                "mri_mask_shape_alignment": all_aligned,
                "base64_preview_generated": all_b64_valid,
            },
            "federated_integration": {
                "fl_callback_executed": True,
                "fl_round_artifact_exists": fl_artifact_exists,
            },
        },
        "issues_identified": [],
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("-" * 70)
    logger.info(f"  Integrated ML Pipeline Verification Status: {report['status']}")
    logger.info(f"  Report saved to: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    logger.info("=" * 70)

    return report


if __name__ == "__main__":
    run_integrated_ml_pipeline_verification()

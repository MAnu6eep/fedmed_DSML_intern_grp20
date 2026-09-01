"""experiments/centralized_baseline.py

End-to-end centralized 3D U-Net baseline training and validation execution script.
Generates/loads synthetic MRI dataset, applies MONAI transforms, trains 3D U-Net,
calculates Dice metrics, and saves model checkpoints + telemetry logs.
"""
import json
import logging
import sys
import time
from pathlib import Path
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fedmed.core.model import get_model
from fedmed.core.training import run_local_training
from fedmed.data.loader import create_brats_dataloader
from scripts.setup_data import generate_mock_brats_data

# Configure artifact output paths
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "outputs" / "centralized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "training.log"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.pt"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

# Setup Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("centralized_baseline")


def main():
    logger.info("=" * 60)
    logger.info("Starting Centralized 3D U-Net Baseline Experiment")
    logger.info("=" * 60)

    # 1. Ensure synthetic dataset exists
    raw_data_dir = PROJECT_ROOT / "data" / "raw" / "brats"
    images_dir = raw_data_dir / "imagesTr"
    labels_dir = raw_data_dir / "labelsTr"

    if not images_dir.exists() or len(list(images_dir.glob("*.nii.gz"))) == 0:
        logger.info(f"Generating synthetic 3D BraTS NIfTI dataset at {raw_data_dir}...")
        generate_mock_brats_data(raw_data_dir, num_samples=4)

    # Gather dataset files
    image_files = sorted(list(images_dir.glob("*.nii.gz")))
    label_files = sorted(list(labels_dir.glob("*.nii.gz")))

    logger.info(f"Found {len(image_files)} image cases and {len(label_files)} label cases.")

    data_dicts = [
        {"image": str(img), "label": str(lbl)}
        for img, lbl in zip(image_files, label_files)
    ]

    # Partition train / val splits
    split_idx = max(1, int(len(data_dicts) * 0.75))
    train_dicts = data_dicts[:split_idx]
    val_dicts = data_dicts[split_idx:] if split_idx < len(data_dicts) else data_dicts[:1]

    logger.info(f"Dataset split: {len(train_dicts)} training, {len(val_dicts)} validation samples.")

    # 2. Build MONAI DataLoaders
    logger.info("Initializing MONAI DataLoaders...")
    train_loader = create_brats_dataloader(train_dicts, batch_size=2, is_train=True, use_cache=False)
    val_loader = create_brats_dataloader(val_dicts, batch_size=2, is_train=False, use_cache=False)

    # 3. Instantiate 3D U-Net Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Instantiating 3D U-Net model on device: {device}")
    model = get_model(in_channels=4, out_channels=1).to(device)

    # 4. Run Training & Validation Loop
    epochs = 2
    learning_rate = 1e-4
    logger.info(f"Starting training run: {epochs} epochs, lr={learning_rate}...")

    start_time = time.time()
    metrics = run_local_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    total_time = time.time() - start_time
    avg_epoch_time = total_time / epochs

    logger.info("-" * 50)
    logger.info(f"✓ Training Completed in {total_time:.2f}s (Avg {avg_epoch_time:.2f}s/epoch)")
    logger.info(f"  • Final Training Loss   : {metrics['train_loss']:.4f}")
    logger.info(f"  • Final Validation Loss : {metrics['val_loss']:.4f}")
    logger.info(f"  • Final Validation Dice : {metrics['val_dice'] * 100:.2f}%")
    logger.info("-" * 50)

    # 5. Save Experiment Artifacts
    logger.info("Saving experiment artifacts...")

    # Save PyTorch Model Checkpoint
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "epochs": epochs,
        "metrics": metrics,
        "in_channels": 4,
        "out_channels": 1,
    }
    torch.save(checkpoint_data, CHECKPOINT_PATH)
    logger.info(f"  ✓ Saved checkpoint: {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}")

    # Save Metrics JSON
    metrics_payload = {
        "experiment": "centralized_baseline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": epochs,
        "total_time_seconds": round(total_time, 2),
        "avg_epoch_time_seconds": round(avg_epoch_time, 2),
        "train_loss": round(metrics["train_loss"], 6),
        "val_loss": round(metrics["val_loss"], 6),
        "val_dice": round(metrics["val_dice"], 6),
        "device": str(device),
        "dataset_samples": len(data_dicts),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"  ✓ Saved metrics JSON: {METRICS_PATH.relative_to(PROJECT_ROOT)}")

    logger.info("=" * 60)
    logger.info("Centralized Baseline Experiment Finished Successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

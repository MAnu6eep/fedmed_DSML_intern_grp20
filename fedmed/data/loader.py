"""fedmed/data/loader.py
MONAI 3D Data Loader and Preprocessing Transforms Preview for BraTS MRI Scans.
"""
from pathlib import Path
from typing import Dict, List, Optional

import monai.transforms as mt
from monai.data import Dataset, DataLoader

def get_brats_transforms(keys: List[str] = ["image", "label"]) -> mt.Compose:
    """Returns the standard MONAI transform pipeline for 3D multi-modal BraTS scans.
    
    Args:
        keys: Data dictionary keys to transform (default: ["image", "label"])
        
    Returns:
        MONAI Compose transform object.
    """
    return mt.Compose([
        mt.LoadImaged(keys=keys),
        mt.EnsureChannelFirstd(keys=keys),
        mt.Orientationd(keys=keys, axcodes="RAS"),
        mt.Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        mt.NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ])

def create_brats_dataloader(
    data_dir: Path,
    batch_size: int = 2,
    num_workers: int = 0
) -> DataLoader:
    """Creates a PyTorch/MONAI DataLoader for synthetic or real BraTS data.
    
    Args:
        data_dir: Path to directory containing imagesTr and labelsTr
        batch_size: Batch size for loading
        num_workers: Worker processes for dataloader
        
    Returns:
        MONAI DataLoader instance.
    """
    images_dir = data_dir / "imagesTr"
    labels_dir = data_dir / "labelsTr"
    
    image_files = sorted(list(images_dir.glob("*.nii.gz")))
    label_files = sorted(list(labels_dir.glob("*.nii.gz")))
    
    data_dicts = [
        {"image": img, "label": lbl}
        for img, lbl in zip(image_files, label_files)
    ]
    
    transforms = get_brats_transforms()
    ds = Dataset(data=data_dicts, transform=transforms)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

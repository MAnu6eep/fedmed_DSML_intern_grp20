"""fedmed/data/loader.py
Volumetric medical image loader and MONAI dictionary transforms for BraTS NIfTI files.
"""
from pathlib import Path
from typing import Dict, List, Tuple
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    Spacingd,
)

def get_brats_transforms(roi_size: Tuple[int, int, int] = (64, 64, 32)) -> Tuple[Compose, Compose]:
    """Returns training and validation dictionary transform pipelines."""
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=2,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    return train_transforms, val_transforms


def create_brats_dataloader(
    data_dicts: List[Dict[str, str]],
    batch_size: int = 2,
    is_train: bool = True,
    use_cache: bool = False,
) -> DataLoader:
    """Creates a PyTorch DataLoader with MONAI dictionary transforms."""
    train_tf, val_tf = get_brats_transforms()
    transforms = train_tf if is_train else val_tf

    if use_cache:
        dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0)
    else:
        dataset = Dataset(data=data_dicts, transform=transforms)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=2,
        pin_memory=True,
    )

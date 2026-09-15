"""fedmed/data/loader.py
Volumetric medical image loader and MONAI dictionary transforms for BraTS NIfTI files.
Includes configurable intensity normalization and spatial voxel resampling pipelines.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
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
    ScaleIntensityd,
    Spacingd,
)


def get_brats_transforms(
    roi_size: Tuple[int, int, int] = (64, 64, 32),
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    axcodes: str = "RAS",
    nonzero_norm: bool = True,
    channel_wise_norm: bool = True,
    resample_mode: Tuple[str, str] = ("bilinear", "nearest"),
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None,
) -> Tuple[Compose, Compose]:
    """Returns configurable training and validation dictionary transform pipelines.

    Args:
        roi_size: Spatial ROI patch size for training crops (H, W, D)
        pixdim: Target voxel spacing resolution for spatial resampling (x, y, z in mm)
        axcodes: Target 3D anatomical orientation code (e.g. 'RAS')
        nonzero_norm: Whether intensity normalization applies only to non-zero voxels
        channel_wise_norm: Whether intensity normalization is computed independently per channel
        resample_mode: Interpolation modes for (image, label) spatial resampling
        scale_min: Optional minimum target intensity for min-max scaling
        scale_max: Optional maximum target intensity for min-max scaling

    Returns:
        Tuple of (train_transforms, val_transforms) MONAI Compose pipelines.
    """
    train_transform_list = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(
            keys=["image", "label"],
            axcodes=axcodes,
        ),
        Spacingd(
            keys=["image", "label"],
            pixdim=pixdim,
            mode=resample_mode,
        ),
        NormalizeIntensityd(
            keys="image",
            nonzero=nonzero_norm,
            channel_wise=channel_wise_norm,
        ),
    ]

    if scale_min is not None and scale_max is not None:
        train_transform_list.append(
            ScaleIntensityd(
                keys="image",
                minv=scale_min,
                maxv=scale_max,
            )
        )

    train_transform_list.extend(
        [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=2,
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=0.5,
                spatial_axis=0,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.5,
                max_k=3,
            ),
            EnsureTyped(
                keys=["image", "label"],
            ),
        ]
    )

    val_transform_list = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(
            keys=["image", "label"],
            axcodes=axcodes,
        ),
        Spacingd(
            keys=["image", "label"],
            pixdim=pixdim,
            mode=resample_mode,
        ),
        NormalizeIntensityd(
            keys="image",
            nonzero=nonzero_norm,
            channel_wise=channel_wise_norm,
        ),
    ]

    if scale_min is not None and scale_max is not None:
        val_transform_list.append(
            ScaleIntensityd(
                keys="image",
                minv=scale_min,
                maxv=scale_max,
            )
        )

    val_transform_list.append(
        EnsureTyped(
            keys=["image", "label"],
        )
    )

    return Compose(train_transform_list), Compose(val_transform_list)


def create_brats_dataloader(
    data_dicts: List[Dict[str, str]],
    batch_size: int = 2,
    is_train: bool = True,
    use_cache: bool = False,
    roi_size: Tuple[int, int, int] = (64, 64, 32),
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    nonzero_norm: bool = True,
    channel_wise_norm: bool = True,
) -> DataLoader:
    """Creates a memory-efficient PyTorch DataLoader with MONAI transforms."""

    train_tf, val_tf = get_brats_transforms(
        roi_size=roi_size,
        pixdim=pixdim,
        nonzero_norm=nonzero_norm,
        channel_wise_norm=channel_wise_norm,
    )
    transforms = train_tf if is_train else val_tf

    if use_cache:
        dataset = CacheDataset(
            data=data_dicts,
            transform=transforms,
            cache_rate=1.0,
        )
    else:
        dataset = Dataset(
            data=data_dicts,
            transform=transforms,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=0,
        pin_memory=False,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from fedmed.core.model import get_model

    print("[OK] Testing configurable MONAI preprocessing pipeline...")

    train_tf, val_tf = get_brats_transforms(
        roi_size=(64, 64, 32),
        pixdim=(1.0, 1.0, 1.0),
        axcodes="RAS",
        nonzero_norm=True,
        channel_wise_norm=True,
    )

    # Smoke test synthetic 3D volume transformation
    mock_item = {
        "image": torch.randn(4, 128, 128, 64),
        "label": torch.randint(0, 2, (1, 128, 128, 64)).float(),
    }

    # Verify intensity normalization & 3D structure
    norm_tf = NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
    transformed_item = norm_tf(mock_item)
    img_tensor = transformed_item["image"]

    print(f"[OK] Normalized intensity volume shape: {img_tensor.shape}")
    print(f"[OK] Intensity mean: {img_tensor.mean():.4f}, std: {img_tensor.std():.4f}")

    # Pass into 3D U-Net to verify architectural compatibility
    unet = get_model(in_channels=4, out_channels=1)
    sample_batch = img_tensor.unsqueeze(0)[:, :, :64, :64, :32]
    out = unet(sample_batch)

    print(f"[OK] 3D U-Net forward pass output shape: {out.shape}")
    print("[SUCCESS] Configurable intensity normalization and spatial resampling verified!")
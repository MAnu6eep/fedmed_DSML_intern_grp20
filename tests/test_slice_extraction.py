"""tests/test_slice_extraction.py

Automated unit tests for 3D tumor slice extraction and visualization payload formatting.
"""

import numpy as np
import pytest
import torch

from fedmed.data.slice_extraction import (
    convert_slice_to_visualization_format,
    extract_tumor_slices,
    extract_visualization_payload,
    find_best_tumor_slices,
)


def test_find_best_tumor_slices_with_tumor():
    """Verifies that slices containing the largest tumor regions are ranked first."""
    mask = np.zeros((64, 64, 32), dtype=np.float32)
    # Add small tumor region at z=10 (10x10=100 voxels)
    mask[20:30, 20:30, 10] = 1.0
    # Add larger tumor region at z=20 (20x20=400 voxels)
    mask[20:40, 20:40, 20] = 1.0

    best_slices = find_best_tumor_slices(mask, axis=2, top_k=2)

    assert len(best_slices) == 2
    assert best_slices[0] == 20  # Largest tumor slice
    assert best_slices[1] == 10  # Second largest tumor slice


def test_find_best_tumor_slices_empty_mask():
    """Verifies graceful fallback to middle slices when no tumor voxels exist."""
    empty_mask = np.zeros((64, 64, 32), dtype=np.float32)
    fallback_slices = find_best_tumor_slices(empty_mask, axis=2, top_k=3)

    assert len(fallback_slices) == 3
    # Middle slice of 32 is 16; top_k=3 centered around 16
    assert 15 in fallback_slices or 16 in fallback_slices


def test_extract_tumor_slices_matching_dimensions():
    """Verifies that extracted MRI slice and mask slice have matching 2D spatial dimensions."""
    image_tensor = torch.randn(4, 64, 64, 32)
    mask_tensor = torch.zeros(1, 64, 64, 32)
    mask_tensor[0, 15:25, 15:25, 12] = 1.0

    img_slice, mask_slice = extract_tumor_slices(image_tensor, mask_tensor, axis=2, slice_idx=12)

    assert isinstance(img_slice, np.ndarray)
    assert isinstance(mask_slice, np.ndarray)
    assert img_slice.shape == (64, 64)
    assert mask_slice.shape == (64, 64)
    assert img_slice.shape == mask_slice.shape
    assert np.sum(mask_slice) == 100


def test_convert_slice_to_visualization_format():
    """Verifies schema structure, normalization bounds, and tumor metadata in formatted slice output."""
    img_slice = np.random.randn(64, 64)
    mask_slice = np.zeros((64, 64), dtype=np.float32)
    mask_slice[10:20, 10:20] = 1.0

    formatted = convert_slice_to_visualization_format(
        image_slice=img_slice,
        mask_slice=mask_slice,
        slice_idx=15,
        axis=2,
        metadata={"hospital_id": "hospital_1"},
    )

    assert formatted["slice_index"] == 15
    assert formatted["axis"] == 2
    assert formatted["orientation"] == "axial"
    assert formatted["spatial_shape"] == [64, 64]
    assert formatted["tumor_area_pixels"] == 100
    assert formatted["has_tumor"] is True
    assert formatted["metadata"]["hospital_id"] == "hospital_1"

    # Check intensity bounds [0, 255]
    mri_arr = np.array(formatted["mri_slice"])
    assert mri_arr.min() >= 0
    assert mri_arr.max() <= 255


def test_extract_visualization_payload():
    """Verifies high-level payload extraction across axial, coronal, and sagittal orientations."""
    image_3d = np.random.randn(32, 32, 32)
    mask_3d = np.zeros((32, 32, 32), dtype=np.float32)
    mask_3d[10:20, 10:20, 10:20] = 1.0

    payload = extract_visualization_payload(image_3d, mask_3d, num_slices=2)

    assert payload["status"] == "SUCCESS"
    assert "slices_by_orientation" in payload
    by_orient = payload["slices_by_orientation"]

    for orient in ["sagittal", "coronal", "axial"]:
        assert orient in by_orient
        assert len(by_orient[orient]) == 2
        assert by_orient[orient][0]["has_tumor"] is True

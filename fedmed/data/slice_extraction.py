"""fedmed/data/slice_extraction.py

Utilities to extract meaningful 2D slices from 3D tumor segmentation outputs
and MRI volumes for visualization in the FedMed dashboard.
"""

import base64
import io
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def _to_numpy(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Converts PyTorch tensor or NumPy array to NumPy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def find_best_tumor_slices(
    mask: Union[torch.Tensor, np.ndarray],
    axis: int = 2,
    top_k: int = 3,
) -> List[int]:
    """Identifies slice indices along a spatial axis containing the highest tumor voxel content.

    Args:
        mask: 3D or 4D/5D binary/categorical segmentation mask tensor or array.
        axis: Spatial axis for slice extraction (0=sagittal, 1=coronal, 2=axial).
        top_k: Maximum number of top slice indices to return.

    Returns:
        List of integer slice indices ordered by descending tumor voxel count.
    """
    arr = _to_numpy(mask)

    # Squeeze leading batch/channel dimensions if present
    while arr.ndim > 3:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D spatial mask tensor/array, but got shape {arr.shape}")

    num_slices = arr.shape[axis]
    if num_slices == 0:
        return [0]

    # Compute positive tumor voxel counts along the specified slice axis
    sum_axes = tuple(i for i in range(3) if i != axis)
    slice_counts = np.sum(arr > 0, axis=sum_axes)

    total_tumor_voxels = np.sum(slice_counts)

    if total_tumor_voxels == 0:
        # Fallback to middle slices if tumor is absent
        mid = num_slices // 2
        half_k = top_k // 2
        start_idx = max(0, mid - half_k)
        end_idx = min(num_slices, start_idx + top_k)
        return list(range(start_idx, end_idx))

    # Rank slice indices by tumor area descending
    sorted_indices = np.argsort(slice_counts)[::-1]
    best_indices = [int(idx) for idx in sorted_indices if slice_counts[idx] > 0]

    if len(best_indices) < top_k:
        # Fill remaining slots with adjacent slices if available
        for idx in sorted_indices:
            if int(idx) not in best_indices:
                best_indices.append(int(idx))
            if len(best_indices) == top_k:
                break

    return best_indices[:top_k]


def extract_tumor_slices(
    image: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    axis: int = 2,
    slice_idx: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts matching 2D MRI and 2D tumor mask slices along a specified axis.

    Args:
        image: 3D or 4D MRI image tensor/array.
        mask: 3D or 4D tumor mask tensor/array.
        axis: Spatial axis for slice extraction (0=sagittal, 1=coronal, 2=axial).
        slice_idx: Target slice index. If None, auto-selects slice with highest tumor area.

    Returns:
        Tuple of (image_slice, mask_slice) 2D NumPy arrays with matching spatial dimensions.
    """
    img_arr = _to_numpy(image)
    mask_arr = _to_numpy(mask)

    # Handle multi-channel image (e.g. 4 channels -> select channel 0 FLAIR or average)
    while img_arr.ndim > 3:
        img_arr = img_arr[0]
    while mask_arr.ndim > 3:
        mask_arr = mask_arr[0]

    if img_arr.ndim != 3 or mask_arr.ndim != 3:
        raise ValueError(
            f"Expected 3D spatial inputs, got image shape {img_arr.shape} and mask shape {mask_arr.shape}"
        )

    if img_arr.shape != mask_arr.shape:
        raise ValueError(
            f"Image spatial shape {img_arr.shape} does not match mask spatial shape {mask_arr.shape}"
        )

    if slice_idx is None:
        best_slices = find_best_tumor_slices(mask_arr, axis=axis, top_k=1)
        slice_idx = best_slices[0]

    slice_idx = max(0, min(img_arr.shape[axis] - 1, int(slice_idx)))

    if axis == 0:
        img_slice = img_arr[slice_idx, :, :]
        mask_slice = mask_arr[slice_idx, :, :]
    elif axis == 1:
        img_slice = img_arr[:, slice_idx, :]
        mask_slice = mask_arr[:, slice_idx, :]
    elif axis == 2:
        img_slice = img_arr[:, :, slice_idx]
        mask_slice = mask_arr[:, :, slice_idx]
    else:
        raise ValueError(f"Invalid spatial axis {axis}. Must be 0 (sagittal), 1 (coronal), or 2 (axial).")

    # Verify matching dimensions
    assert img_slice.shape == mask_slice.shape, (
        f"Extracted MRI slice shape {img_slice.shape} does not match mask slice shape {mask_slice.shape}"
    )

    return img_slice, mask_slice


def convert_slice_to_visualization_format(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    slice_idx: int,
    axis: int = 2,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Formats 2D MRI and mask slices into a visualization-ready structure.

    Args:
        image_slice: 2D NumPy array of the MRI slice.
        mask_slice: 2D NumPy array of the tumor mask slice.
        slice_idx: Slice index within the 3D volume.
        axis: Spatial axis index (0=sagittal, 1=coronal, 2=axial).
        metadata: Optional dictionary of additional metadata (e.g., patient ID, round num).

    Returns:
        Dictionary payload containing 2D arrays, tumor area, metadata, and optional base64 preview.
    """
    axis_names = {0: "sagittal", 1: "coronal", 2: "axial"}
    orientation_name = axis_names.get(axis, "axial")

    # Normalize image slice to uint8 [0, 255]
    img_min, img_max = image_slice.min(), image_slice.max()
    if img_max > img_min:
        norm_img = ((image_slice - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        norm_img = np.zeros_like(image_slice, dtype=np.uint8)

    binary_mask = (mask_slice > 0).astype(np.uint8)
    tumor_area_pixels = int(np.sum(binary_mask))

    payload = {
        "slice_index": int(slice_idx),
        "axis": axis,
        "orientation": orientation_name,
        "spatial_shape": [int(s) for s in image_slice.shape],
        "tumor_area_pixels": tumor_area_pixels,
        "has_tumor": tumor_area_pixels > 0,
        "mri_slice": norm_img.tolist(),
        "mask_slice": binary_mask.tolist(),
        "metadata": metadata or {},
    }

    # Generate lightweight base64 PNG preview strings if PIL is available
    try:
        from PIL import Image

        # Convert image and mask to PNG bytes
        img_pil = Image.fromarray(norm_img)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        payload["mri_base64"] = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Colorize tumor mask (red overlay for visualization)
        mask_rgba = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.uint8)
        mask_rgba[binary_mask > 0] = [255, 0, 0, 180]  # Semi-transparent red
        mask_pil = Image.fromarray(mask_rgba, mode="RGBA")
        buffer_mask = io.BytesIO()
        mask_pil.save(buffer_mask, format="PNG")
        payload["mask_base64"] = base64.b64encode(buffer_mask.getvalue()).decode("utf-8")
    except Exception:
        payload["mri_base64"] = None
        payload["mask_base64"] = None

    return payload


def extract_visualization_payload(
    image_3d: Union[torch.Tensor, np.ndarray],
    mask_3d: Union[torch.Tensor, np.ndarray],
    num_slices: int = 3,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """High-level utility to extract visualization payloads across axial, coronal, and sagittal views.

    Args:
        image_3d: 3D or 4D MRI volume image tensor/array.
        mask_3d: 3D or 4D tumor segmentation mask tensor/array.
        num_slices: Number of top slices to extract per orientation.
        metadata: Optional experiment/hospital metadata.

    Returns:
        Structured visualization payload dictionary with slices grouped by orientation view.
    """
    orientations = {0: "sagittal", 1: "coronal", 2: "axial"}
    result: Dict[str, Any] = {
        "status": "SUCCESS",
        "metadata": metadata or {},
        "slices_by_orientation": {},
    }

    for axis_idx, axis_name in orientations.items():
        best_indices = find_best_tumor_slices(mask_3d, axis=axis_idx, top_k=num_slices)
        extracted_list = []
        for s_idx in best_indices:
            img_s, mask_s = extract_tumor_slices(image_3d, mask_3d, axis=axis_idx, slice_idx=s_idx)
            formatted = convert_slice_to_visualization_format(
                image_slice=img_s,
                mask_slice=mask_s,
                slice_idx=s_idx,
                axis=axis_idx,
                metadata=metadata,
            )
            extracted_list.append(formatted)
        result["slices_by_orientation"][axis_name] = extracted_list

    return result


if __name__ == "__main__":
    print("Testing 3D tumor slice extraction utilities...")

    # Create synthetic 3D volume (128x128x64) with a synthetic tumor region at z=30..40
    synth_img = np.random.randn(128, 128, 64)
    synth_mask = np.zeros((128, 128, 64), dtype=np.float32)
    synth_mask[40:80, 40:80, 30:40] = 1.0  # Tumor region

    # 1. Test finding best tumor slices
    best_axial = find_best_tumor_slices(synth_mask, axis=2, top_k=3)
    print(f"[OK] Best axial tumor slices: {best_axial}")
    assert len(best_axial) == 3
    assert 30 <= best_axial[0] < 40

    # 2. Test extracting slices and dimension check
    img_slice, mask_slice = extract_tumor_slices(synth_img, synth_mask, axis=2, slice_idx=best_axial[0])
    assert img_slice.shape == mask_slice.shape == (128, 128)
    print(f"[OK] Extracted matching 2D slices shape: {img_slice.shape}")

    # 3. Test empty tumor mask fallback
    empty_mask = np.zeros((128, 128, 64), dtype=np.float32)
    fallback_slices = find_best_tumor_slices(empty_mask, axis=2, top_k=3)
    print(f"[OK] Empty mask fallback slice indices: {fallback_slices}")
    assert len(fallback_slices) == 3

    # 4. Test full visualization payload creation
    payload = extract_visualization_payload(synth_img, synth_mask, num_slices=2)
    assert payload["status"] == "SUCCESS"
    assert "axial" in payload["slices_by_orientation"]
    print(f"[SUCCESS] 3D tumor slice extraction verified!")

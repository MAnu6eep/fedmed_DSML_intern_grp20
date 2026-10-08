"""scripts/setup_data.py
Generates synthetic 3D multi-modal Brain MRI NIfTI volumes (BraTS-like) for isolated hospital nodes.
"""
from pathlib import Path
import nibabel as nib
import numpy as np

def generate_mock_brats_data(output_dir: Path, num_samples: int = 3, prefix: str = "BRATS"):
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} mock 3D BraTS NIfTI cases at {output_dir}...")
    affine = np.eye(4)

    for i in range(1, num_samples + 1):
        shape_3d = (64, 64, 32)
        mock_volume = np.random.randn(*shape_3d, 4).astype(np.float32)
        
        mock_mask = np.zeros(shape_3d, dtype=np.uint8)
        mock_mask[20:45, 20:45, 10:25] = np.random.choice([0, 1], size=(25, 25, 15), p=[0.3, 0.7])

        img_nii = nib.Nifti1Image(mock_volume, affine)
        mask_nii = nib.Nifti1Image(mock_mask, affine)

        img_path = images_dir / f"{prefix}_{i:03d}.nii.gz"
        mask_path = labels_dir / f"{prefix}_{i:03d}.nii.gz"

        nib.save(img_nii, img_path)
        nib.save(mask_nii, mask_path)
        print(f"  [OK] Created: {img_path.name} & {mask_path.name}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent

    # Global raw baseline data
    generate_mock_brats_data(base_dir / "data" / "raw" / "brats", num_samples=3, prefix="BRATS_RAW")

    # Isolated Hospital Datasets
    for hosp in ["hospital_a", "hospital_b", "hospital_c", "hospital_d", "hospital_e"]:
        hosp_data_dir = base_dir / "hospitals" / hosp / "data"
        generate_mock_brats_data(hosp_data_dir, num_samples=3, prefix=f"{hosp.upper()}")

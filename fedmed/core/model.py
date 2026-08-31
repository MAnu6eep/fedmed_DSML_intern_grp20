"""fedmed/core/model.py
3D U-Net architectural definition utilizing MONAI deep learning primitives for volumetric brain tumor segmentation.
"""
from typing import Sequence, Tuple
import torch
import torch.nn as nn
from monai.networks.nets import UNet


class FedMedUNet3D(nn.Module):
    """Configurable 3D U-Net for multi-modal MRI segmentation (e.g., 4 channels -> 1 or 3 classes)."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        channels: Sequence[int] = (16, 32, 64, 128, 256),
        strides: Sequence[int] = (2, 2, 2, 2),
        num_res_units: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout,
            norm="batch",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass expecting input shape: (Batch, Channels, Height, Width, Depth)."""
        return self.net(x)


def get_model(in_channels: int = 4, out_channels: int = 1) -> FedMedUNet3D:
    """Factory helper to instantiate the standard FedMed 3D U-Net."""
    return FedMedUNet3D(in_channels=in_channels, out_channels=out_channels)


if __name__ == "__main__":
    # Smoke test tensor forward-pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    dummy_input = torch.randn(1, 4, 64, 64, 32).to(device)
    output = model(dummy_input)
    print(f"✓ Model initialized successfully. Output shape: {output.shape}")

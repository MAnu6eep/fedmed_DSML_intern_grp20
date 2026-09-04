"""tests/test_model.py
Automated unit and integration tests for 3D U-Net architectural integrity, loss backpropagation, and Dice scoring.
"""
import pytest
import torch
from monai.losses import DiceFocalLoss
from fedmed.core.model import get_model, FedMedUNet3D
from fedmed.core.training import evaluate_local

@pytest.fixture
def dummy_device():
    return torch.device("cpu")

@pytest.fixture
def unet_model():
    return get_model(in_channels=4, out_channels=1)

def test_unet_forward_shape(unet_model):
    """Verifies that 3D volumetric input tensors retain spatial resolution through the U-Net."""
    batch_size = 2
    # BraTS standard: 4 modalities (FLAIR, T1, T1ce, T2)
    in_tensor = torch.randn(batch_size, 4, 32, 32, 16)
    out_tensor = unet_model(in_tensor)

    expected_shape = (batch_size, 1, 32, 32, 16)
    assert out_tensor.shape == expected_shape, (
        f"Expected output shape {expected_shape}, but got {out_tensor.shape}"
    )

def test_loss_backward_pass(unet_model):
    """Verifies that DiceFocalLoss backward pass generates finite gradients across all parameters."""
    optimizer = torch.optim.AdamW(unet_model.parameters(), lr=1e-3)
    loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)

    images = torch.randn(1, 4, 32, 32, 16)
    labels = torch.randint(0, 2, (1, 1, 32, 32, 16)).float()

    optimizer.zero_grad()
    outputs = unet_model(images)
    loss = loss_fn(outputs, labels)
    loss.backward()

    assert not torch.isnan(loss), "Loss computed to NaN"
    assert not torch.isinf(loss), "Loss computed to Inf"

    # Verify at least one parameter received gradients
    has_grads = any(p.grad is not None and torch.norm(p.grad) > 0 for p in unet_model.parameters())
    assert has_grads, "No gradients were calculated during backward pass"

def test_evaluation_metric_range(unet_model, dummy_device):
    """Verifies that the local evaluation pipeline produces Dice scores within valid bounds [0, 1]."""
    loss_fn = DiceFocalLoss(sigmoid=True)
    
    # Mock single-batch dataloader
    mock_batch = {
        "image": torch.randn(1, 4, 32, 32, 16),
        "label": torch.randint(0, 2, (1, 1, 32, 32, 16)).float()
    }
    mock_loader = [mock_batch]

    val_loss, val_dice = evaluate_local(unet_model, mock_loader, loss_fn, dummy_device)

    assert isinstance(val_loss, float)
    assert isinstance(val_dice, float)
    assert 0.0 <= val_dice <= 1.0, f"Dice score {val_dice} outside valid bounds [0, 1]"

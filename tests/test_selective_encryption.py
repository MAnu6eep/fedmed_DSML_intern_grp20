import torch

from fedmed.privacy.tenseal_engine import TenSEALEngine


def test_select_parameters():
    """Selected parameters should be separated from plaintext parameters."""
    engine = TenSEALEngine()

    state_dict = {
        "conv1.weight": torch.randn(2, 2),
        "conv1.bias": torch.randn(2),
        "conv2.weight": torch.randn(2, 2),
    }

    encrypted_parameters = [
        "conv1.weight",
        "conv2.weight",
    ]

    encrypted, plaintext = engine.select_parameters(
        state_dict,
        encrypted_parameters,
    )

    assert set(encrypted.keys()) == {
        "conv1.weight",
        "conv2.weight",
    }

    assert set(plaintext.keys()) == {
        "conv1.bias",
    }


def test_prepare_selected_parameters():
    """Selected parameters should be encrypted and serialized."""
    engine = TenSEALEngine()

    state_dict = {
        "conv1.weight": torch.randn(2, 2),
        "conv1.bias": torch.randn(2),
    }

    prepared = engine.prepare_model_parameters(
        state_dict,
        ["conv1.weight"],
    )

    assert "conv1.weight" in prepared["encrypted"]
    assert "conv1.bias" in prepared["plaintext"]

    ciphertext_data = prepared["encrypted"]["conv1.weight"]

    assert isinstance(ciphertext_data["ciphertext"], bytes)
    assert ciphertext_data["shape"] == (2, 2)


def test_encrypted_parameter_can_be_decrypted():
    """Serialized encrypted parameters should be recoverable."""
    engine = TenSEALEngine()

    original = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=torch.float32,
    )

    state_dict = {
        "conv1.weight": original,
    }

    prepared = engine.prepare_model_parameters(
        state_dict,
        ["conv1.weight"],
    )

    encrypted_data = prepared["encrypted"]["conv1.weight"]

    ciphertext = engine.deserialize_ciphertext(
        encrypted_data["ciphertext"]
    )

    decrypted = engine.decrypt_vector(
        ciphertext,
        torch.Size(encrypted_data["shape"]),
    )

    assert torch.allclose(
        decrypted,
        original,
        atol=1e-3,
    )


def test_unselected_parameters_remain_plaintext():
    """Parameters not selected for encryption remain plaintext."""
    engine = TenSEALEngine()

    weight = torch.randn(2, 2)
    bias = torch.randn(2)

    state_dict = {
        "weight": weight,
        "bias": bias,
    }

    prepared = engine.prepare_model_parameters(
        state_dict,
        ["weight"],
    )

    assert torch.equal(
        prepared["plaintext"]["bias"],
        bias,
    )

    assert "weight" in prepared["encrypted"]
    assert "bias" not in prepared["encrypted"]
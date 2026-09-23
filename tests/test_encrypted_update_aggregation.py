import torch
import numpy as np

from fedmed.privacy.encrypted_update import (
    aggregate_encrypted_parameters,
    create_encrypted_update,
    decode_encrypted_update_from_flower,
    deserialize_encrypted_update,
    encode_encrypted_update_for_flower,
    serialize_encrypted_update,
)
from fedmed.privacy.tenseal_engine import TenSEALEngine


def test_encrypted_updates_are_aggregated_without_plaintext_conversion():
    engine = TenSEALEngine()

    client_1 = {
        "layer.weight": torch.tensor([1.0, 2.0, 3.0]),
        "layer.bias": torch.tensor([10.0]),
    }

    client_2 = {
        "layer.weight": torch.tensor([4.0, 5.0, 6.0]),
        "layer.bias": torch.tensor([20.0]),
    }

    update_1 = create_encrypted_update(
        engine,
        client_1,
        ["layer.weight"],
    )

    update_2 = create_encrypted_update(
        engine,
        client_2,
        ["layer.weight"],
    )

    # Simulate client -> server transmission.
    update_1 = deserialize_encrypted_update(
        serialize_encrypted_update(update_1)
    )
    update_2 = deserialize_encrypted_update(
        serialize_encrypted_update(update_2)
    )

    # Server aggregates ciphertexts directly.
    aggregated = aggregate_encrypted_parameters(
        engine,
        [update_1, update_2],
    )

    assert "layer.weight" in aggregated

    # Decryption happens only after aggregation for verification.
    ciphertext = engine.deserialize_ciphertext(
        aggregated["layer.weight"]
    )

    result = engine.decrypt_vector(
        ciphertext,
        original_shape=torch.Size([3]),
    )

    expected = torch.tensor([5.0, 7.0, 9.0])

    assert torch.allclose(result, expected, atol=1e-3)


def test_unselected_parameters_remain_plaintext():
    engine = TenSEALEngine()

    state_dict = {
        "layer.weight": torch.tensor([1.0, 2.0, 3.0]),
        "layer.bias": torch.tensor([10.0]),
    }

    update = create_encrypted_update(
        engine,
        state_dict,
        ["layer.weight"],
    )

    assert "layer.weight" in update["encrypted"]
    assert "layer.bias" in update["plaintext"]

    assert isinstance(
        update["encrypted"]["layer.weight"]["ciphertext"],
        bytes,
    )

    assert isinstance(
        update["plaintext"]["layer.bias"],
        __import__("numpy").ndarray,
    )
def test_encrypted_update_can_travel_through_flower_transport():
    engine = TenSEALEngine()

    state_dict = {
        "layer.weight": torch.tensor([1.0, 2.0, 3.0]),
        "layer.bias": torch.tensor([10.0]),
    }

    update = create_encrypted_update(
        engine,
        state_dict,
        ["layer.weight"],
    )

    # Simulate Flower transport.
    payload = encode_encrypted_update_for_flower(update)

    assert isinstance(payload, np.ndarray)
    assert payload.dtype == np.uint8

    received = decode_encrypted_update_from_flower(payload)

    assert "layer.weight" in received["encrypted"]
    assert "layer.bias" in received["plaintext"]



def test_end_to_end_selective_encrypted_aggregation():
    engine = TenSEALEngine()

    client_1 = {
        "layer.weight": torch.tensor([1.0, 2.0, 3.0]),
        "layer.bias": torch.tensor([10.0]),
    }

    client_2 = {
        "layer.weight": torch.tensor([4.0, 5.0, 6.0]),
        "layer.bias": torch.tensor([20.0]),
    }

    selected = ["layer.weight"]

    # Client-side selective encryption.
    update_1 = create_encrypted_update(
        engine,
        client_1,
        selected,
    )
    update_2 = create_encrypted_update(
        engine,
        client_2,
        selected,
    )

    # Verify selected parameter is encrypted.
    assert "layer.weight" in update_1["encrypted"]
    assert "layer.weight" in update_2["encrypted"]

    # Verify non-selected parameter remains plaintext.
    assert "layer.bias" in update_1["plaintext"]
    assert "layer.bias" in update_2["plaintext"]

    # Simulate Flower transport.
    update_1 = decode_encrypted_update_from_flower(
        encode_encrypted_update_for_flower(update_1)
    )
    update_2 = decode_encrypted_update_from_flower(
        encode_encrypted_update_for_flower(update_2)
    )

    # Server-side encrypted aggregation.
    aggregated = aggregate_encrypted_parameters(
        engine,
        [update_1, update_2],
    )

    # Verify structure.
    assert "layer.weight" in aggregated

    # Decrypt only after aggregation for verification.
    ciphertext = engine.deserialize_ciphertext(
        aggregated["layer.weight"]
    )

    result = engine.decrypt_vector(
        ciphertext,
        original_shape=torch.Size([3]),
    )

    expected = torch.tensor([5.0, 7.0, 9.0])

    assert torch.allclose(result, expected, atol=1e-3)
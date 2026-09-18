"""Encrypted model update helpers for FedMed."""

from typing import Dict, List, Any
import pickle

import numpy as np
import torch

from fedmed.privacy.tenseal_engine import TenSEALEngine


def create_encrypted_update(
    engine: TenSEALEngine,
    state_dict: Dict[str, torch.Tensor],
    encrypted_parameters: List[str],
) -> Dict[str, Any]:
    """Prepare a model update with selected parameters encrypted."""

    prepared = engine.prepare_model_parameters(
        state_dict,
        encrypted_parameters,
    )

    plaintext = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in prepared["plaintext"].items()
    }

    return {
        "encrypted": prepared["encrypted"],
        "plaintext": plaintext,
    }


def serialize_encrypted_update(update: Dict[str, Any]) -> bytes:
    """Serialize an encrypted update for transmission."""

    return pickle.dumps(update)


def deserialize_encrypted_update(payload: bytes) -> Dict[str, Any]:
    """Deserialize an encrypted update received by the server."""

    return pickle.loads(payload)


def aggregate_encrypted_parameters(
    engine: TenSEALEngine,
    encrypted_updates: List[Dict[str, Any]],
) -> Dict[str, bytes]:
    """
    Aggregate selected encrypted parameters without decrypting them.

    Decryption is intentionally not performed during aggregation.
    """

    if not encrypted_updates:
        return {}

    aggregated = {}

    parameter_names = set(encrypted_updates[0]["encrypted"].keys())

    for name in parameter_names:
        ciphertext = None

        for update in encrypted_updates:
            encrypted = update["encrypted"].get(name)

            if encrypted is None:
                raise ValueError(
                    f"Encrypted parameter '{name}' is missing from an update."
                )

            current = engine.deserialize_ciphertext(
                encrypted["ciphertext"]
            )

            if ciphertext is None:
                ciphertext = current
            else:
                ciphertext += current

        aggregated[name] = engine.serialize_ciphertext(ciphertext)

    return aggregated
def encode_encrypted_update_for_flower(
    update: Dict[str, Any],
) -> np.ndarray:
    """Encode an encrypted update as a uint8 NumPy array for Flower."""
    payload = serialize_encrypted_update(update)
    return np.frombuffer(payload, dtype=np.uint8).copy()


def decode_encrypted_update_from_flower(
    payload: np.ndarray,
) -> Dict[str, Any]:
    """Decode an encrypted update received through Flower."""
    if payload.dtype != np.uint8:
        payload = payload.astype(np.uint8)

    return deserialize_encrypted_update(payload.tobytes())
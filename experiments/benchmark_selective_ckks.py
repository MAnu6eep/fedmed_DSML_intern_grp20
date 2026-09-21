"""Benchmark selective CKKS encryption against plaintext aggregation."""

import json
import time
from pathlib import Path

import torch

from fedmed.privacy.encrypted_update import (
    aggregate_encrypted_parameters,
    create_encrypted_update,
    serialize_encrypted_update,
)
from fedmed.privacy.tenseal_engine import TenSEALEngine


NUM_CLIENTS = 3
NUM_VALUES = 100
NUM_REPEATS = 3
ENCRYPTED_PARAMETERS = ["layer.weight"]

OUTPUT_PATH = Path("experiments/outputs/selective_ckks_benchmark.json")


def create_client_update(client_id: int) -> dict[str, torch.Tensor]:
    """Create a deterministic model update for one simulated client."""
    start = float(client_id * 10)

    return {
        "layer.weight": torch.arange(
            start,
            start + NUM_VALUES,
            dtype=torch.float32,
        ),
        "layer.bias": torch.tensor([float(client_id)]),
    }


def plaintext_aggregate(updates):
    """Aggregate model updates using normal plaintext addition."""
    result = {}

    parameter_names = updates[0].keys()

    for name in parameter_names:
        result[name] = sum(
            (update[name] for update in updates),
            torch.zeros_like(updates[0][name]),
        )

    return result



def benchmark_plaintext(updates):
    """Measure normal plaintext aggregation and client payload size."""
    timings = []
    payload_sizes = []

    # Measure the total client -> server plaintext payload.
    plaintext_payload_size = sum(
        sum(tensor.numpy().nbytes for tensor in update.values())
        for update in updates
    )

    for _ in range(NUM_REPEATS):
        start = time.perf_counter()

        aggregated = plaintext_aggregate(updates)

        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        payload_sizes.append(plaintext_payload_size)

    return {
        "aggregation_time_seconds": sum(timings) / len(timings),
        "payload_size_bytes": payload_sizes[0],
        "result": aggregated,
    }




def benchmark_selective_ckks(updates, engine):
    """Measure selective CKKS encryption and encrypted aggregation."""
    encryption_times = []
    aggregation_times = []
    payload_sizes = []

    encrypted_updates = []

    for _ in range(NUM_REPEATS):
        current_updates = []

        encryption_start = time.perf_counter()

        for update in updates:
            encrypted_update = create_encrypted_update(
                engine,
                update,
                ENCRYPTED_PARAMETERS,
            )
            current_updates.append(encrypted_update)

        encryption_elapsed = time.perf_counter() - encryption_start
        encryption_times.append(encryption_elapsed)

        encrypted_updates = current_updates

        payload_size = sum(
            len(serialize_encrypted_update(update))
            for update in encrypted_updates
        )
        payload_sizes.append(payload_size)

        aggregation_start = time.perf_counter()

        aggregated = aggregate_encrypted_parameters(
            engine,
            encrypted_updates,
        )

        aggregation_elapsed = time.perf_counter() - aggregation_start
        aggregation_times.append(aggregation_elapsed)

    return {
        "encryption_time_seconds": (
            sum(encryption_times) / len(encryption_times)
        ),
        "encrypted_aggregation_time_seconds": (
            sum(aggregation_times) / len(aggregation_times)
        ),
        "payload_size_bytes": payload_sizes[0],
        "aggregated": aggregated,
        "encrypted_updates": encrypted_updates,
    }


def verify_correctness(engine, plaintext_result, encrypted_result):
    """Verify encrypted aggregation matches plaintext aggregation."""
    ciphertext = engine.deserialize_ciphertext(
        encrypted_result["layer.weight"]
    )

    decrypted = engine.decrypt_vector(
        ciphertext,
        original_shape=plaintext_result["layer.weight"].shape,
    )

    expected = plaintext_result["layer.weight"]

    max_absolute_error = torch.max(
        torch.abs(decrypted - expected)
    ).item()

    return {
        "passed": bool(torch.allclose(
            decrypted,
            expected,
            atol=1e-3,
        )),
        "max_absolute_error": max_absolute_error,
    }


def main():
    """Run and save the selective CKKS benchmark."""
    updates = [
        create_client_update(client_id)
        for client_id in range(1, NUM_CLIENTS + 1)
    ]

    engine = TenSEALEngine()

    plaintext = benchmark_plaintext(updates)

    selective = benchmark_selective_ckks(
        updates,
        engine,
    )

    correctness = verify_correctness(
        engine,
        plaintext["result"],
        selective["aggregated"],
    )

    plaintext_time = plaintext["aggregation_time_seconds"]
    encryption_time = selective["encryption_time_seconds"]
    encrypted_aggregation_time = (
        selective["encrypted_aggregation_time_seconds"]
    )

    result = {
        "configuration": {
            "num_clients": NUM_CLIENTS,
            "num_values_per_parameter": NUM_VALUES,
            "num_repeats": NUM_REPEATS,
            "encrypted_parameters": ENCRYPTED_PARAMETERS,
        },
        "baseline": {
            "aggregation_time_seconds": plaintext_time,
            "payload_size_bytes": plaintext["payload_size_bytes"],
        },
        "selective_ckks": {
            "encryption_time_seconds": encryption_time,
            "encrypted_aggregation_time_seconds": (
                encrypted_aggregation_time
            ),
            "payload_size_bytes": selective["payload_size_bytes"],
        },
        "overhead": {
            "encryption_time_vs_plaintext_aggregation_ratio": (
                encryption_time / plaintext_time
                if plaintext_time > 0
                else None
            ),
            "encrypted_aggregation_vs_plaintext_ratio": (
                encrypted_aggregation_time / plaintext_time
                if plaintext_time > 0
                else None
            ),
            "payload_size_ratio": (
                selective["payload_size_bytes"]
                / plaintext["payload_size_bytes"]
            ),
        },
        "correctness": correctness,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
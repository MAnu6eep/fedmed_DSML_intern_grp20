"""
Day 4 benchmark:
Compare standard FedAvg with privacy-enabled secure aggregation.

Measures:
- Standard FedAvg aggregation time
- SecAgg+ processing time
- Encryption time
- Serialization time
- Decryption time
- Plaintext communication size
- Encrypted communication size
- Communication overhead
- Total processing overhead

This benchmark uses simulated client model updates so that the
privacy overhead can be measured independently of model training.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

try:
    import tenseal as ts
except ImportError as exc:
    raise ImportError(
        "TenSEAL is required. Activate your project virtual environment "
        "and install TenSEAL before running this benchmark."
    ) from exc


# ============================================================
# Configuration
# ============================================================

NUM_CLIENTS = 5
MODEL_SIZE = 10_000

# Number of benchmark repetitions.
NUM_RUNS = 3

# CKKS parameters.
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
GLOBAL_SCALE = 2**40

OUTPUT_DIR = Path("docs")
OUTPUT_FILE = OUTPUT_DIR / "secagg_benchmark_results.json"


# ============================================================
# CKKS Context
# ============================================================


def create_ckks_context() -> ts.Context:
    """
    Create a CKKS context for encrypted model updates.
    """

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )

    context.global_scale = GLOBAL_SCALE

    # Keep the secret key locally.
    context.generate_galois_keys()

    return context


# ============================================================
# Simulated Client Updates
# ============================================================


def create_client_updates() -> list[torch.Tensor]:
    """
    Generate simulated model updates for multiple hospitals.

    Each tensor represents a flattened model update received
    from one federated-learning client.
    """

    torch.manual_seed(42)

    updates = [
        torch.randn(MODEL_SIZE, dtype=torch.float32)
        for _ in range(NUM_CLIENTS)
    ]

    return updates


# ============================================================
# Standard FedAvg
# ============================================================


def fedavg_aggregate(
    client_updates: list[torch.Tensor],
) -> torch.Tensor:
    """
    Standard plaintext FedAvg aggregation.

    For equal client weights:

        global_update = mean(client_updates)
    """

    stacked = torch.stack(client_updates)

    return torch.mean(stacked, dim=0)


# ============================================================
# Plaintext Serialization
# ============================================================


def serialize_plaintext(
    update: torch.Tensor,
) -> bytes:
    """
    Serialize a plaintext tensor using PyTorch.
    """

    return update.numpy().tobytes()


# ============================================================
# SecAgg+ Style Encryption
# ============================================================


def encrypt_update(
    context: ts.Context,
    update: torch.Tensor,
) -> ts.CKKSVector:
    """
    Encrypt a client model update using CKKS.
    """

    values = update.detach().cpu().double().tolist()

    return ts.ckks_vector(context, values)


# ============================================================
# Encrypted Aggregation
# ============================================================


def aggregate_encrypted(
    encrypted_updates: list[ts.CKKSVector],
) -> ts.CKKSVector:
    """
    Add encrypted client updates.

    CKKS supports homomorphic addition, so the server can combine
    encrypted updates without seeing the plaintext model updates.
    """

    aggregated = encrypted_updates[0]

    for encrypted_update in encrypted_updates[1:]:
        aggregated += encrypted_update

    return aggregated


# ============================================================
# SecAgg+ Benchmark
# ============================================================


def benchmark_secagg(
    client_updates: list[torch.Tensor],
) -> dict:
    """
    Benchmark the privacy-preserving aggregation pipeline.

    The benchmark measures:

        encryption
        serialization
        encrypted aggregation
        decryption
    """

    context = create_ckks_context()

    encryption_times = []
    serialization_times = []
    aggregation_times = []
    decryption_times = []

    encrypted_sizes = []

    # --------------------------------------------------------
    # Repeat benchmark to reduce timing noise.
    # --------------------------------------------------------

    for _ in range(NUM_RUNS):

        encrypted_updates = []

        # ====================================================
        # Encryption
        # ====================================================

        encryption_start = time.perf_counter()

        for update in client_updates:
            encrypted_update = encrypt_update(
                context,
                update,
            )

            encrypted_updates.append(encrypted_update)

        encryption_time = (
            time.perf_counter() - encryption_start
        )

        encryption_times.append(encryption_time)

        # ====================================================
        # Serialization
        # ====================================================

        serialization_start = time.perf_counter()

        serialized_updates = [
            encrypted_update.serialize()
            for encrypted_update in encrypted_updates
        ]

        serialization_time = (
            time.perf_counter() - serialization_start
        )

        serialization_times.append(serialization_time)

        # Measure encrypted communication size.

        encrypted_size = sum(
            len(data)
            for data in serialized_updates
        )

        encrypted_sizes.append(encrypted_size)

        # ====================================================
        # Encrypted Aggregation
        # ====================================================

        aggregation_start = time.perf_counter()

        aggregated_encrypted = aggregate_encrypted(
            encrypted_updates
        )

        aggregation_time = (
            time.perf_counter() - aggregation_start
        )

        aggregation_times.append(aggregation_time)

        # ====================================================
        # Decryption
        # ====================================================

        decryption_start = time.perf_counter()

        decrypted_values = aggregated_encrypted.decrypt()

        # Convert result so that the decryption operation is
        # actually materialized before timing ends.
        _ = len(decrypted_values)

        decryption_time = (
            time.perf_counter() - decryption_start
        )

        decryption_times.append(decryption_time)

    # ========================================================
    # Plaintext communication size
    # ========================================================

    plaintext_size = sum(
        len(serialize_plaintext(update))
        for update in client_updates
    )

    # ========================================================
    # Average measurements
    # ========================================================

    avg_encryption = statistics.mean(encryption_times)
    avg_serialization = statistics.mean(serialization_times)
    avg_aggregation = statistics.mean(aggregation_times)
    avg_decryption = statistics.mean(decryption_times)

    avg_encrypted_size = statistics.mean(encrypted_sizes)

    total_secagg_time = (
        avg_encryption
        + avg_serialization
        + avg_aggregation
        + avg_decryption
    )

    communication_ratio = (
        avg_encrypted_size / plaintext_size
    )

    communication_overhead_percent = (
        (avg_encrypted_size - plaintext_size)
        / plaintext_size
        * 100
    )

    return {
        "encryption_time_seconds": avg_encryption,
        "serialization_time_seconds": avg_serialization,
        "aggregation_time_seconds": avg_aggregation,
        "decryption_time_seconds": avg_decryption,
        "total_time_seconds": total_secagg_time,
        "plaintext_bytes": plaintext_size,
        "encrypted_bytes": avg_encrypted_size,
        "communication_ratio": communication_ratio,
        "communication_overhead_percent": (
            communication_overhead_percent
        ),
    }


# ============================================================
# FedAvg Benchmark
# ============================================================


def benchmark_fedavg(
    client_updates: list[torch.Tensor],
) -> dict:
    """
    Benchmark standard plaintext FedAvg.
    """

    aggregation_times = []

    for _ in range(NUM_RUNS):

        start = time.perf_counter()

        global_update = fedavg_aggregate(
            client_updates
        )

        elapsed = time.perf_counter() - start

        # Make sure the result is materialized.
        _ = global_update.numel()

        aggregation_times.append(elapsed)

    avg_time = statistics.mean(aggregation_times)

    plaintext_size = sum(
        len(serialize_plaintext(update))
        for update in client_updates
    )

    return {
        "aggregation_time_seconds": avg_time,
        "total_time_seconds": avg_time,
        "communication_bytes": plaintext_size,
    }


# ============================================================
# Comparison
# ============================================================


def calculate_comparison(
    fedavg_results: dict,
    secagg_results: dict,
) -> dict:
    """
    Calculate the performance trade-off introduced by SecAgg+.
    """

    fedavg_time = fedavg_results["total_time_seconds"]

    secagg_time = secagg_results["total_time_seconds"]

    if fedavg_time > 0:
        time_ratio = secagg_time / fedavg_time

        time_overhead_percent = (
            (secagg_time - fedavg_time)
            / fedavg_time
            * 100
        )
    else:
        time_ratio = None
        time_overhead_percent = None

    return {
        "time_ratio": time_ratio,
        "time_overhead_percent": time_overhead_percent,
        "communication_ratio": secagg_results[
            "communication_ratio"
        ],
        "communication_overhead_percent": secagg_results[
            "communication_overhead_percent"
        ],
    }


# ============================================================
# Console Output
# ============================================================


def print_results(
    fedavg_results: dict,
    secagg_results: dict,
    comparison: dict,
) -> None:
    """
    Display benchmark results in a readable format.
    """

    print()
    print("=" * 65)
    print("SECAGG+ PERFORMANCE BENCHMARK")
    print("=" * 65)

    print()
    print("Configuration")
    print("-" * 65)

    print(f"Number of clients : {NUM_CLIENTS}")
    print(f"Model size       : {MODEL_SIZE} parameters")
    print(f"Benchmark runs   : {NUM_RUNS}")

    print()
    print("STANDARD FEDAVG")
    print("-" * 65)

    print(
        f"Aggregation time : "
        f"{fedavg_results['aggregation_time_seconds']:.6f} sec"
    )

    print(
        f"Communication    : "
        f"{fedavg_results['communication_bytes']:,} bytes"
    )

    print()
    print("FEDAVG + SECAGG+")
    print("-" * 65)

    print(
        f"Encryption time  : "
        f"{secagg_results['encryption_time_seconds']:.6f} sec"
    )

    print(
        f"Serialization    : "
        f"{secagg_results['serialization_time_seconds']:.6f} sec"
    )

    print(
        f"Aggregation time : "
        f"{secagg_results['aggregation_time_seconds']:.6f} sec"
    )

    print(
        f"Decryption time  : "
        f"{secagg_results['decryption_time_seconds']:.6f} sec"
    )

    print(
        f"Total time       : "
        f"{secagg_results['total_time_seconds']:.6f} sec"
    )

    print(
        f"Encrypted bytes  : "
        f"{secagg_results['encrypted_bytes']:,.0f}"
    )

    print()
    print("PERFORMANCE COMPARISON")
    print("-" * 65)

    print(
        f"Time ratio       : "
        f"{comparison['time_ratio']:.2f}x"
    )

    print(
        f"Time overhead    : "
        f"{comparison['time_overhead_percent']:.2f}%"
    )

    print(
        f"Communication    : "
        f"{comparison['communication_ratio']:.2f}x"
    )

    print(
        f"Communication    : "
        f"{comparison['communication_overhead_percent']:.2f}%"
    )

    print()
    print("=" * 65)


# ============================================================
# Save Results
# ============================================================


def save_results(
    fedavg_results: dict,
    secagg_results: dict,
    comparison: dict,
) -> None:
    """
    Save benchmark results as JSON.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "benchmark": {
            "num_clients": NUM_CLIENTS,
            "model_size": MODEL_SIZE,
            "num_runs": NUM_RUNS,
            "scheme": "CKKS",
            "poly_modulus_degree": POLY_MODULUS_DEGREE,
            "coeff_mod_bit_sizes": COEFF_MOD_BIT_SIZES,
            "global_scale": GLOBAL_SCALE,
        },
        "fedavg": fedavg_results,
        "secagg_plus": secagg_results,
        "comparison": comparison,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print()
    print(f"Results saved to: {OUTPUT_FILE}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    """
    Run the complete benchmark.
    """

    print()
    print("Creating simulated client updates...")

    client_updates = create_client_updates()

    print(
        f"Created {len(client_updates)} client updates "
        f"with {MODEL_SIZE:,} parameters each."
    )

    # --------------------------------------------------------
    # FedAvg
    # --------------------------------------------------------

    print()
    print("Running standard FedAvg benchmark...")

    fedavg_results = benchmark_fedavg(
        client_updates
    )

    # --------------------------------------------------------
    # SecAgg+
    # --------------------------------------------------------

    print("Running SecAgg+ benchmark...")

    secagg_results = benchmark_secagg(
        client_updates
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison = calculate_comparison(
        fedavg_results,
        secagg_results,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_results(
        fedavg_results,
        secagg_results,
        comparison,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        fedavg_results,
        secagg_results,
        comparison,
    )


if __name__ == "__main__":
    main()


"""Initial CKKS performance and overhead benchmark for FedMed."""

import sys
from pathlib import Path
import statistics
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fedmed.privacy.tenseal_engine import TenSEALEngine

VECTOR_SIZES = [100, 1_000, 4_000]
REPEATS = 20
WARMUP_RUNS = 3


def measure_time(operation, repeats=REPEATS):
    """Run warm-up iterations and return average time in milliseconds."""

    # Warm-up runs
    for _ in range(WARMUP_RUNS):
        operation()

    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return statistics.mean(times)

def benchmark_vector_size(engine: TenSEALEngine, vector_size: int):
    """Benchmark CKKS operations for one vector size."""

    print(f"\n{'=' * 60}")
    print(f"Benchmarking vector size: {vector_size}")
    print(f"{'=' * 60}")

    vector_a = torch.randn(vector_size, dtype=torch.float32)
    vector_b = torch.randn(vector_size, dtype=torch.float32)

    # ---------------------------------------------------------
    # Plaintext addition
    # ---------------------------------------------------------
    plaintext_add_time = measure_time(
        lambda: vector_a + vector_b
    )

    # ---------------------------------------------------------
    # Encryption
    # ---------------------------------------------------------
    encryption_time_a = measure_time(
        lambda: engine.encrypt_tensor(vector_a)
    )

    encryption_time_b = measure_time(
        lambda: engine.encrypt_tensor(vector_b)
    )

    encryption_time = (encryption_time_a + encryption_time_b) / 2

    # Create encrypted vectors for later benchmarks
    encrypted_a = engine.encrypt_tensor(vector_a)
    encrypted_b = engine.encrypt_tensor(vector_b)

    # ---------------------------------------------------------
    # Encrypted aggregation / addition
    # ---------------------------------------------------------
    encrypted_add_time = measure_time(
        lambda: encrypted_a + encrypted_b
    )

    encrypted_sum = encrypted_a + encrypted_b

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------
    serialization_time = measure_time(
        lambda: engine.serialize_ciphertext(encrypted_a)
    )

    serialized_ciphertext = engine.serialize_ciphertext(encrypted_a)

    # ---------------------------------------------------------
    # Deserialization
    # ---------------------------------------------------------
    deserialization_time = measure_time(
        lambda: engine.deserialize_ciphertext(serialized_ciphertext)
    )

    # ---------------------------------------------------------
    # Decryption
    # ---------------------------------------------------------
    decryption_time = measure_time(
        lambda: engine.decrypt_vector(encrypted_sum)
    )

    # ---------------------------------------------------------
    # Size overhead
    # ---------------------------------------------------------
    plaintext_size_bytes = vector_a.numel() * vector_a.element_size()
    ciphertext_size_bytes = len(serialized_ciphertext)

    size_overhead = ciphertext_size_bytes / plaintext_size_bytes

    # ---------------------------------------------------------
    # Correctness check
    # ---------------------------------------------------------
    decrypted_sum = engine.decrypt_vector(encrypted_sum)
    expected_sum = vector_a + vector_b

    max_error = torch.max(
        torch.abs(decrypted_sum - expected_sum)
    ).item()

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------
    print(f"Plaintext addition:        {plaintext_add_time:.4f} ms")
    print(f"CKKS encryption:           {encryption_time:.4f} ms")
    print(f"CKKS encrypted addition:   {encrypted_add_time:.4f} ms")
    print(f"Serialization:             {serialization_time:.4f} ms")
    print(f"Deserialization:           {deserialization_time:.4f} ms")
    print(f"Decryption:                {decryption_time:.4f} ms")

    print()
    print(f"Plaintext size:            {plaintext_size_bytes:,} bytes")
    print(f"Ciphertext size:           {ciphertext_size_bytes:,} bytes")
    print(f"Size overhead:             {size_overhead:.2f}x")
    print(f"Maximum decryption error:  {max_error:.8f}")

    return {
        "vector_size": vector_size,
        "plaintext_add_ms": plaintext_add_time,
        "encryption_ms": encryption_time,
        "encrypted_add_ms": encrypted_add_time,
        "serialization_ms": serialization_time,
        "deserialization_ms": deserialization_time,
        "decryption_ms": decryption_time,
        "plaintext_size_bytes": plaintext_size_bytes,
        "ciphertext_size_bytes": ciphertext_size_bytes,
        "size_overhead": size_overhead,
        "max_error": max_error,
    }


def main():
    """Run CKKS performance benchmarks."""

    print("=" * 60)
    print("FedMed CKKS Performance Benchmark")
    print("=" * 60)

    print("\nInitializing TenSEAL CKKS engine...")
    engine = TenSEALEngine()

    results = []

    for vector_size in VECTOR_SIZES:
        result = benchmark_vector_size(engine, vector_size)
        results.append(result)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(
        f"{'Vector':>10} | "
        f"{'Encrypt ms':>12} | "
        f"{'Enc Add ms':>12} | "
        f"{'Decrypt ms':>12} | "
        f"{'Overhead':>10}"
    )

    print("-" * 80)

    for result in results:
        print(
            f"{result['vector_size']:>10} | "
            f"{result['encryption_ms']:>12.4f} | "
            f"{result['encrypted_add_ms']:>12.4f} | "
            f"{result['decryption_ms']:>12.4f} | "
            f"{result['size_overhead']:>9.2f}x"
        )


if __name__ == "__main__":
    main()
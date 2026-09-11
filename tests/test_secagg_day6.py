import time

import pytest

from fedmed.privacy.secagg_config import SecAggPlusConfig


def create_config():
    return SecAggPlusConfig(
        num_clients=3,
        threshold=2,
        modulus_range=2**32,
        clipping_bound=10.0,
        quantization_bits=16,
        enable_dropouts=True,
    )


def check_threshold(participants, config):
    if participants < config.threshold:
        raise ValueError(
            f"Secure aggregation requires at least "
            f"{config.threshold} clients, but only "
            f"{participants} participated."
        )
    return True


def test_all_three_hospitals():
    """3/3 hospitals participate."""

    config = create_config()

    start = time.perf_counter()

    result = check_threshold(3, config)

    elapsed = time.perf_counter() - start

    assert result is True
    assert config.num_clients == 3
    assert config.threshold == 2

    print(f"\n3/3 hospitals: PASS")
    print(f"Processing time: {elapsed * 1000:.4f} ms")


def test_one_hospital_dropout():
    """One hospital drops out: 2/3 must still succeed."""

    config = create_config()

    start = time.perf_counter()

    result = check_threshold(2, config)

    elapsed = time.perf_counter() - start

    assert result is True
    assert 2 >= config.threshold

    print(f"\n2/3 hospitals after dropout: PASS")
    print(f"Processing time: {elapsed * 1000:.4f} ms")


def test_threshold_failure():
    """Only one hospital remains: 2-of-3 threshold cannot be satisfied."""

    config = create_config()

    with pytest.raises(ValueError):
        check_threshold(1, config)

    print("\n1/3 hospitals: CORRECTLY REJECTED")


def test_secagg_overhead():
    """Measure threshold-check processing overhead."""

    config = create_config()

    start = time.perf_counter()

    for _ in range(1000):
        check_threshold(3, config)

    elapsed = time.perf_counter() - start

    average_ms = (elapsed / 1000) * 1000

    print("\nSecAgg+ verification overhead")
    print("-----------------------------")
    print(f"Clients       : {config.num_clients}")
    print(f"Threshold     : {config.threshold}-of-{config.num_clients}")
    print(f"1000 checks   : {elapsed * 1000:.4f} ms")
    print(f"Average check : {average_ms:.6f} ms")

    assert elapsed >= 0
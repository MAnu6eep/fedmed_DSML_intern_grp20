from typing import Dict, List, Sequence
import random


def partition_iid(
    volumes: Sequence[str],
    num_clients: int = 3,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split MRI volumes uniformly across federated clients.

    Each volume is assigned to exactly one client.
    The assignment is randomized using a fixed seed so that
    experiments are reproducible.

    Args:
        volumes: Collection of volume identifiers or paths.
        num_clients: Number of federated clients.
        seed: Random seed for reproducible shuffling.

    Returns:
        Dictionary mapping client IDs to their assigned volumes.

    Raises:
        ValueError: If num_clients is less than 1.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")

    shuffled = list(volumes)
    random.Random(seed).shuffle(shuffled)

    partitions = {
        f"client_{i + 1}": []
        for i in range(num_clients)
    }

    for index, volume in enumerate(shuffled):
        client_id = f"client_{index % num_clients + 1}"
        partitions[client_id].append(volume)

    return partitions


if __name__ == "__main__":
    volumes = [f"volume_{i:02d}" for i in range(10)]

    partitions = partition_iid(
        volumes,
        num_clients=3,
        seed=42,
    )

    for client, client_volumes in partitions.items():
        print(f"{client}: {client_volumes}")
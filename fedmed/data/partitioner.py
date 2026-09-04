from typing import Dict, List, Sequence
import random
import numpy as np

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


def partition_dirichlet(
    volumes: Sequence[str],
    labels: Sequence[int],
    num_clients: int = 3,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split MRI volumes across federated clients using
    a Dirichlet-based non-IID distribution.

    Args:
        volumes: Collection of volume identifiers or paths.
        labels: Class label corresponding to each volume.
        num_clients: Number of federated hospital clients.
        alpha: Dirichlet concentration parameter.
               Smaller alpha -> more heterogeneous distributions.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping client IDs to assigned volumes.
    """

    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")

    if alpha <= 0:
        raise ValueError("alpha must be greater than 0.")

    if len(volumes) != len(labels):
        raise ValueError("volumes and labels must have the same length.")

    if len(volumes) == 0:
        return {
            f"client_{i + 1}": []
            for i in range(num_clients)
        }

    rng = np.random.default_rng(seed)

    volumes = list(volumes)
    labels = np.asarray(labels)

    unique_classes = np.unique(labels)

    partitions = {
        f"client_{i + 1}": []
        for i in range(num_clients)
    }

    for class_id in unique_classes:

        # Find all samples belonging to this class
        class_indices = np.where(labels == class_id)[0]

        # Shuffle samples of this class
        rng.shuffle(class_indices)

        # Generate Dirichlet proportions
        proportions = rng.dirichlet(
            np.repeat(alpha, num_clients)
        )

        # Convert proportions into sample counts
        counts = (proportions * len(class_indices)).astype(int)

        # Distribute remaining samples
        remainder = len(class_indices) - counts.sum()

        for i in range(remainder):
            counts[i % num_clients] += 1

        start = 0

        for client_idx, count in enumerate(counts):

            selected_indices = class_indices[
                start:start + count
            ]

            client_id = f"client_{client_idx + 1}"

            for idx in selected_indices:
                partitions[client_id].append(
                    volumes[idx]
                )

            start += count

    # Shuffle each client's final dataset
    for client_id in partitions:
        rng.shuffle(partitions[client_id])

    return partitions


def partition_stats(partitions: Dict[str, List[str]]) -> None:
    """Print the number of volumes assigned to each client."""
    print("\n===== PARTITION STATISTICS =====")

    total = 0
    for client_id, client_volumes in partitions.items():
        count = len(client_volumes)
        total += count
        print(f"{client_id}: {count} volumes")

    print(f"Total volumes: {total}")



if __name__ == "__main__":

    volumes = [
        f"volume_{i:02d}"
        for i in range(12)
    ]

    labels = [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2
    ]

    print("\n===== IID PARTITION =====")

    iid_partitions = partition_iid(
        volumes,
        num_clients=3,
        seed=42
    )

    for client, client_volumes in iid_partitions.items():
        print(
            f"{client}: "
            f"{len(client_volumes)} samples"
        )

    partition_stats(iid_partitions)

    print("\n===== DIRICHLET NON-IID PARTITION =====")

    dirichlet_partitions = partition_dirichlet(
        volumes,
        labels,
        num_clients=3,
        alpha=0.5,
        seed=42
    )

    for client, client_volumes in dirichlet_partitions.items():
        print(
            f"{client}: "
            f"{len(client_volumes)} samples -> "
            f"{client_volumes}"
        )
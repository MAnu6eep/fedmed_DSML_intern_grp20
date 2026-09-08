"""
fedmed/resilience/dropout.py

Controlled hospital dropout simulation and availability detection.
"""

from typing import Dict, List

from fedmed.nodes.node_manager import check_node_health
from fedmed.nodes.registry import registry


def simulate_dropout(hospital_id: str) -> bool:
    """
    Simulate a hospital becoming unavailable.

    Returns True when the hospital exists and its state was changed
    to offline.
    """
    node = registry.get_node(hospital_id)

    if node is None:
        return False

    return registry.update_status(hospital_id, "offline")


def detect_dropouts() -> Dict[str, bool]:
    """
    Check all registered hospitals and update their availability status.

    Returns a mapping of hospital IDs to their current health state.
    """
    results: Dict[str, bool] = {}

    for node in registry.list_nodes():
        healthy = check_node_health(node.host, node.port)

        registry.update_status(
            node.id,
            "online" if healthy else "offline",
        )

        results[node.id] = healthy

    return results


def get_active_hospitals() -> List[str]:
    """
    Return only hospitals currently marked as online.
    """
    return [
        node.id
        for node in registry.list_nodes()
        if node.status == "online"
    ]


def get_active_hospital_count() -> int:
    """Return the number of hospitals currently available for training."""
    return len(get_active_hospitals())


def get_available_hospitals() -> List[str]:
    """
    Detect hospital availability and return currently active hospitals.

    Offline hospitals are excluded from the current participation set.
    """
    detect_dropouts()
    return get_active_hospitals()


def recover_hospital(hospital_id: str) -> bool:
    """
    Mark an offline hospital as online after it becomes reachable again.

    Returns True when the hospital exists, is reachable, and its status
    was restored to online.
    """
    node = registry.get_node(hospital_id)

    if node is None:
        return False

    if node.status == "online":
        return True

    if not check_node_health(node.host, node.port):
        return False

    return registry.update_status(hospital_id, "online")


def detect_recoveries() -> Dict[str, bool]:
    """
    Detect previously offline hospitals that have become reachable again.

    Returns a mapping of hospital IDs to their current health state.
    Offline hospitals are restored to online when their health endpoint
    becomes reachable.
    """
    results: Dict[str, bool] = {}

    for node in registry.list_nodes():
        healthy = check_node_health(node.host, node.port)

        if healthy and node.status == "offline":
            registry.update_status(node.id, "online")

        results[node.id] = healthy

    return results

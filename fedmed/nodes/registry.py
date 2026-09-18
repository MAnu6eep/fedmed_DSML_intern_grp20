"""fedmed/nodes/registry.py

Runtime registry and state manager for active hospital nodes.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HospitalNode(BaseModel):
    """Represents the runtime state and connection details of a hospital node."""

    id: str
    name: str
    host: str = "127.0.0.1"
    port: int
    grpc_port: int
    data_dir: str
    sample_count: int = 0
    status: str = "offline"  # online, training, offline, error
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)

    last_heartbeat_latency: Optional[float] = None
    heartbeat_successes: int = 0
    heartbeat_failures: int = 0
    consecutive_failures: int = 0
    participating: bool = False


class NodeRegistry:
    """Central in-memory registry for dynamically managed hospital nodes."""

    def __init__(self):
        self._nodes: Dict[str, HospitalNode] = {}

    def add_node(self, node: HospitalNode) -> bool:
        """Add a new hospital node to the registry.

        Returns:
            True if the node was added successfully.
            False if a node with the same ID already exists.
        """
        if node.id in self._nodes:
            return False

        node.last_heartbeat = datetime.utcnow()
        self._nodes[node.id] = node
        return True

    def register(self, node: HospitalNode) -> None:
        """Register or replace a hospital node.

        Kept for backward compatibility with the existing node startup flow.
        """
        node.last_heartbeat = datetime.utcnow()
        self._nodes[node.id] = node

    def update_node(self, node_id: str, **updates) -> bool:
        """Update mutable fields of an existing hospital node.

        Returns:
            True if the node was updated successfully.
            False if the node does not exist.
        """
        node = self._nodes.get(node_id)

        if node is None:
            return False

        allowed_fields = {
            "name",
            "host",
            "port",
            "grpc_port",
            "data_dir",
            "sample_count",
            "status",
        }

        for field, value in updates.items():
            if field not in allowed_fields:
                raise ValueError(f"Unsupported node field: {field}")

            setattr(node, field, value)

        node.last_heartbeat = datetime.utcnow()
        return True

    def update_status(
        self,
        node_id: str,
        status: str,
        sample_count: Optional[int] = None,
    ) -> bool:
        """Update health status and sample count for a registered node."""
        node = self._nodes.get(node_id)

        if node is None:
            return False

        node.status = status
        node.last_heartbeat = datetime.utcnow()
        node.participating = status in {"online", "training"}

        if sample_count is not None:
            node.sample_count = sample_count

        return True

    def record_heartbeat(
        self,
        node_id: str,
        healthy: bool,
        latency: float,
        failure_threshold: int = 3,
    ) -> bool:
        """Record a heartbeat result and synchronize node availability."""
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be greater than zero")

        node = self._nodes.get(node_id)

        if node is None:
            return False

        node.last_heartbeat = datetime.utcnow()
        node.last_heartbeat_latency = max(0.0, float(latency))

        if healthy:
            node.heartbeat_successes += 1
            node.consecutive_failures = 0

            if node.status in {"offline", "error"}:
                node.status = "online"

            node.participating = node.status in {"online", "training"}

        else:
            node.heartbeat_failures += 1
            node.consecutive_failures += 1

            if node.consecutive_failures >= failure_threshold:
                node.status = "offline"
                node.participating = False

        return healthy

    def get_live_node_info(self) -> List[Dict[str, object]]:
        """Return dashboard-ready live information for all registered nodes."""
        return [
            {
                "hospital_id": node.id,
                "name": node.name,
                "host": node.host,
                "port": node.port,
                "grpc_port": node.grpc_port,
                "status": node.status,
                "active": node.status in {"online", "training"},
                "participating": node.participating,
                "samples": node.sample_count,
                "heartbeat_successes": node.heartbeat_successes,
                "heartbeat_failures": node.heartbeat_failures,
                "consecutive_failures": node.consecutive_failures,
                "last_heartbeat": node.last_heartbeat.isoformat(),
                "latency_ms": (
                    round(node.last_heartbeat_latency * 1000, 3)
                    if node.last_heartbeat_latency is not None
                    else None
                ),
            }
            for node in self._nodes.values()
        ]

    def get_node(self, node_id: str) -> Optional[HospitalNode]:
        """Retrieve a hospital node by ID."""
        return self._nodes.get(node_id)

    def list_nodes(self) -> List[HospitalNode]:
        """Return all currently registered hospital nodes."""
        return list(self._nodes.values())

    def remove_node(self, node_id: str) -> bool:
        """Remove a hospital node from the registry.

        Returns:
            True if the node was removed.
            False if the node was not registered.
        """
        if node_id not in self._nodes:
            return False

        del self._nodes[node_id]
        return True

    def get_active_count(self) -> int:
        """Return the number of hospitals currently online."""
        return sum(
            1
            for node in self._nodes.values()
            if node.status == "online"
        )


# Global singleton instance
registry = NodeRegistry()

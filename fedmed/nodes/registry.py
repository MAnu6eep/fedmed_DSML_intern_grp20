"""fedmed/nodes/registry.py
Runtime registry and state manager for active hospital nodes.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HospitalNode(BaseModel):
    id: str
    name: str
    host: str = "127.0.0.1"
    port: int
    grpc_port: int
    data_dir: str
    sample_count: int = 0
    status: str = "offline"  # online, training, offline, error
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)


class NodeRegistry:
    """In-memory registry tracking connected hospital nodes."""

    def __init__(self):
        self._nodes: Dict[str, HospitalNode] = {}

    def register(self, node: HospitalNode) -> None:
        """Registers or updates a hospital node."""
        node.last_heartbeat = datetime.utcnow()
        self._nodes[node.id] = node

    def update_status(
        self,
        node_id: str,
        status: str,
        sample_count: Optional[int] = None,
    ) -> bool:
        """Updates health status and sample count for a registered node."""
        if node_id in self._nodes:
            self._nodes[node_id].status = status
            self._nodes[node_id].last_heartbeat = datetime.utcnow()

            if sample_count is not None:
                self._nodes[node_id].sample_count = sample_count

            return True

        return False

    def get_node(self, node_id: str) -> Optional[HospitalNode]:
        return self._nodes.get(node_id)

    def list_nodes(self) -> List[HospitalNode]:
        return list(self._nodes.values())

    def get_active_count(self) -> int:
        return sum(
            1 for node in self._nodes.values() if node.status != "offline"
        )


# Global singleton instance
registry = NodeRegistry()

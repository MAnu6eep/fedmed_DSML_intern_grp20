"""Heartbeat monitoring for dynamically registered FedMed hospital nodes."""

from __future__ import annotations

from datetime import datetime
import threading
import time
from typing import Dict, Optional

from fedmed.nodes.node_manager import check_node_health
from fedmed.nodes.registry import NodeRegistry, registry


DEFAULT_INTERVAL = 5.0
DEFAULT_FAILURE_THRESHOLD = 3


class HeartbeatMonitor:
    """Continuously monitor registered hospital nodes."""

    def __init__(
        self,
        node_registry: NodeRegistry = registry,
        interval: float = DEFAULT_INTERVAL,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be greater than zero")

        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be greater than zero")

        self.registry = node_registry
        self.interval = interval
        self.failure_threshold = failure_threshold

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def check_node(self, node_id: str) -> bool:
        """Perform one heartbeat check for a registered node."""
        node = self.registry.get_node(node_id)

        if node is None:
            return False

        start_time = time.perf_counter()
        healthy = check_node_health(
            node.health_host,
            node.health_port,
        )
        latency = time.perf_counter() - start_time

        return self.registry.record_heartbeat(
            node_id=node_id,
            healthy=healthy,
            latency=latency,
            failure_threshold=self.failure_threshold,
        )

    def check_all(self) -> Dict[str, bool]:
        """Perform one heartbeat check for every registered node."""
        results: Dict[str, bool] = {}

        for node in self.registry.list_nodes():
            results[node.id] = self.check_node(node.id)

        return results

    def _run(self) -> None:
        """Run heartbeat checks continuously until stopped."""
        while not self._stop_event.is_set():
            self.check_all()
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        """Start background heartbeat monitoring."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="fedmed-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop background heartbeat monitoring."""
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1)

        self._thread = None


heartbeat_monitor = HeartbeatMonitor()
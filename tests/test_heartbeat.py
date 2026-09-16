from fedmed.nodes.heartbeat import HeartbeatMonitor
from fedmed.nodes.registry import HospitalNode, NodeRegistry
import time

def make_node(node_id: str, port: int = 8081) -> HospitalNode:
    return HospitalNode(
        id=node_id,
        name=f"Hospital {node_id}",
        host="127.0.0.1",
        port=port,
        grpc_port=port + 1000,
        data_dir=f"/data/{node_id}",
    )


def test_successful_heartbeat_records_latency_and_success(monkeypatch):
    node_registry = NodeRegistry()
    node_registry.add_node(make_node("hospital_a"))

    monkeypatch.setattr(
        "fedmed.nodes.heartbeat.check_node_health",
        lambda host, port: True,
    )

    monitor = HeartbeatMonitor(node_registry)

    assert monitor.check_node("hospital_a") is True

    node = node_registry.get_node("hospital_a")

    assert node is not None
    assert node.heartbeat_successes == 1
    assert node.heartbeat_failures == 0
    assert node.consecutive_failures == 0
    assert node.last_heartbeat_latency is not None
    assert node.last_heartbeat_latency >= 0


def test_temporary_failure_does_not_immediately_mark_node_offline(monkeypatch):
    node_registry = NodeRegistry()
    node = make_node("hospital_a")
    node.status = "online"
    node_registry.add_node(node)

    monkeypatch.setattr(
        "fedmed.nodes.heartbeat.check_node_health",
        lambda host, port: False,
    )

    monitor = HeartbeatMonitor(
        node_registry,
        failure_threshold=3,
    )

    assert monitor.check_node("hospital_a") is False

    updated = node_registry.get_node("hospital_a")

    assert updated is not None
    assert updated.heartbeat_failures == 1
    assert updated.consecutive_failures == 1
    assert updated.status == "online"


def test_node_becomes_offline_after_consecutive_failures(monkeypatch):
    node_registry = NodeRegistry()
    node = make_node("hospital_a")
    node.status = "online"
    node_registry.add_node(node)

    monkeypatch.setattr(
        "fedmed.nodes.heartbeat.check_node_health",
        lambda host, port: False,
    )

    monitor = HeartbeatMonitor(
        node_registry,
        failure_threshold=3,
    )

    for _ in range(3):
        monitor.check_node("hospital_a")

    updated = node_registry.get_node("hospital_a")

    assert updated is not None
    assert updated.heartbeat_failures == 3
    assert updated.consecutive_failures == 3
    assert updated.status == "offline"


def test_successful_heartbeat_resets_consecutive_failures(monkeypatch):
    node_registry = NodeRegistry()
    node = make_node("hospital_a")
    node.status = "online"
    node_registry.add_node(node)

    responses = iter([False, False, True])

    monkeypatch.setattr(
        "fedmed.nodes.heartbeat.check_node_health",
        lambda host, port: next(responses),
    )

    monitor = HeartbeatMonitor(
        node_registry,
        failure_threshold=3,
    )

    monitor.check_node("hospital_a")
    monitor.check_node("hospital_a")
    monitor.check_node("hospital_a")

    updated = node_registry.get_node("hospital_a")

    assert updated is not None
    assert updated.heartbeat_failures == 2
    assert updated.heartbeat_successes == 1
    assert updated.consecutive_failures == 0
    assert updated.status == "online"


def test_check_all_monitors_dynamic_nodes(monkeypatch):
    node_registry = NodeRegistry()

    node_registry.add_node(make_node("hospital_a", 8081))
    node_registry.add_node(make_node("hospital_b", 8082))
    node_registry.add_node(make_node("hospital_c", 8083))

    monkeypatch.setattr(
        "fedmed.nodes.heartbeat.check_node_health",
        lambda host, port: port != 8082,
    )

    monitor = HeartbeatMonitor(node_registry)

    results = monitor.check_all()

    assert results == {
        "hospital_a": True,
        "hospital_b": False,
        "hospital_c": True,
    }

def test_heartbeat_monitor_starts_and_stops(monkeypatch):
    node_registry = NodeRegistry()
    node_registry.add_node(make_node("hospital_a"))

    calls = []

    monkeypatch.setattr(
        "fedmed.nodes.heartbeat.check_node_health",
        lambda host, port: calls.append((host, port)) or True,
    )

    monitor = HeartbeatMonitor(
        node_registry,
        interval=0.05,
    )

    monitor.start()
    time.sleep(0.15)
    monitor.stop()

    assert calls
    assert monitor._thread is None


def test_heartbeat_monitor_start_is_idempotent(monkeypatch):
    node_registry = NodeRegistry()
    node_registry.add_node(make_node("hospital_a"))

    monkeypatch.setattr(
        "fedmed.nodes.heartbeat.check_node_health",
        lambda host, port: True,
    )

    monitor = HeartbeatMonitor(
        node_registry,
        interval=0.05,
    )

    monitor.start()
    first_thread = monitor._thread

    monitor.start()
    second_thread = monitor._thread

    monitor.stop()

    assert first_thread is second_thread
from fedmed.nodes.node_manager import (
    check_node_health,
    start_node,
    verify_registered_nodes,
)
from fedmed.nodes.registry import registry


def test_all_hospitals_register_with_expected_configuration():
    """All three hospitals should register with their configured ports."""
    registry._nodes.clear()

    for hospital_id in ("hospital_a", "hospital_b", "hospital_c"):
        start_node(hospital_id)

    nodes = {
        node.id: node
        for node in registry.list_nodes()
    }

    assert set(nodes) == {"hospital_a", "hospital_b", "hospital_c"}

    assert nodes["hospital_a"].port == 8081
    assert nodes["hospital_a"].grpc_port == 9091

    assert nodes["hospital_b"].port == 8082
    assert nodes["hospital_b"].grpc_port == 9092

    assert nodes["hospital_c"].port == 8083
    assert nodes["hospital_c"].grpc_port == 9093


def test_unavailable_node_is_detected():
    """An unreachable hospital health endpoint should return False."""
    assert check_node_health("127.0.0.1", 65535) is False


def test_one_unavailable_node_does_not_block_others(monkeypatch):
    """Available hospitals remain detectable when one hospital is offline."""
    registry._nodes.clear()

    for hospital_id in ("hospital_a", "hospital_b", "hospital_c"):
        start_node(hospital_id)

    health = {
        "hospital_a": True,
        "hospital_b": False,
        "hospital_c": True,
    }

    def fake_check_node_health(host, port, timeout=2.0):
        if port == 8081:
            return health["hospital_a"]
        if port == 8082:
            return health["hospital_b"]
        if port == 8083:
            return health["hospital_c"]
        return False

    monkeypatch.setattr(
        "fedmed.nodes.node_manager.check_node_health",
        fake_check_node_health,
    )

    results = verify_registered_nodes()

    assert results == {
        "hospital_a": True,
        "hospital_b": False,
        "hospital_c": True,
    }

    assert registry.get_node("hospital_a").status == "online"
    assert registry.get_node("hospital_b").status == "offline"
    assert registry.get_node("hospital_c").status == "online"

    assert registry.get_active_count() == 2
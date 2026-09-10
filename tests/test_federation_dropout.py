from fedmed.resilience.dropout import get_available_hospitals
from fedmed.nodes.node_manager import start_node
from fedmed.nodes.registry import registry


def test_federation_excludes_dropped_hospital(monkeypatch):
    """The federation layer must exclude an unavailable hospital."""
    registry._nodes.clear()

    for hospital_id in (
        "hospital_a",
        "hospital_b",
        "hospital_c",
    ):
        start_node(hospital_id)

    def fake_check_node_health(host, port, timeout=2.0):
        return port != 8082

    monkeypatch.setattr(
        "fedmed.resilience.dropout.check_node_health",
        fake_check_node_health,
    )

    active = get_available_hospitals()

    assert active == [
        "hospital_a",
        "hospital_c",
    ]

    assert registry.get_node("hospital_b").status == "offline"
    assert registry.get_active_count() == 2

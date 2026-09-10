from fedmed.nodes.node_manager import start_node
from fedmed.nodes.registry import registry
from fedmed.resilience.dropout import (
    detect_dropouts,
    detect_recoveries,
    get_active_hospitals,
)


def test_secure_federation_dropout_and_recovery_sequence(monkeypatch):
    """
    Validate the node-level secure-federation failure sequence:

    all hospitals available
        -> hospital_b unavailable
        -> remaining hospitals stay active
        -> hospital_b recovers
        -> all hospitals rejoin
    """
    registry._nodes.clear()

    for hospital_id in (
        "hospital_a",
        "hospital_b",
        "hospital_c",
    ):
        start_node(hospital_id)

    health = {
        8081: True,
        8082: True,
        8083: True,
    }

    def fake_check_node_health(host, port, timeout=2.0):
        return health.get(port, False)

    monkeypatch.setattr(
        "fedmed.resilience.dropout.check_node_health",
        fake_check_node_health,
    )

    # Initial secure federation availability.
    assert get_active_hospitals() == [
        "hospital_a",
        "hospital_b",
        "hospital_c",
    ]

    # Hospital B drops during federation.
    health[8082] = False
    dropout_result = detect_dropouts()

    assert dropout_result == {
        "hospital_a": True,
        "hospital_b": False,
        "hospital_c": True,
    }

    assert registry.get_node("hospital_b").status == "offline"
    assert get_active_hospitals() == [
        "hospital_a",
        "hospital_c",
    ]

    # Hospital B reconnects.
    health[8082] = True
    recovery_result = detect_recoveries()

    assert recovery_result == {
        "hospital_a": True,
        "hospital_b": True,
        "hospital_c": True,
    }

    assert registry.get_node("hospital_b").status == "online"
    assert get_active_hospitals() == [
        "hospital_a",
        "hospital_b",
        "hospital_c",
    ]

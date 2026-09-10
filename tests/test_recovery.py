from fedmed.nodes.node_manager import start_node
from fedmed.nodes.registry import registry
from fedmed.resilience.dropout import (
    detect_dropouts,
    detect_recoveries,
    get_active_hospitals,
    recover_hospital,
    simulate_dropout,
)


def setup_three_hospitals():
    registry._nodes.clear()

    for hospital_id in (
        "hospital_a",
        "hospital_b",
        "hospital_c",
    ):
        start_node(hospital_id)


def test_offline_hospital_recovers_when_health_endpoint_returns(monkeypatch):
    """An offline hospital should return to online when it is reachable."""
    setup_three_hospitals()

    simulate_dropout("hospital_b")

    assert registry.get_node("hospital_b").status == "offline"
    assert registry.get_active_count() == 2

    monkeypatch.setattr(
        "fedmed.resilience.dropout.check_node_health",
        lambda host, port, timeout=2.0: True,
    )

    results = detect_recoveries()

    assert results == {
        "hospital_a": True,
        "hospital_b": True,
        "hospital_c": True,
    }

    assert registry.get_node("hospital_b").status == "online"
    assert registry.get_active_count() == 3
    assert get_active_hospitals() == [
        "hospital_a",
        "hospital_b",
        "hospital_c",
    ]


def test_recovery_requires_hospital_to_be_reachable(monkeypatch):
    """A dropped hospital stays offline until it becomes reachable."""
    setup_three_hospitals()

    simulate_dropout("hospital_b")

    monkeypatch.setattr(
        "fedmed.resilience.dropout.check_node_health",
        lambda host, port, timeout=2.0: port != 8082,
    )

    assert recover_hospital("hospital_b") is False
    assert registry.get_node("hospital_b").status == "offline"
    assert registry.get_active_count() == 2


def test_recovery_rejoins_after_dropout_without_changing_other_nodes(monkeypatch):
    """The complete dropout -> recovery -> rejoin sequence is preserved."""
    setup_three_hospitals()

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

    initial_nodes = get_active_hospitals()
    assert initial_nodes == [
        "hospital_a",
        "hospital_b",
        "hospital_c",
    ]

    # Hospital B drops out.
    health[8082] = False
    detect_dropouts()

    assert registry.get_node("hospital_b").status == "offline"
    assert get_active_hospitals() == [
        "hospital_a",
        "hospital_c",
    ]

    # Hospital B becomes reachable again.
    health[8082] = True
    assert recover_hospital("hospital_b") is True

    assert registry.get_node("hospital_b").status == "online"
    assert get_active_hospitals() == [
        "hospital_a",
        "hospital_b",
        "hospital_c",
    ]
    assert registry.get_active_count() == 3

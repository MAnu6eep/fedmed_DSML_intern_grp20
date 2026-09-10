from fedmed.nodes.node_manager import start_node
from fedmed.nodes.registry import registry
from fedmed.resilience.dropout import (
    detect_dropouts,
    get_active_hospitals,
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


def test_simulate_dropout_marks_hospital_offline():
    """A controlled dropout should mark the selected hospital offline."""
    setup_three_hospitals()

    assert simulate_dropout("hospital_b") is True

    assert registry.get_node("hospital_a").status == "online"
    assert registry.get_node("hospital_b").status == "offline"
    assert registry.get_node("hospital_c").status == "online"

    assert registry.get_active_count() == 2
    assert get_active_hospitals() == [
        "hospital_a",
        "hospital_c",
    ]


def test_simulate_dropout_unknown_hospital():
    """Unknown hospitals should not modify the registry."""
    setup_three_hospitals()

    assert simulate_dropout("hospital_x") is False
    assert registry.get_active_count() == 3


def test_dropout_detection_during_federated_training(monkeypatch):
    """
    A hospital becoming unavailable during a training round
    must be detected and removed from active participation.
    """
    setup_three_hospitals()

    health = {
        "hospital_a": True,
        "hospital_b": True,
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
        "fedmed.resilience.dropout.check_node_health",
        fake_check_node_health,
    )

    # Before the simulated training interruption,
    # all hospitals are available.
    initial = detect_dropouts()

    assert initial == {
        "hospital_a": True,
        "hospital_b": True,
        "hospital_c": True,
    }

    assert get_active_hospitals() == [
        "hospital_a",
        "hospital_b",
        "hospital_c",
    ]

    # Simulate hospital B failing during the round.
    health["hospital_b"] = False

    after_dropout = detect_dropouts()

    assert after_dropout == {
        "hospital_a": True,
        "hospital_b": False,
        "hospital_c": True,
    }

    assert registry.get_node("hospital_b").status == "offline"
    assert get_active_hospitals() == [
        "hospital_a",
        "hospital_c",
    ]
    assert registry.get_active_count() == 2

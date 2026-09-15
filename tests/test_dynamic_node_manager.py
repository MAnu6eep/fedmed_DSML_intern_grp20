from fedmed.nodes.registry import HospitalNode, NodeRegistry


def make_node(
    node_id: str,
    port: int = 8081,
    grpc_port: int = 9091,
    sample_count: int = 10,
    status: str = "offline",
) -> HospitalNode:
    return HospitalNode(
        id=node_id,
        name=f"Hospital {node_id}",
        host="127.0.0.1",
        port=port,
        grpc_port=grpc_port,
        data_dir=f"/data/{node_id}",
        sample_count=sample_count,
        status=status,
    )


def test_add_and_retrieve_node():
    registry = NodeRegistry()
    node = make_node("hospital_x")

    assert registry.add_node(node) is True
    assert registry.get_node("hospital_x") == node


def test_duplicate_node_is_rejected():
    registry = NodeRegistry()
    node = make_node("hospital_x")

    assert registry.add_node(node) is True
    assert registry.add_node(node) is False
    assert len(registry.list_nodes()) == 1


def test_update_node():
    registry = NodeRegistry()
    registry.add_node(make_node("hospital_x"))

    assert registry.update_node(
        "hospital_x",
        host="192.168.1.20",
        port=8088,
        grpc_port=9098,
        sample_count=250,
        status="online",
    ) is True

    node = registry.get_node("hospital_x")

    assert node is not None
    assert node.host == "192.168.1.20"
    assert node.port == 8088
    assert node.grpc_port == 9098
    assert node.sample_count == 250
    assert node.status == "online"


def test_update_unknown_node_fails():
    registry = NodeRegistry()

    assert registry.update_node(
        "missing_hospital",
        status="online",
    ) is False


def test_remove_node():
    registry = NodeRegistry()
    registry.add_node(make_node("hospital_x"))

    assert registry.remove_node("hospital_x") is True
    assert registry.get_node("hospital_x") is None
    assert registry.list_nodes() == []


def test_remove_unknown_node_fails():
    registry = NodeRegistry()

    assert registry.remove_node("missing_hospital") is False


def test_registry_supports_dynamic_number_of_nodes():
    registry = NodeRegistry()

    for index in range(1, 6):
        registry.add_node(
            make_node(
                f"hospital_{index}",
                port=8000 + index,
                grpc_port=9000 + index,
                sample_count=index * 100,
            )
        )

    assert len(registry.list_nodes()) == 5


def test_active_count_tracks_online_nodes():
    registry = NodeRegistry()

    registry.add_node(make_node("hospital_a", status="online"))
    registry.add_node(make_node("hospital_b", status="offline"))
    registry.add_node(make_node("hospital_c", status="online"))

    assert registry.get_active_count() == 2
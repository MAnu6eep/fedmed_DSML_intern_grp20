"""
tests.test_phase4_docker_hospital_environment
================================================
Unit and Integration tests verifying Phase 4 Docker Hospital Environment:
- Unique hospital identities & independent configurations
- Storage & Data privacy isolation across containers
- Network connectivity & hospital health probes
- Scalability across 1, 3, and 5 hospital nodes
"""

import os
from pathlib import Path
import pytest
import yaml

from fedmed.nodes.node_manager import (
    check_node_health,
    load_hospital_config,
    start_node,
    verify_registered_nodes,
)
from fedmed.nodes.registry import registry


def test_unique_hospital_identities_and_configurations():
    """Verify that all 5 hospital nodes load unique configurations and network ports."""
    registry._nodes.clear()

    hospitals = ("hospital_a", "hospital_b", "hospital_c", "hospital_d", "hospital_e")
    expected_ports = {
        "hospital_a": (8081, 9091),
        "hospital_b": (8082, 9092),
        "hospital_c": (8083, 9093),
        "hospital_d": (8084, 9094),
        "hospital_e": (8085, 9095),
    }

    for hosp_id in hospitals:
        node = start_node(hosp_id)
        assert node.id == hosp_id
        assert node.port == expected_ports[hosp_id][0]
        assert node.grpc_port == expected_ports[hosp_id][1]
        assert hosp_id in node.data_dir

    registered_nodes = {n.id for n in registry.list_nodes()}
    assert registered_nodes == set(hospitals)


def test_storage_and_privacy_isolation_across_hospitals():
    """Verify that each hospital has its own isolated dataset directory and cannot cross-access others."""
    base_path = Path(__file__).resolve().parent.parent / "hospitals"

    hospitals = ("hospital_a", "hospital_b", "hospital_c", "hospital_d", "hospital_e")
    dataset_files = {}

    for hosp in hospitals:
        hosp_data_dir = base_path / hosp / "data" / "imagesTr"
        assert hosp_data_dir.exists(), f"Hospital data directory missing for {hosp}"

        files = sorted([f.name for f in hosp_data_dir.glob("*.nii.gz")])
        assert len(files) > 0, f"No MRI volumes found for {hosp}"
        dataset_files[hosp] = files

    # Verify strict dataset file isolation
    for hosp_i in hospitals:
        for hosp_j in hospitals:
            if hosp_i != hosp_j:
                # File names contain hospital identity prefix
                assert not any(f.startswith(hosp_j.upper()) for f in dataset_files[hosp_i]), (
                    f"{hosp_i} contains files belonging to {hosp_j}"
                )


def test_hospital_health_and_status_probe(monkeypatch):
    """Verify health reporting for online and offline hospitals."""
    registry._nodes.clear()

    for hosp_id in ("hospital_a", "hospital_b", "hospital_c"):
        start_node(hosp_id)

    health_map = {"hospital_a": True, "hospital_b": False, "hospital_c": True}

    def mock_check_health(host, port, timeout=2.0):
        port_to_id = {8081: "hospital_a", 8082: "hospital_b", 8083: "hospital_c"}
        return health_map.get(port_to_id.get(port), False)

    monkeypatch.setattr("fedmed.nodes.node_manager.check_node_health", mock_check_health)

    results = verify_registered_nodes()
    assert results == {"hospital_a": True, "hospital_b": False, "hospital_c": True}
    assert registry.get_active_count() == 2


def test_scalability_1_to_5_hospitals():
    """Verify that hospital registration dynamically scales from 1 to 5 nodes without error."""
    registry._nodes.clear()

    all_hospitals = ["hospital_a", "hospital_b", "hospital_c", "hospital_d", "hospital_e"]

    # Test 1 hospital
    start_node(all_hospitals[0])
    assert registry.get_active_count() == 1

    # Scale to 3 hospitals
    for hosp in all_hospitals[1:3]:
        start_node(hosp)
    assert len(registry.list_nodes()) == 3

    # Scale to 5 hospitals
    for hosp in all_hospitals[3:]:
        start_node(hosp)
    assert len(registry.list_nodes()) == 5

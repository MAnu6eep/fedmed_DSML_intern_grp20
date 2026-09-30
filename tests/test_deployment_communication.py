import json
import socket
import subprocess
from urllib.request import urlopen


SERVICES = [
    "federation-server",
    "hospital-a",
    "hospital-b",
    "hospital-c",
    "fedmed-api",
    "dashboard",
]


def run_compose(*args):
    result = subprocess.run(
        ["docker", "compose", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_hospitals_can_reach_federation_server():
    for hospital in ("hospital-a", "hospital-b", "hospital-c"):
        result = run_compose(
            "exec",
            "-T",
            hospital,
            "python",
            "-c",
            (
                "import socket; "
                "s=socket.create_connection(('federation-server',9092),5); "
                "s.close(); "
                "print('OK')"
            ),
        )
        assert "OK" in result


def test_fastapi_health_and_connected_nodes():
    with urlopen("http://localhost:8000/api/health", timeout=10) as response:
        assert response.status == 200
        payload = json.loads(response.read().decode())

    assert payload["status"] == "healthy"
    assert payload["nodes_connected"] == 3


def test_dashboard_is_reachable():
    with urlopen("http://localhost:5173", timeout=10) as response:
        assert response.status == 200
        html = response.read().decode()

    assert '<div id="root"></div>' in html
    assert "/src/main.tsx" in html


def test_dashboard_can_reach_fastapi():
    result = run_compose(
        "exec",
        "-T",
        "dashboard",
        "node",
        "-e",
        (
            "fetch('http://fedmed-api:8000/api/health')"
            ".then(r => { if (!r.ok) process.exit(1); return r.text(); })"
            ".then(console.log)"
            ".catch(() => process.exit(1))"
        ),
    )

    payload = json.loads(result)
    assert payload["status"] == "healthy"
    assert payload["nodes_connected"] == 3


def test_all_required_services_have_running_containers():
    result = run_compose("ps", "-q")
    container_ids = [line.strip() for line in result.splitlines() if line.strip()]

    assert len(container_ids) == len(SERVICES)
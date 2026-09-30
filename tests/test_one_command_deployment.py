import json
import socket
import subprocess
import time
from urllib.error import URLError
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


def wait_for_http(url, timeout=120, interval=2):
    deadline = time.time() + timeout
    last_error = None

    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return response.read()
        except (URLError, ConnectionError, TimeoutError, OSError) as exc:
            last_error = exc

        time.sleep(interval)

    raise AssertionError(
        f"{url} did not become available within {timeout} seconds. "
        f"Last error: {last_error}"
    )


def test_one_command_stack_startup():
    run_compose("up", "-d")

    try:
        result = run_compose("ps", "-q")
        container_ids = [
            line.strip()
            for line in result.splitlines()
            if line.strip()
        ]

        assert len(container_ids) == len(SERVICES)

        # Verify the federation server is reachable from every hospital.
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

        # Wait for FastAPI and verify all three hospitals are connected.
        api_body = wait_for_http(
            "http://localhost:8000/api/health",
            timeout=120,
            interval=2,
        )
        payload = json.loads(api_body.decode())

        assert payload["status"] == "healthy"
        assert payload["nodes_connected"] == 3

        # Wait for the React/Vite dashboard to become ready.
        dashboard_body = wait_for_http(
            "http://localhost:5173",
            timeout=120,
            interval=2,
        )
        dashboard_html = dashboard_body.decode()

        assert '<div id="root"></div>' in dashboard_html
        assert "/src/main.tsx" in dashboard_html

    finally:
        run_compose("down")
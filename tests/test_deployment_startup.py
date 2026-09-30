import json
import subprocess


EXPECTED_SERVICES = {
    "federation-server",
    "hospital-a",
    "hospital-b",
    "hospital-c",
    "fedmed-api",
    "dashboard",
}


def run_compose(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args],
        check=True,
        text=True,
        capture_output=True,
    )


def test_complete_stack_services_are_defined() -> None:
    result = run_compose("config", "--services")

    services = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    }

    assert services == EXPECTED_SERVICES


def test_complete_stack_services_are_running() -> None:
    result = run_compose("ps", "-q")

    container_ids = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    assert len(container_ids) == len(EXPECTED_SERVICES)


def test_required_services_have_running_containers() -> None:
    for service in sorted(EXPECTED_SERVICES):
        result = run_compose("ps", "-q", service)

        container_id = result.stdout.strip()

        assert container_id, (
            f"No running container found for service: {service}"
        )
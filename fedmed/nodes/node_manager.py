"""Baseline entrypoint for running a FedMed hospital node."""

import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import yaml

from fedmed.nodes.registry import HospitalNode, registry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HEALTH_HOST = "0.0.0.0"
HEALTH_PORT = 8080


class HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP health endpoint for the hospital container."""

    def do_GET(self):
        if self.path == "/":
            body = b'{"status":"healthy"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP request logging."""
        return


def load_hospital_config(hospital_id: str) -> dict:
    """Load the YAML configuration for the selected hospital."""
    config_path = PROJECT_ROOT / "hospitals" / hospital_id / "config.yaml"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Hospital configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid hospital configuration: {config_path}")

    return config


def check_node_health(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check whether a hospital node health endpoint is reachable."""
    url = f"http://{host}:{port}/"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def verify_registered_nodes() -> dict[str, bool]:
    """Check the health of all registered hospital nodes."""
    results: dict[str, bool] = {}

    for node in registry.list_nodes():
        healthy = check_node_health(node.host, node.port)

        registry.update_status(
            node.id,
            "online" if healthy else "offline",
        )

        results[node.id] = healthy

    return results


def start_node(hospital_id: str) -> HospitalNode:
    """Create and register a hospital node from its YAML configuration."""
    config = load_hospital_config(hospital_id)

    node_config = config["node"]
    data_config = config["data"]

    node = HospitalNode(
        id=node_config["id"],
        name=node_config["name"],
        host=node_config.get("host", "127.0.0.1"),
        port=node_config["port"],
        grpc_port=node_config["grpc_port"],
        data_dir=data_config["data_dir"],
    )

    registry.register(node)
    registry.update_status(node.id, "online", sample_count=node.sample_count)

    return node


def start_health_server() -> HTTPServer:
    """Start the minimal HTTP health server on port 8080."""
    server = HTTPServer((HEALTH_HOST, HEALTH_PORT), HealthHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    """Start the baseline hospital node process."""
    hospital_id = os.getenv("HOSPITAL_ID", "hospital_a")

    node = start_node(hospital_id)
    start_health_server()

    print(
        f"FedMed hospital node started: "
        f"{node.id} ({node.name}) "
        f"HTTP={node.port} gRPC={node.grpc_port} "
        f"health={HEALTH_PORT} status={node.status}",
        flush=True,
    )

    # Keep the container process alive without requiring interactive stdin.
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print(f"FedMed hospital node stopping: {node.id}", flush=True)


if __name__ == "__main__":
    main()

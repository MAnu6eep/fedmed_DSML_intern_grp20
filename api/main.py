"""
api/main.py

FastAPI backend server for FedMed telemetry,
live training metrics, and node orchestration.
"""

from datetime import datetime, timezone
from typing import Dict
import os
import urllib.error
import urllib.request

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="FedMed Telemetry API",
    description=(
        "Live metrics, secure aggregation status, and node health "
        "monitor for Federated Medical Segmentation."
    ),
    version="0.1.0",
)

HOSPITAL_NODES = {
    "hospital_a": (
        os.getenv("HOSPITAL_A_HOST", "hospital-a"),
        int(os.getenv("HOSPITAL_A_PORT", "8080")),
    ),
    "hospital_b": (
        os.getenv("HOSPITAL_B_HOST", "hospital-b"),
        int(os.getenv("HOSPITAL_B_PORT", "8080")),
    ),
    "hospital_c": (
        os.getenv("HOSPITAL_C_HOST", "hospital-c"),
        int(os.getenv("HOSPITAL_C_PORT", "8080")),
    ),
}


def check_hospital_health(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True when a hospital health endpoint responds with HTTP 200."""
    url = f"http://{host}:{port}/"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def get_available_hospitals() -> dict[str, bool]:
    """Check the current health of all configured hospital nodes."""
    return {
        hospital_id: check_hospital_health(host, port)
        for hospital_id, (host, port) in HOSPITAL_NODES.items()
    }


# Enable CORS for React Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    active_strategy: str
    nodes_connected: int
    uptime_status: str


@app.get("/api/health", response_model=HealthResponse)
def get_health_status() -> Dict[str, object]:
    """Return system health and the number of reachable hospital nodes."""
    node_health = get_available_hospitals()
    nodes_connected = sum(node_health.values())

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_strategy": "FedAvg",
        "nodes_connected": nodes_connected,
        "uptime_status": "operational",
    }


@app.get("/")
def root():
    return {
        "message": "FedMed Backend API is running. Access docs at /docs"
    }

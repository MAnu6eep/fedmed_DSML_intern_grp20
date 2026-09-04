"""
api/main.py

FastAPI backend server for FedMed telemetry,
live training metrics, and node orchestration.
"""

from datetime import datetime, timezone
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="FedMed Telemetry API",
    description=(
        "Live metrics, secure aggregation status, and node health "
        "monitor for Federated Medical Segmentation."
    ),
    version="1.1.0",
)


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
    """Returns baseline system operational health and server state."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_strategy": "FedAvg",
        "nodes_connected": 3,
        "uptime_status": "operational",
    }


@app.get("/")
def root():
    return {
        "message": "FedMed Backend API is running. Access docs at /docs"
    }

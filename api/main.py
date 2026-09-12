"""
api/main.py

FastAPI backend server for FedMed telemetry,
live training metrics, and node orchestration.
"""

from datetime import datetime, timezone
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.telemetry import (
    FederationEventType,
    create_event,
    telemetry_manager,
)


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


class Hospital(BaseModel):
    hospital_id: str
    name: str
    status: str
    port: int
    samples: int
    loss: float
    dice: float


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


@app.get("/api/hospitals", response_model=list[Hospital])
def get_hospitals():
    """Returns current telemetry/status of all hospital nodes."""
    return [
        {
            "hospital_id": "hospital_a",
            "name": "General Hospital Neuro",
            "status": "online",
            "port": 8081,
            "samples": 150,
            "loss": 0.2841,
            "dice": 0.912,
        },
        {
            "hospital_id": "hospital_b",
            "name": "St. Jude Imaging",
            "status": "training",
            "port": 8082,
            "samples": 120,
            "loss": 0.3152,
            "dice": 0.887,
        },
        {
            "hospital_id": "hospital_c",
            "name": "Metro Health Oncology",
            "status": "online",
            "port": 8083,
            "samples": 180,
            "loss": 0.2617,
            "dice": 0.924,
        },
    ]


@app.get("/api/metrics")
def get_metrics():
    """Returns current global training metrics."""
    return {
        "round": 1,
        "loss": 0.286,
        "dice": 0.908,
        "active_strategy": "FedAvg",
    }


@app.get("/")
def root():
    return {
        "message": " FedMed Backend API live! Docs available at /docs"
    }


# WebSocket telemetry endpoint
@app.websocket("/ws/telemetry")
async def telemetry_websocket(websocket: WebSocket):
    await telemetry_manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        telemetry_manager.disconnect(websocket)


# Temporary endpoint for testing live telemetry
@app.post("/api/telemetry/test/round-start")
async def test_round_start():
    event = create_event(
        event_type=FederationEventType.ROUND_STARTED,
        round_number=1,
        payload={
            "participants": 3,
            "total_clients": 3,
        },
    )

    await telemetry_manager.broadcast(event)

    return event


@app.post("/api/telemetry/test/training")
async def test_training():

    events = [
        create_event(
            event_type=FederationEventType.TRAINING_STARTED,
            round_number=1,
            hospital_id="hospital_a",
            status="training",
            payload={
                "samples": 150,
            },
        ),

        create_event(
            event_type=FederationEventType.TRAINING_COMPLETED,
            round_number=1,
            hospital_id="hospital_a",
            status="online",
            payload={
                "samples": 150,
                "participants": 1,
                "loss": 0.31,
                "dice": 0.86,
            },
        ),

        create_event(
            event_type=FederationEventType.TRAINING_STARTED,
            round_number=1,
            hospital_id="hospital_b",
            status="training",
            payload={
                "samples": 120,
            },
        ),

        create_event(
            event_type=FederationEventType.TRAINING_COMPLETED,
            round_number=1,
            hospital_id="hospital_b",
            status="online",
            payload={
                "samples": 120,
                "participants": 2,
                "loss": 0.29,
                "dice": 0.88,
            },
        ),

        create_event(
            event_type=FederationEventType.TRAINING_STARTED,
            round_number=1,
            hospital_id="hospital_c",
            status="training",
            payload={
                "samples": 180,
            },
        ),

        create_event(
            event_type=FederationEventType.TRAINING_COMPLETED,
            round_number=1,
            hospital_id="hospital_c",
            status="online",
            payload={
                "samples": 180,
                "participants": 3,
                "loss": 0.27,
                "dice": 0.90,
            },
        ),
    ]

    for event in events:
        await telemetry_manager.broadcast(event)

    return {
        "message": "Training telemetry broadcasted",
        "events": events,
    }


@app.post("/api/telemetry/test/aggregation-start")
async def test_aggregation_start():

    event = create_event(
        event_type=FederationEventType.AGGREGATION_STARTED,
        round_number=1,
        payload={},
    )

    await telemetry_manager.broadcast(event)

    return event


@app.post("/api/telemetry/test/aggregation-complete")
async def test_aggregation_complete():

    event = create_event(
        event_type=FederationEventType.AGGREGATION_COMPLETED,
        round_number=1,
        payload={
            "loss": 0.286,
            "dice": 0.908,
            "participants": 3,
            "total_clients": 3,
            "aggregation_time_ms": 1840,
        },
    )

    await telemetry_manager.broadcast(event)

    return event


@app.post("/api/telemetry/test/dropout")
async def test_dropout():

    event = create_event(
        event_type=FederationEventType.CLIENT_DISCONNECTED,
        round_number=1,
        hospital_id="hospital_b",
        status="offline",
        payload={
            "reason": "simulated_dropout",
            "participants": 2,
            "total_clients": 3,
        },
    )

    await telemetry_manager.broadcast(event)

    return event


@app.post("/api/telemetry/test/round-complete")
async def test_round_complete():

    event = create_event(
        event_type=FederationEventType.ROUND_COMPLETED,
        round_number=1,
        payload={
            "loss": 0.286,
            "dice": 0.908,
            "participants": 3,
            "total_clients": 3,
        },
    )

    await telemetry_manager.broadcast(event)

    return event
@app.post("/api/telemetry/test/reconnect")
async def test_reconnect():
    event = create_event(
        event_type=FederationEventType.CLIENT_CONNECTED,
        round_number=1,
        hospital_id="hospital_b",
        status="online",
        payload={
            "reason": "simulated_reconnection",
            "participants": 3,
            "total_clients": 3,
        },
    )

    await telemetry_manager.broadcast(event)
    return event
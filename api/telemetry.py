from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel, Field


class FederationEventType(str, Enum):
    ROUND_STARTED = "round_started"
    ROUND_COMPLETED = "round_completed"

    CLIENT_CONNECTED = "client_connected"
    CLIENT_DISCONNECTED = "client_disconnected"

    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETED = "training_completed"

    AGGREGATION_STARTED = "aggregation_started"
    AGGREGATION_COMPLETED = "aggregation_completed"


class FederationEvent(BaseModel):
    event_type: FederationEventType
    timestamp: str
    round: int
    hospital_id: str | None = None
    status: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def create_event(
    event_type: FederationEventType,
    round_number: int,
    hospital_id: str | None = None,
    status: str | None = None,
    payload: dict[str, Any] | None = None,
) -> FederationEvent:
    return FederationEvent(
        event_type=event_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
        round=round_number,
        hospital_id=hospital_id,
        status=status,
        payload=payload or {},
    )


class TelemetryManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, event: FederationEvent):
        disconnected = []

        for websocket in self.connections:
            try:
                await websocket.send_json(
                    event.model_dump(mode="json")
                )
            except Exception:
                disconnected.append(websocket)

        for websocket in disconnected:
            self.disconnect(websocket)


telemetry_manager = TelemetryManager()
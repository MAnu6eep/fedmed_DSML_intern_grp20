import type { FederationTelemetryEvent } from "../types/federationTelemetry";

const WS_URL = "ws://127.0.0.1:8000/ws/telemetry";

export class TelemetryClient {
  private socket: WebSocket | null = null;

  connect(
    onEvent: (event: FederationTelemetryEvent) => void,
    onOpen?: () => void,
    onError?: () => void,
    onClose?: () => void
  ) {
    this.socket = new WebSocket(WS_URL);

    this.socket.onopen = () => {
      console.log("FedMed telemetry WebSocket connected");

      if (onOpen) {
        onOpen();
      }
    };

    this.socket.onmessage = (message) => {
      try {
        const event = JSON.parse(
          message.data
        ) as FederationTelemetryEvent;

        onEvent(event);
      } catch (error) {
        console.error("Invalid telemetry event:", error);
      }
    };

    this.socket.onerror = () => {
      console.error("Telemetry WebSocket error");

      if (onError) {
        onError();
      }
    };

    this.socket.onclose = () => {
      console.log("FedMed telemetry WebSocket disconnected");

      if (onClose) {
        onClose();
      }
    };
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }
}
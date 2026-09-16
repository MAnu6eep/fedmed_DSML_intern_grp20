export type FederationEventType =
  | "round_started"
  | "round_completed"
  | "client_connected"
  | "client_disconnected"
  | "training_started"
  | "training_completed"
  | "aggregation_started"
  | "aggregation_completed";

export interface FederationTelemetryPayload {
  loss?: number;
  dice?: number;
  participants?: number;
  total_clients?: number;
  duration_ms?: number;
  duration_seconds?: number;
  aggregation_time_ms?: number;
  samples?: number;

  // Training metrics
  training_loss?: number;
  validation_loss?: number;

  // Communication metrics
  communication_payload_size?: number;

  // Round metrics
  round_duration?: number;
}

export interface FederationTelemetryEvent {
  event_type: FederationEventType;
  timestamp: string;
  round: number;
  hospital_id: string | null;
  status: string | null;
  payload: FederationTelemetryPayload;
}
export type FederationEventType =
  | "round_started"
  | "round_completed"
  | "client_connected"
  | "client_disconnected"
  | "training_started"
  | "training_completed"
  | "aggregation_started"
  | "aggregation_completed";

export interface FederationTelemetryEvent {
  event_type: FederationEventType;
  timestamp: string;
  round: number;
  hospital_id: string | null;
  status: string | null;
  payload: Record<string, unknown>;
}
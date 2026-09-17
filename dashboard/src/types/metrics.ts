export interface FederationMetricPoint {
  round: number;
  timestamp: string;

  trainingLoss?: number;
  validationLoss?: number;
  diceScore?: number;

  communicationPayloadSize?: number;
  roundDuration?: number;
}
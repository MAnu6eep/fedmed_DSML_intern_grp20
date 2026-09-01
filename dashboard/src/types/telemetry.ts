export interface HospitalTelemetry {
    id: string;
    name: string;
    host: string;
    port: number;
    grpcPort: number;
    sampleCount: number;
    status: 'online' | 'training' | 'offline' | 'error';
    currentRound: number;
    localLoss: number;
    localDice: number;
    lastHeartbeat: string;
    isSecAggActive: boolean;
  }
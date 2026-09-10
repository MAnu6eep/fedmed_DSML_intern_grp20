import { create } from "zustand";
import type {
  FederationTelemetryEvent,
} from "../types/federationTelemetry";

export interface ConvergenceMetric {
  round: number;
  loss: number;
  dice: number;
}

interface TelemetryStore {
  events: FederationTelemetryEvent[];
  connected: boolean;

  currentRound: number;
  trainingProgress: number;

  globalLoss: number;
  globalDice: number;

  participants: number;
  totalClients: number;

  aggregationActive: boolean;
  aggregationDuration: number;

  convergence: ConvergenceMetric[];

  addEvent: (event: FederationTelemetryEvent) => void;
  setConnected: (connected: boolean) => void;
  clearEvents: () => void;
}

export const useTelemetryStore = create<TelemetryStore>((set) => ({
  events: [],
  connected: false,

  currentRound: 0,
  trainingProgress: 0,

  globalLoss: 0,
  globalDice: 0,

  participants: 0,
  totalClients: 3,

  aggregationActive: false,
  aggregationDuration: 0,

  convergence: [],

  addEvent: (event) => {
    set((state) => {
      let progress = state.trainingProgress;

      let globalLoss = state.globalLoss;
      let globalDice = state.globalDice;

      let participants = state.participants;
      let totalClients = state.totalClients;

      let aggregationActive = state.aggregationActive;
      let aggregationDuration = state.aggregationDuration;

      let convergence = state.convergence;

      // New federated round
      if (event.event_type === "round_started") {
        progress = 0;

        if (event.payload.total_clients !== undefined) {
          totalClients = event.payload.total_clients;
        }

        if (event.payload.participants !== undefined) {
          participants = event.payload.participants;
        }
      }

      // Client starts local training
      if (event.event_type === "training_started") {
        progress = Math.min(
          state.trainingProgress + 10,
          90
        );
      }

      // Client completes local training
      if (event.event_type === "training_completed") {
        progress = Math.min(
          state.trainingProgress + 20,
          90
        );

        if (event.payload.participants !== undefined) {
          participants = event.payload.participants;
        }
      }

      // Aggregation starts
      if (event.event_type === "aggregation_started") {
        aggregationActive = true;
      }

      // Aggregation completes
      if (event.event_type === "aggregation_completed") {
        aggregationActive = false;
        progress = 90;

        if (event.payload.duration_seconds !== undefined) {
          aggregationDuration =
            event.payload.duration_seconds;
        }

        if (event.payload.aggregation_time_ms !== undefined) {
          aggregationDuration =
            event.payload.aggregation_time_ms / 1000;
        }

        if (event.payload.loss !== undefined) {
          globalLoss = event.payload.loss;
        }

        if (event.payload.dice !== undefined) {
          globalDice = event.payload.dice;
        }

        if (
          event.payload.loss !== undefined &&
          event.payload.dice !== undefined
        ) {
          const metric: ConvergenceMetric = {
            round: event.round,
            loss: event.payload.loss,
            dice: event.payload.dice,
          };

          convergence = [
            ...state.convergence.filter(
              (item) => item.round !== event.round
            ),
            metric,
          ].sort((a, b) => a.round - b.round);
        }
      }

      // Round completed
      if (event.event_type === "round_completed") {
        progress = 100;

        if (event.payload.loss !== undefined) {
          globalLoss = event.payload.loss;
        }

        if (event.payload.dice !== undefined) {
          globalDice = event.payload.dice;
        }

        if (
          event.payload.loss !== undefined &&
          event.payload.dice !== undefined
        ) {
          const metric: ConvergenceMetric = {
            round: event.round,
            loss: event.payload.loss,
            dice: event.payload.dice,
          };

          convergence = [
            ...state.convergence.filter(
              (item) => item.round !== event.round
            ),
            metric,
          ].sort((a, b) => a.round - b.round);
        }
      }

      return {
        events: [...state.events.slice(-49), event],

        currentRound: Math.max(
          state.currentRound,
          event.round
        ),

        trainingProgress: progress,

        globalLoss,
        globalDice,

        participants,
        totalClients,

        aggregationActive,
        aggregationDuration,

        convergence,
      };
    });
  },

  setConnected: (connected) => {
    set({ connected });
  },

  clearEvents: () => {
    set({
      events: [],
      connected: false,
      currentRound: 0,
      trainingProgress: 0,
      globalLoss: 0,
      globalDice: 0,
      participants: 0,
      totalClients: 3,
      aggregationActive: false,
      aggregationDuration: 0,
      convergence: [],
    });
  },
}));
import { create } from "zustand";
import type { FederationTelemetryEvent } from "../types/federationTelemetry";

interface TelemetryStore {
  events: FederationTelemetryEvent[];
  connected: boolean;
  currentRound: number;
  trainingProgress: number;

  addEvent: (event: FederationTelemetryEvent) => void;
  setConnected: (connected: boolean) => void;
  clearEvents: () => void;
}

export const useTelemetryStore = create<TelemetryStore>((set) => ({
  events: [],
  connected: false,
  currentRound: 0,
  trainingProgress: 0,

  addEvent: (event) => {
    set((state) => {
      let progress = state.trainingProgress;

      if (event.event_type === "round_started") {
        progress = 0;
      }

      if (event.event_type === "round_completed") {
        progress = 100;
      }

      if (event.event_type === "training_started") {
        progress = Math.min(
          state.trainingProgress + 10,
          90
        );
      }

      if (event.event_type === "training_completed") {
        progress = Math.min(
          state.trainingProgress + 20,
          90
        );
      }

      if (event.event_type === "aggregation_completed") {
        progress = 90;
      }

      return {
        events: [...state.events.slice(-49), event],
        currentRound: Math.max(
          state.currentRound,
          event.round
        ),
        trainingProgress: progress,
      };
    });
  },

  setConnected: (connected) => {
    set({ connected });
  },

  clearEvents: () => {
    set({
      events: [],
      currentRound: 0,
      trainingProgress: 0,
    });
  },
}));
import { create } from "zustand";
import type { FederationTelemetryEvent } from "../types/federationTelemetry";

interface TelemetryStore {
  events: FederationTelemetryEvent[];
  connected: boolean;

  addEvent: (event: FederationTelemetryEvent) => void;
  setConnected: (connected: boolean) => void;
  clearEvents: () => void;
}

export const useTelemetryStore = create<TelemetryStore>((set) => ({
  events: [],
  connected: false,

  addEvent: (event) => {
    set((state) => ({
      events: [...state.events.slice(-49), event],
    }));
  },

  setConnected: (connected) => {
    set({ connected });
  },

  clearEvents: () => {
    set({ events: [] });
  },
}));
import React, { useEffect } from "react";

import { Sidebar } from "./components/Sidebar";
import { Header } from "./components/Header";
import { HospitalNodeCard } from "./components/HospitalNodeCard";

import { useHospitalStore } from "./store/hospitalStore";
import { TelemetryClient } from "./api/telemetry";
import { useTelemetryStore } from "./store/telemetryStore";

import type { Hospital } from "./api/client";
import type { FederationEventType } from "./types/federationTelemetry";

export const App: React.FC = () => {
  // Hospital state
  const hospitals = useHospitalStore(
    (state) => state.hospitals
  );

  const fetchHospitals = useHospitalStore(
    (state) => state.fetchHospitals
  );

  const updateHospitalStatus = useHospitalStore(
    (state) => state.updateHospitalStatus
  );

  // Telemetry state
  const addEvent = useTelemetryStore(
    (state) => state.addEvent
  );

  const setConnected = useTelemetryStore(
    (state) => state.setConnected
  );

  const currentRound = useTelemetryStore(
    (state) => state.currentRound
  );

  const trainingProgress = useTelemetryStore(
    (state) => state.trainingProgress
  );

  const events = useTelemetryStore(
    (state) => state.events
  );

  const connected = useTelemetryStore(
    (state) => state.connected
  );

  // Federation event → hospital status
  const statusByEvent: Partial<
    Record<FederationEventType, Hospital["status"]>
  > = {
    client_connected: "online",
    client_disconnected: "offline",
    training_started: "training",
    training_completed: "online",
  };

  // Fetch hospitals
  useEffect(() => {
    fetchHospitals();
  }, [fetchHospitals]);

  // WebSocket telemetry connection
  useEffect(() => {
    const telemetryClient = new TelemetryClient();

    telemetryClient.connect(
      // Event received
      (event) => {
        console.log("Telemetry event:", event);

        addEvent(event);

        if (event.hospital_id) {
          const nextStatus =
            statusByEvent[event.event_type];

          if (nextStatus) {
            updateHospitalStatus(
              event.hospital_id,
              nextStatus
            );
          }
        }
      },

      // Connected
      () => {
        console.log(
          "FedMed telemetry WebSocket connected"
        );

        setConnected(true);
      },

      // Error
      () => {
        console.error(
          "FedMed telemetry WebSocket error"
        );

        setConnected(false);
      },

      // Closed
      () => {
        console.log(
          "FedMed telemetry WebSocket disconnected"
        );

        setConnected(false);
      }
    );

    return () => {
      telemetryClient.disconnect();
      setConnected(false);
    };
  }, [
    addEvent,
    setConnected,
    updateHospitalStatus,
  ]);

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-100">

      <Sidebar />

      <div className="flex-1 flex flex-col">

        <Header />

        <main className="p-8 space-y-6 flex-1 overflow-y-auto">

          {/* Page heading */}
          <div>
            <h2 className="text-xl font-semibold text-white">
              Federated Hospital Nodes
            </h2>

            <p className="text-sm text-slate-400">
              Live federated training activity
            </p>
          </div>


          {/* Federation summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

            {/* Current Round */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">

              <p className="text-sm text-slate-400">
                Current Federated Round
              </p>

              <p className="text-3xl font-bold text-white mt-2">
                Round {currentRound}
              </p>

              <p className="text-xs text-slate-500 mt-2">
                Live federation round
              </p>

            </div>


            {/* Training Progress */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">

              <div className="flex items-center justify-between">

                <div>
                  <p className="text-sm text-slate-400">
                    Training Progress
                  </p>

                  <p className="text-3xl font-bold text-white mt-2">
                    {trainingProgress}%
                  </p>
                </div>

                <div
                  className={`h-3 w-3 rounded-full ${
                    trainingProgress === 100
                      ? "bg-green-500"
                      : "bg-blue-500"
                  }`}
                />

              </div>

              <div className="w-full bg-slate-800 rounded-full h-2 mt-4">

                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                  style={{
                    width: `${trainingProgress}%`,
                  }}
                />

              </div>

            </div>


            {/* WebSocket */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">

              <p className="text-sm text-slate-400">
                Telemetry Connection
              </p>

              <div className="flex items-center gap-3 mt-3">

                <div
                  className={`h-3 w-3 rounded-full ${
                    connected
                      ? "bg-green-500"
                      : "bg-red-500"
                  }`}
                />

                <p className="text-lg font-semibold text-white">
                  {connected
                    ? "WebSocket Connected"
                    : "Disconnected"}
                </p>

              </div>

              <p className="text-xs text-slate-500 mt-2">
                Real-time federation updates
              </p>

            </div>

          </div>


          {/* Hospital nodes */}
          <div>

            <h3 className="text-lg font-semibold text-white mb-4">
              Hospital Nodes
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

              {hospitals.map((hospital) => (

                <HospitalNodeCard
                  key={hospital.hospital_id}

                  node={{
                    id: hospital.hospital_id,
                    name: hospital.name,
                    host: "127.0.0.1",
                    port: hospital.port,
                    grpcPort: hospital.port,
                    sampleCount: hospital.samples,
                    status: hospital.status,
                    currentRound: currentRound,
                    localLoss: hospital.loss,
                    localDice: hospital.dice,
                    lastHeartbeat:
                      new Date().toISOString(),
                    isSecAggActive: true,
                  }}
                />

              ))}

            </div>

          </div>


          {/* Live activity */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">

            <div className="flex items-center justify-between mb-4">

              <div>
                <h3 className="text-lg font-semibold text-white">
                  Live Federation Activity
                </h3>

                <p className="text-sm text-slate-400">
                  Real-time events from federation server
                </p>
              </div>

              <span
                className={`px-3 py-1 rounded-full text-xs font-medium ${
                  connected
                    ? "bg-green-500/10 text-green-400"
                    : "bg-red-500/10 text-red-400"
                }`}
              >
                {connected
                  ? "LIVE"
                  : "OFFLINE"}
              </span>

            </div>


            <div className="space-y-2">

              {events.length === 0 ? (

                <p className="text-sm text-slate-500">
                  Waiting for federation events...
                </p>

              ) : (

                events
                  .slice()
                  .reverse()
                  .slice(0, 8)
                  .map((event, index) => (

                    <div
                      key={`${event.timestamp}-${index}`}
                      className="flex items-center justify-between bg-slate-950 rounded-lg px-4 py-3"
                    >

                      <div>

                        <p className="text-sm text-slate-200">
                          {event.event_type
                            .replaceAll("_", " ")
                            .replace(
                              /\b\w/g,
                              (char) =>
                                char.toUpperCase()
                            )}
                        </p>

                        <p className="text-xs text-slate-500">
                          {event.hospital_id
                            ? event.hospital_id
                            : "Federation Server"}
                        </p>

                      </div>

                      <span className="text-xs text-slate-500">
                        Round {event.round}
                      </span>

                    </div>

                  ))

              )}

            </div>

          </div>

        </main>

      </div>

    </div>
  );
};

export default App;
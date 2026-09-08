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
  const hospitals = useHospitalStore(
    (state) => state.hospitals
  );

  const fetchHospitals = useHospitalStore(
    (state) => state.fetchHospitals
  );

  const updateHospitalStatus = useHospitalStore(
    (state) => state.updateHospitalStatus
  );

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

  const globalLoss = useTelemetryStore(
    (state) => state.globalLoss
  );

  const globalDice = useTelemetryStore(
    (state) => state.globalDice
  );

  const participants = useTelemetryStore(
    (state) => state.participants
  );

  const totalClients = useTelemetryStore(
    (state) => state.totalClients
  );

  const aggregationActive = useTelemetryStore(
    (state) => state.aggregationActive
  );

  const aggregationDuration = useTelemetryStore(
    (state) => state.aggregationDuration
  );

  const convergence = useTelemetryStore(
    (state) => state.convergence
  );

  const events = useTelemetryStore(
    (state) => state.events
  );

  const connected = useTelemetryStore(
    (state) => state.connected
  );

  const statusByEvent: Partial<
    Record<FederationEventType, Hospital["status"]>
  > = {
    client_connected: "online",
    client_disconnected: "offline",
    training_started: "training",
    training_completed: "online",
  };

  useEffect(() => {
    fetchHospitals();
  }, [fetchHospitals]);

  useEffect(() => {
    const telemetryClient = new TelemetryClient();

    telemetryClient.connect(
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

      () => {
        setConnected(true);
      },

      () => {
        setConnected(false);
      },

      () => {
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

          <div>
            <h2 className="text-xl font-semibold text-white">
              Federated Hospital Nodes
            </h2>

            <p className="text-sm text-slate-400">
              Live federated training and convergence telemetry
            </p>
          </div>


          {/* GLOBAL METRICS */}

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <p className="text-sm text-slate-400">
                Global FL Round
              </p>

              <p className="text-3xl font-bold text-white mt-2">
                Round {currentRound}
              </p>

              <p className="text-xs text-slate-500 mt-2">
                Current federated round
              </p>
            </div>


            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <p className="text-sm text-slate-400">
                Global Loss
              </p>

              <p className="text-3xl font-bold text-white mt-2">
                {globalLoss > 0
                  ? globalLoss.toFixed(4)
                  : "--"}
              </p>

              <p className="text-xs text-slate-500 mt-2">
                Global model loss
              </p>
            </div>


            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <p className="text-sm text-slate-400">
                Global Dice
              </p>

              <p className="text-3xl font-bold text-white mt-2">
                {globalDice > 0
                  ? globalDice.toFixed(4)
                  : "--"}
              </p>

              <p className="text-xs text-slate-500 mt-2">
                Global segmentation score
              </p>
            </div>


            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <p className="text-sm text-slate-400">
                Client Participation
              </p>

              <p className="text-3xl font-bold text-white mt-2">
                {participants}/{totalClients}
              </p>

              <p className="text-xs text-slate-500 mt-2">
                Clients participating this round
              </p>
            </div>

          </div>


          {/* TRAINING + AGGREGATION */}

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">

              <p className="text-sm text-slate-400">
                Training Progress
              </p>

              <p className="text-3xl font-bold text-white mt-2">
                {trainingProgress}%
              </p>

              <div className="w-full bg-slate-800 rounded-full h-2 mt-4">

                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                  style={{
                    width: `${trainingProgress}%`,
                  }}
                />

              </div>

            </div>


            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">

              <p className="text-sm text-slate-400">
                Aggregation Status
              </p>

              <div className="flex items-center gap-3 mt-3">

                <div
                  className={`h-3 w-3 rounded-full ${
                    aggregationActive
                      ? "bg-yellow-500"
                      : "bg-green-500"
                  }`}
                />

                <p className="text-lg font-semibold text-white">
                  {aggregationActive
                    ? "Aggregation Running"
                    : "Aggregation Complete"}
                </p>

              </div>

              <p className="text-xs text-slate-500 mt-2">
                {aggregationDuration > 0
                  ? `Last aggregation: ${aggregationDuration.toFixed(2)}s`
                  : "Waiting for aggregation telemetry"}
              </p>

            </div>


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


          {/* CONVERGENCE */}

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">

            <div className="mb-6">

              <h3 className="text-lg font-semibold text-white">
                Federated Model Convergence
              </h3>

              <p className="text-sm text-slate-400">
                Global loss and Dice score across completed rounds
              </p>

            </div>

            {convergence.length === 0 ? (

              <div className="h-64 flex items-center justify-center">
                <p className="text-sm text-slate-500">
                  Waiting for convergence metrics...
                </p>
              </div>

            ) : (

              <div className="w-full h-72 flex items-end gap-4">

                {convergence.map((metric) => (

                  <div
                    key={metric.round}
                    className="flex-1 h-full flex flex-col justify-end"
                  >

                    <div className="flex items-end justify-center gap-2 h-full">

                      <div
                        className="w-5 bg-blue-500 rounded-t"
                        style={{
                          height: `${Math.min(
                            metric.dice * 100,
                            100
                          )}%`,
                        }}
                        title={`Round ${metric.round} Dice: ${metric.dice}`}
                      />

                      <div
                        className="w-5 bg-red-500 rounded-t"
                        style={{
                          height: `${Math.min(
                            metric.loss * 100,
                            100
                          )}%`,
                        }}
                        title={`Round ${metric.round} Loss: ${metric.loss}`}
                      />

                    </div>

                    <p className="text-xs text-slate-500 text-center mt-2">
                      R{metric.round}
                    </p>

                  </div>

                ))}

              </div>

            )}

            <div className="flex gap-6 mt-4 text-xs text-slate-400">

              <div className="flex items-center gap-2">
                <span className="h-2 w-2 bg-blue-500 rounded-full" />
                Dice
              </div>

              <div className="flex items-center gap-2">
                <span className="h-2 w-2 bg-red-500 rounded-full" />
                Loss
              </div>

            </div>

          </div>


          {/* HOSPITAL NODES */}

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


          {/* LIVE ACTIVITY */}

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
                {connected ? "LIVE" : "OFFLINE"}
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

                      <div className="text-right">

                        <p className="text-xs text-slate-400">
                          Round {event.round}
                        </p>

                        <p className="text-xs text-slate-600">
                          {new Date(
                            event.timestamp
                          ).toLocaleTimeString()}
                        </p>

                      </div>

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
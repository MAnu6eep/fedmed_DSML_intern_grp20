import React, { useEffect } from "react";
import { Sidebar } from "./components/Sidebar";
import { Header } from "./components/Header";
import { HospitalNodeCard } from "./components/HospitalNodeCard";
import { useHospitalStore } from "./store/hospitalStore";

export const App: React.FC = () => {
  const hospitals = useHospitalStore((state) => state.hospitals);
  const fetchHospitals = useHospitalStore(
    (state) => state.fetchHospitals
  );

  useEffect(() => {
    fetchHospitals();
  }, [fetchHospitals]);

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
              Active participant nodes in current training round
            </p>
          </div>

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
                  currentRound: 0,
                  localLoss: hospital.loss,
                  localDice: hospital.dice,
                  lastHeartbeat: new Date().toISOString(),
                  isSecAggActive: true,
                }}
              />
            ))}
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;
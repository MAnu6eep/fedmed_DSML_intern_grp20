import React, { useEffect, useState } from "react";

// NAVIGATION & PRESENTATION
import { NavigationHeader, type ViewMode } from "./components/NavigationHeader";
import { PresentationMode } from "./components/presentation/PresentationMode";

// 14 INTERACTIVE PIPELINE SECTIONS
import { HeroSection } from "./components/interactive/HeroSection";
import { ProblemSection } from "./components/interactive/ProblemSection";
import { DataLayerSection } from "./components/interactive/DataLayerSection";
import { HospitalNodesSection } from "./components/interactive/HospitalNodesSection";
import { UNetArchitectureSection } from "./components/interactive/UNetArchitectureSection";
import { LocalTrainingSection } from "./components/interactive/LocalTrainingSection";
import { SecureCommSection } from "./components/interactive/SecureCommSection";
import { FederatedServerSection } from "./components/interactive/FederatedServerSection";
import { RoundsTimelineSection } from "./components/interactive/RoundsTimelineSection";
import { DataPartitionSection } from "./components/interactive/DataPartitionSection";
import { EvaluationSection } from "./components/interactive/EvaluationSection";
import { ComparisonSection } from "./components/interactive/ComparisonSection";
import { ResilienceSection } from "./components/interactive/ResilienceSection";
import { SegmentationSection } from "./components/interactive/SegmentationSection";
import { CompletePipelineSection } from "./components/interactive/CompletePipelineSection";

// EXISTING LIVE TELEMETRY DASHBOARD COMPONENTS
import { Sidebar } from "./components/Sidebar";
import { Header } from "./components/Header";
import { HospitalNodeCard } from "./components/HospitalNodeCard";
import { MetricsChart } from "./components/MetricsChart";
import SegmentationViewer from "./components/SegmentationViewer";
import type { SegmentationSlice } from "./types/segmentation";
import type { FederationMetricPoint } from "./types/metrics";
import { useHospitalStore } from "./store/hospitalStore";
import { TelemetryClient } from "./api/telemetry";
import { useTelemetryStore } from "./store/telemetryStore";
import type { Hospital } from "./api/client";
import type { FederationEventType, FederationTelemetryEvent } from "./types/federationTelemetry";

interface LocalMetric {
  loss: number;
  dice: number;
}

export const App: React.FC = () => {
  const [viewMode, setViewMode] = useState<ViewMode>("pipeline");

  // TELEMETRY STORE HOOKS
  const hospitals = useHospitalStore((state) => state.hospitals);
  const fetchHospitals = useHospitalStore((state) => state.fetchHospitals);
  const updateHospitalStatus = useHospitalStore((state) => state.updateHospitalStatus);

  const addEvent = useTelemetryStore((state) => state.addEvent);
  const setConnected = useTelemetryStore((state) => state.setConnected);
  const currentRound = useTelemetryStore((state) => state.currentRound);
  const globalLoss = useTelemetryStore((state) => state.globalLoss);
  const globalDice = useTelemetryStore((state) => state.globalDice);
  const participants = useTelemetryStore((state) => state.participants);
  const totalClients = useTelemetryStore((state) => state.totalClients);
  const connected = useTelemetryStore((state) => state.connected);

  const [localMetrics] = useState<Record<string, LocalMetric>>({});
  const [metrics, setMetrics] = useState<FederationMetricPoint[]>([]);
  const [segmentationSlices, setSegmentationSlices] = useState<SegmentationSlice[]>([]);
  const [segmentationLoading, setSegmentationLoading] = useState(true);

  const statusByEvent: Partial<Record<FederationEventType, Hospital["status"]>> = {
    client_connected: "online",
    client_disconnected: "offline",
    training_started: "training",
    training_completed: "online",
  };

  useEffect(() => {
    fetchHospitals();
  }, [fetchHospitals]);

  useEffect(() => {
    const loadMetrics = async () => {
      try {
        const response = await fetch("http://127.0.0.1:8000/api/metrics");
        if (!response.ok) return;
        const data = await response.json();
        setMetrics([
          {
            round: Number(data.round ?? 0),
            timestamp: new Date().toISOString(),
            trainingLoss: typeof data.training_loss === "number" ? data.training_loss : typeof data.loss === "number" ? data.loss : undefined,
            validationLoss: typeof data.validation_loss === "number" ? data.validation_loss : undefined,
            diceScore: typeof data.dice === "number" ? data.dice : undefined,
          },
        ]);
      } catch (error) {
        // Fallback for static presentation
      }
    };

    loadMetrics();
  }, []);

  useEffect(() => {
    const loadSegmentation = async () => {
      try {
        setSegmentationLoading(true);
        const response = await fetch("http://127.0.0.1:8000/api/segmentation");
        if (!response.ok) throw new Error("Segmentation API unavailable");
        const data = await response.json();
        if (Array.isArray(data.slices)) {
          setSegmentationSlices(
            data.slices.map((slice: { index: number; mri: string; ground_truth?: string | null; prediction?: string | null }) => ({
              sliceIndex: slice.index,
              mriSlice: slice.mri,
              groundTruthMask: slice.ground_truth ?? "",
              predictedMask: slice.prediction ?? "",
            }))
          );
        }
      } catch (error) {
        // Static mode fallback
      } finally {
        setSegmentationLoading(false);
      }
    };

    loadSegmentation();
  }, []);

  useEffect(() => {
    const telemetryClient = new TelemetryClient();
    telemetryClient.connect(
      (event: FederationTelemetryEvent) => {
        addEvent(event);
        if (event.hospital_id) {
          const nextStatus = statusByEvent[event.event_type];
          if (nextStatus) updateHospitalStatus(event.hospital_id, nextStatus);
        }
      },
      () => setConnected(true),
      () => setConnected(false),
      () => setConnected(false)
    );
    return () => {
      telemetryClient.disconnect();
      setConnected(false);
    };
  }, [addEvent, setConnected, updateHospitalStatus]);

  const scrollToPipeline = () => {
    const el = document.getElementById("problem");
    if (el) el.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500 selection:text-slate-950">
      {/* GLOBAL NAVIGATION HEADER */}
      <NavigationHeader
        currentView={viewMode}
        onViewChange={setViewMode}
        connected={connected}
      />

      {/* FULLSCREEN PRESENTATION MODE MODAL */}
      {viewMode === "presentation" && (
        <PresentationMode onExit={() => setViewMode("pipeline")} />
      )}

      {/* VIEW MODE 1: INTERACTIVE SYSTEM PIPELINE VISUALIZER */}
      {viewMode === "pipeline" && (
        <main className="space-y-0">
          <HeroSection
            onExplore={scrollToPipeline}
            onStartPresentation={() => setViewMode("presentation")}
          />
          <ProblemSection />
          <DataLayerSection />
          <HospitalNodesSection />
          <UNetArchitectureSection />
          <LocalTrainingSection />
          <SecureCommSection />
          <FederatedServerSection />
          <RoundsTimelineSection />
          <DataPartitionSection />
          <EvaluationSection />
          <ComparisonSection />
          <ResilienceSection />
          <SegmentationSection />
          <CompletePipelineSection />

          {/* FOOTER */}
          <footer className="py-12 px-6 bg-slate-950 border-t border-slate-800 text-center text-xs text-slate-400 space-y-2">
            <p className="font-bold text-slate-200">
              FedMed • Hybrid Privacy-Preserving Federated Learning for 3D Brain Tumor Segmentation
            </p>
            <p className="text-slate-400">
              Built with MONAI, PyTorch, Flower, TenSEAL (CKKS), SecAgg+, FastAPI, React & Tailwind CSS.
            </p>
          </footer>
        </main>
      )}

      {/* VIEW MODE 2: LIVE TELEMETRY DASHBOARD */}
      {viewMode === "telemetry" && (
        <div className="flex min-h-screen">
          <Sidebar />
          <div className="flex-1 flex flex-col">
            <Header />
            <main className="p-8 space-y-6 flex-1 overflow-y-auto">
              <div>
                <h2 className="text-xl font-semibold text-white">Federated Hospital Nodes Telemetry</h2>
                <p className="text-sm text-slate-400">Live federated training metrics and node status monitor</p>
              </div>

              {/* GLOBAL METRICS */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                  <p className="text-sm text-slate-400">Global FL Round</p>
                  <p className="text-3xl font-bold text-white mt-2">Round {currentRound}</p>
                </div>
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                  <p className="text-sm text-slate-400">Global Loss</p>
                  <p className="text-3xl font-bold text-white mt-2">{globalLoss > 0 ? globalLoss.toFixed(4) : "--"}</p>
                </div>
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                  <p className="text-sm text-slate-400">Global Dice Score</p>
                  <p className="text-3xl font-bold text-white mt-2">{globalDice > 0 ? globalDice.toFixed(4) : "--"}</p>
                </div>
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                  <p className="text-sm text-slate-400">Participation Quorum</p>
                  <p className="text-3xl font-bold text-white mt-2">{participants}/{totalClients}</p>
                </div>
              </div>

              <MetricsChart data={metrics} />

              {/* HOSPITAL NODES */}
              <div>
                <h3 className="text-lg font-semibold text-white mb-4">Hospital Nodes</h3>
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
                        localLoss: localMetrics[hospital.hospital_id]?.loss ?? hospital.loss,
                        localDice: localMetrics[hospital.hospital_id]?.dice ?? hospital.dice,
                        lastHeartbeat: new Date().toISOString(),
                        isSecAggActive: true,
                      }}
                    />
                  ))}
                </div>
              </div>

              {!segmentationLoading && <SegmentationViewer slices={segmentationSlices} />}
            </main>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
import React, { useState, useEffect } from "react";
import { Sidebar } from "./components/layout/Sidebar";
import { CentralDashboard } from "./views/CentralDashboard";
import { HospitalRegistry } from "./views/HospitalRegistry";
import { SimulateRemoteHospital } from "./views/SimulateRemoteHospital";
import { HospitalSideUI } from "./views/HospitalSideUI";
import { FederationOfModels } from "./views/FederationOfModels";
import { GlobalModelRegistry } from "./views/GlobalModelRegistry";
import { HospitalActivity } from "./views/HospitalActivity";
import { TestingAndExperiments } from "./views/TestingAndExperiments";
import { AboutProject } from "./views/AboutProject";
import { AlertCircle, RefreshCw, X, Play, ShieldCheck } from "lucide-react";

export type ViewState = 
  | "dashboard" 
  | "registry" 
  | "simulate_remote" 
  | "hospital_side" 
  | "federation" 
  | "model_registry" 
  | "activity" 
  | "testing_experiments"
  | "about";

export const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<ViewState>("dashboard");
  const [selectedHospital, setSelectedHospital] = useState<string | null>(null);
  const [executionMode, setExecutionMode] = useState<"standby" | "docker">("standby");

  // Docker entry check modal state
  const [showDockerModal, setShowDockerModal] = useState(false);
  const [dockerModalLoading, setDockerModalLoading] = useState(false);
  const [dockerModalError, setDockerModalError] = useState<string | null>(null);

  // Sync execution mode from backend
  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/execution-mode")
      .then((r) => r.json())
      .then((data) => {
        if (data.mode === "docker" || data.mode === "standby") {
          setExecutionMode(data.mode);
        }
      })
      .catch(() => {});
  }, []);

  // Entry check for Simulate Remote Hospital
  const handleNavigate = async (view: ViewState) => {
    if (view === "simulate_remote") {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/docker/status");
        const data = await res.json();
        if (data.docker_running) {
          // Case A: Docker is running -> continue to simulation screen
          setExecutionMode("docker");
          fetch("http://127.0.0.1:8000/api/execution-mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: "docker" }),
          }).catch(() => {});
          setCurrentView("simulate_remote");
        } else {
          // Case B: Docker is NOT running -> show modal BEFORE entering
          setDockerModalError(null);
          setShowDockerModal(true);
        }
      } catch {
        setDockerModalError(null);
        setShowDockerModal(true);
      }
    } else {
      setCurrentView(view);
    }
  };

  // Modal Action 1: Start Docker Simulation
  const handleStartDockerSimulation = async () => {
    setDockerModalLoading(true);
    setDockerModalError(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/api/docker/start-simulation", {
        method: "POST",
      });
      if (res.ok) {
        setExecutionMode("docker");
        setShowDockerModal(false);
        setCurrentView("simulate_remote");
      } else {
        const err = await res.json();
        setDockerModalError(
          err.detail || "Docker Desktop is currently not running. Please start Docker Desktop on your PC and click Retry."
        );
      }
    } catch (e: any) {
      setDockerModalError(
        e.message || "Failed to reach backend to start Docker. Ensure Docker Desktop is active."
      );
    } finally {
      setDockerModalLoading(false);
    }
  };

  // Modal Action 2: Continue in Standby Mode
  const handleContinueInStandby = () => {
    setExecutionMode("standby");
    fetch("http://127.0.0.1:8000/api/execution-mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: "standby" }),
    }).catch(() => {});
    setShowDockerModal(false);
    setDockerModalError(null);
    setCurrentView("registry"); // Return to native standby hospital-training workflow
  };

  const renderView = () => {
    switch (currentView) {
      case "dashboard":
        return <CentralDashboard onNavigate={handleNavigate} />;
      case "registry":
        return (
          <HospitalRegistry
            onOpenHospital={(id) => {
              setSelectedHospital(id);
              setCurrentView("hospital_side");
            }}
            onNavigateToSimulate={() => handleNavigate("simulate_remote")}
          />
        );
      case "simulate_remote":
        return (
          <SimulateRemoteHospital
            onOpenHospital={(id) => {
              setSelectedHospital(id);
              setCurrentView("hospital_side");
            }}
            onSwitchToStandby={handleContinueInStandby}
          />
        );
      case "hospital_side":
        return (
          <HospitalSideUI
            hospitalId={selectedHospital}
            onBack={() => setCurrentView("registry")}
          />
        );
      case "federation":
        return <FederationOfModels />;
      case "model_registry":
        return <GlobalModelRegistry />;
      case "activity":
        return <HospitalActivity />;
      case "testing_experiments":
        return <TestingAndExperiments />;
      case "about":
        return <AboutProject onBackToDashboard={() => setCurrentView("dashboard")} />;
      default:
        return <CentralDashboard onNavigate={handleNavigate} />;
    }
  };

  return (
    <div className="flex h-screen bg-navy-900 text-beige-100 font-sans overflow-hidden">
      {currentView !== "hospital_side" && (
        <Sidebar 
          currentView={currentView} 
          executionMode={executionMode}
          onViewChange={handleNavigate} 
        />
      )}

      <main className="flex-1 flex flex-col overflow-y-auto bg-navy-900">
        {/* Top Operational Status Header with Mode Indicator */}
        {currentView !== "hospital_side" && (
          <div className="bg-navy-950/70 border-b border-navy-800/80 px-8 py-2.5 flex items-center justify-between text-xs select-none">
            <div className="flex items-center space-x-2">
              <span className="text-slate-400 font-medium text-[11px] uppercase tracking-wider">
                Execution Mode:
              </span>
              <span
                className={`inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold ${
                  executionMode === "docker"
                    ? "bg-blue-500/20 text-blue-300 border border-blue-500/40"
                    : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                }`}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    executionMode === "docker" ? "bg-blue-400 animate-pulse" : "bg-emerald-400"
                  }`}
                />
                <span>
                  {executionMode === "docker"
                    ? "Docker Remote Hospital Simulation"
                    : "Standby / Native Mode"}
                </span>
              </span>
            </div>

            {/* Quick Switch to Standby button if in Docker mode */}
            {executionMode === "docker" && (
              <button
                onClick={handleContinueInStandby}
                className="text-xs text-slate-400 hover:text-emerald-300 flex items-center space-x-1 transition"
                title="Switch back to native standby execution"
              >
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
                <span>Switch to Standby Mode</span>
              </button>
            )}
          </div>
        )}

        {renderView()}
      </main>

      {/* Docker Desktop Check Modal (Case B) */}
      {showDockerModal && (
        <div className="fixed inset-0 z-50 bg-black/75 flex items-center justify-center p-4">
          <div className="bg-navy-800 border border-navy-700 rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-5">
            {/* Modal Header */}
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-3">
                <div className="p-2.5 bg-navy-900 rounded-xl border border-navy-700 text-amber-400 mt-0.5">
                  <AlertCircle className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white tracking-wide">
                    Remote Hospital Simulation
                  </h3>
                  <p className="text-xs text-amber-300/90 font-medium mt-0.5">
                    Docker Desktop is currently not running.
                  </p>
                </div>
              </div>
              <button
                onClick={() => {
                  setShowDockerModal(false);
                  setDockerModalError(null);
                }}
                className="text-slate-400 hover:text-white p-1"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Informational Message */}
            <div className="bg-navy-900/80 border border-navy-700/80 rounded-xl p-4 text-xs text-slate-300 leading-relaxed space-y-2">
              <p>
                FedMed can continue in <strong>Standby Mode</strong> without Docker, or you can start Docker to simulate remote hospitals using isolated containers.
              </p>
              <p className="text-[11px] text-slate-400 pt-1 border-t border-navy-800">
                * Docker is <strong>optional</strong>. Standby Mode executes the entire native training and federation workflow directly on your local machine with minimal RAM usage.
              </p>
            </div>

            {/* Error display if startup failed */}
            {dockerModalError && (
              <div className="bg-red-950/40 border border-red-500/50 p-3 rounded-lg text-xs text-red-200 space-y-1">
                <p className="font-bold text-red-300">Docker Startup State:</p>
                <p className="text-[11px] leading-relaxed">{dockerModalError}</p>
              </div>
            )}

            {/* Action Buttons */}
            <div className="space-y-2.5 pt-1">
              <button
                onClick={handleStartDockerSimulation}
                disabled={dockerModalLoading}
                className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-2.5 px-4 rounded-xl text-xs flex items-center justify-center space-x-2 transition shadow-sm"
              >
                {dockerModalLoading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Starting Docker Infrastructure...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>Start Docker Simulation</span>
                  </>
                )}
              </button>

              <button
                onClick={handleContinueInStandby}
                className="w-full bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/40 font-bold py-2.5 px-4 rounded-xl text-xs flex items-center justify-center space-x-2 transition"
              >
                <span>Continue in Standby Mode</span>
              </button>

              <button
                onClick={() => {
                  setShowDockerModal(false);
                  setDockerModalError(null);
                }}
                className="w-full bg-navy-900 hover:bg-navy-700 text-slate-400 hover:text-white font-medium py-2 rounded-xl text-xs transition border border-navy-700"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
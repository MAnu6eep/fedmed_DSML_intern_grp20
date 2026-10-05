import React from "react";
import { Activity, Play, Layers, Server, ShieldCheck, Zap } from "lucide-react";

export type ViewMode = "pipeline" | "presentation" | "telemetry";

interface NavigationHeaderProps {
  currentView: ViewMode;
  onViewChange: (mode: ViewMode) => void;
  connected?: boolean;
}

export const NavigationHeader: React.FC<NavigationHeaderProps> = ({
  currentView,
  onViewChange,
  connected = false,
}) => {
  return (
    <header className="sticky top-0 z-50 bg-slate-950/90 backdrop-blur-md border-b border-slate-800 px-6 py-4 flex flex-wrap items-center justify-between gap-4">
      {/* BRANDING */}
      <div className="flex items-center gap-3">
        <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-blue-600 to-indigo-600 p-0.5 shadow-lg shadow-cyan-500/20">
          <div className="h-full w-full bg-slate-950 rounded-[10px] flex items-center justify-center">
            <ShieldCheck className="h-5 w-5 text-cyan-400" />
          </div>
        </div>

        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-1.5">
              FED<span className="text-cyan-400">MED</span>
            </h1>
            <span className="px-2 py-0.5 text-[10px] font-semibold tracking-wider uppercase rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              v1.1.0 • 3D MRI
            </span>
          </div>
          <p className="text-xs text-slate-400 font-medium">
            Privacy-Preserving Federated Learning Control Center
          </p>
        </div>
      </div>

      {/* VIEW SWITCHER TABS */}
      <div className="flex items-center bg-slate-900/90 p-1 rounded-xl border border-slate-800">
        <button
          onClick={() => onViewChange("pipeline")}
          className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg transition-all ${
            currentView === "pipeline"
              ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/20"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          Interactive Pipeline Visualizer
        </button>

        <button
          onClick={() => onViewChange("presentation")}
          className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg transition-all ${
            currentView === "presentation"
              ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/20"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
          }`}
        >
          <Play className="w-3.5 h-3.5 text-emerald-400 fill-emerald-400/20" />
          Presentation Mode
        </button>

        <button
          onClick={() => onViewChange("telemetry")}
          className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg transition-all ${
            currentView === "telemetry"
              ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/20"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
          }`}
        >
          <Activity className="w-3.5 h-3.5" />
          Live Telemetry Dashboard
        </button>
      </div>

      {/* SYSTEM STATUS BADGES */}
      <div className="flex items-center gap-3 text-xs">
        <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
          <Server className="w-3.5 h-3.5 text-blue-400" />
          <span>Nodes: <strong className="text-white">3 Active</strong></span>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300">
          <span className={`h-2 w-2 rounded-full animate-pulse ${connected ? "bg-emerald-500" : "bg-cyan-500"}`} />
          <span>API: <strong className="text-white">{connected ? "Connected" : "Static Demo Mode"}</strong></span>
        </div>

        <button
          onClick={() => onViewChange("presentation")}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gradient-to-r from-emerald-500 to-teal-600 text-white font-semibold text-xs shadow-md shadow-emerald-500/20 hover:brightness-110 transition-all"
        >
          <Zap className="w-3.5 h-3.5" />
          PM Review Mode
        </button>
      </div>
    </header>
  );
};

import React, { useState } from "react";
import { ShieldAlert, RefreshCw, CheckCircle2, XCircle } from "lucide-react";

export const ResilienceSection: React.FC = () => {
  const [hospitalCOffline, setHospitalCOffline] = useState(false);

  return (
    <section id="resilience" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-semibold uppercase tracking-wider">
            <ShieldAlert className="w-3.5 h-3.5" />
            SECTION 12 — NODE RESILIENCE & DROPOUT
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            Fault-Tolerant Dynamic Node Manager
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            Monitors node heartbeats. If a hospital site drops offline due to network interruptions, training proceeds using the minimum 2/3 participation threshold.
          </p>
        </div>

        {/* INTERACTIVE SIMULATOR */}
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
          <div className="flex justify-between items-center border-b border-slate-800 pb-4">
            <div>
              <h3 className="text-lg font-bold text-white">Dynamic Dropout Simulation</h3>
              <p className="text-xs text-slate-400">Toggle Hospital C offline state to observe automatic server fault tolerance.</p>
            </div>

            <button
              onClick={() => setHospitalCOffline(!hospitalCOffline)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-bold text-xs transition-all ${
                hospitalCOffline
                  ? "bg-emerald-500 text-slate-950 hover:bg-emerald-400"
                  : "bg-red-500/20 text-red-300 border border-red-500/40 hover:bg-red-500/30"
              }`}
            >
              <RefreshCw className="w-3.5 h-3.5" />
              {hospitalCOffline ? "Reconnect Hospital C" : "Simulate Hospital C Dropout"}
            </button>
          </div>

          {/* HOSPITAL NODES GRID WITH DROPOUT STATUS */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-5 rounded-xl bg-slate-950 border border-emerald-500/40 space-y-2">
              <div className="flex justify-between items-center">
                <strong className="text-white text-sm">Hospital A</strong>
                <span className="flex items-center gap-1 text-[10px] font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
                  <CheckCircle2 className="w-3 h-3" /> ONLINE
                </span>
              </div>
              <p className="text-xs text-slate-400">Heartbeat: Active • Participating in Round</p>
            </div>

            <div className="p-5 rounded-xl bg-slate-950 border border-emerald-500/40 space-y-2">
              <div className="flex justify-between items-center">
                <strong className="text-white text-sm">Hospital B</strong>
                <span className="flex items-center gap-1 text-[10px] font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
                  <CheckCircle2 className="w-3 h-3" /> ONLINE
                </span>
              </div>
              <p className="text-xs text-slate-400">Heartbeat: Active • Participating in Round</p>
            </div>

            <div className={`p-5 rounded-xl bg-slate-950 border transition-all space-y-2 ${
              hospitalCOffline ? "border-red-500/60 opacity-80" : "border-emerald-500/40"
            }`}>
              <div className="flex justify-between items-center">
                <strong className="text-white text-sm">Hospital C</strong>
                <span className={`flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded ${
                  hospitalCOffline ? "bg-red-500/20 text-red-400" : "bg-emerald-500/10 text-emerald-400"
                }`}>
                  {hospitalCOffline ? <XCircle className="w-3 h-3" /> : <CheckCircle2 className="w-3 h-3" />}
                  {hospitalCOffline ? "OFFLINE (DROPOUT)" : "ONLINE"}
                </span>
              </div>
              <p className="text-xs text-slate-400">
                {hospitalCOffline ? "Heartbeat Lost • Re-synchronization Queued" : "Heartbeat: Active • Participating in Round"}
              </p>
            </div>
          </div>

          {/* STATUS NOTIFICATION BAR */}
          <div className={`p-4 rounded-xl text-xs font-mono flex items-center justify-between border ${
            hospitalCOffline
              ? "bg-amber-950/20 border-amber-500/40 text-amber-300"
              : "bg-emerald-950/20 border-emerald-500/40 text-emerald-300"
          }`}>
            <span>
              {hospitalCOffline
                ? "Federation Status: Hospital C Offline • Active quorum = 2/3 • Round proceeds smoothly."
                : "Federation Status: All 3 Hospital Nodes Healthy (3/3 Quorum Active)."}
            </span>
            <span className="font-bold uppercase text-[10px]">fedmed/resilience/dropout.py</span>
          </div>
        </div>
      </div>
    </section>
  );
};

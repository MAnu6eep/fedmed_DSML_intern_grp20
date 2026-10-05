import React, { useState } from "react";
import { AlertTriangle, ShieldCheck, Database, ArrowRight, Lock, Server } from "lucide-react";

export const ProblemSection: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"centralized" | "federated">("centralized");

  return (
    <section id="problem" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-semibold uppercase tracking-wider">
            <AlertTriangle className="w-3.5 h-3.5" />
            SECTION 1 — THE PRIVACY PARADIGM
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            Traditional Centralized ML vs. FedMed Federated Privacy
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            Why healthcare institutions cannot pool raw patient 3D MRI scans into a central cloud database, and how FedMed solves HIPAA / GDPR compliance.
          </p>
        </div>

        {/* INTERACTIVE TOGGLE */}
        <div className="flex justify-center">
          <div className="bg-slate-900 p-1 rounded-xl border border-slate-800 flex gap-2">
            <button
              onClick={() => setActiveTab("centralized")}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs font-bold transition-all ${
                activeTab === "centralized"
                  ? "bg-red-500/20 text-red-300 border border-red-500/40 shadow-lg shadow-red-500/10"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <AlertTriangle className="w-4 h-4 text-red-400" />
              Traditional Centralized Learning (Vulnerable)
            </button>
            <button
              onClick={() => setActiveTab("federated")}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-xs font-bold transition-all ${
                activeTab === "federated"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-lg shadow-emerald-500/10"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              FedMed Architecture (Private & Secure)
            </button>
          </div>
        </div>

        {/* COMPARISON VISUAL DISPLAY */}
        <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 relative overflow-hidden transition-all duration-500">
          {activeTab === "centralized" ? (
            <div className="space-y-6">
              <div className="flex items-center justify-between border-b border-red-900/40 pb-4">
                <div>
                  <h3 className="text-xl font-bold text-red-400 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5" />
                    Traditional Centralized Training Model
                  </h3>
                  <p className="text-xs text-slate-400">
                    Hospitals are required to transmit raw, unencrypted patient MRI volumes across public networks into a central server repository.
                  </p>
                </div>
                <span className="px-3 py-1 rounded-md bg-red-500/20 text-red-400 text-xs font-bold border border-red-500/30">
                  HIGH PRIVACY RISK
                </span>
              </div>

              {/* FLOW ANIMATION CENTRALIZED */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
                {["Hospital A", "Hospital B", "Hospital C"].map((hosp, idx) => (
                  <div key={idx} className="p-4 rounded-xl bg-slate-950 border border-red-900/30 text-center space-y-2 relative">
                    <Database className="w-6 h-6 text-red-400 mx-auto" />
                    <strong className="block text-sm text-slate-200">{hosp}</strong>
                    <span className="text-[10px] text-red-400 font-mono block">RAW MRI DATA (.nii.gz)</span>
                    <div className="flex items-center justify-center gap-1 text-[10px] text-red-400/80 animate-pulse">
                      <span>Transmitting Raw Data</span>
                      <ArrowRight className="w-3 h-3" />
                    </div>
                  </div>
                ))}

                {/* CENTRAL SERVER BOX */}
                <div className="p-5 rounded-xl bg-red-950/40 border border-red-700 text-center space-y-2 relative shadow-xl shadow-red-950/50">
                  <Server className="w-8 h-8 text-red-400 mx-auto animate-bounce" />
                  <strong className="block text-sm text-white">Central Data Pool</strong>
                  <p className="text-[11px] text-red-300">Single Point of Failure & Privacy Breach Vulnerability</p>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-red-950/20 border border-red-900/50 text-xs text-red-300 flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                <div>
                  <strong className="font-semibold block mb-1">Compliance & Regulatory Barriers:</strong>
                  Medical regulations strictly prohibit raw patient MRI transfers across institutional boundaries. Centralization risks patient re-identification and intellectual property leakage.
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="flex items-center justify-between border-b border-emerald-900/40 pb-4">
                <div>
                  <h3 className="text-xl font-bold text-emerald-400 flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5" />
                    FedMed Federated Learning Architecture
                  </h3>
                  <p className="text-xs text-slate-400">
                    Patient MRI data NEVER leaves hospital premises. Local 3D U-Nets train on-site; only encrypted model updates are transmitted.
                  </p>
                </div>
                <span className="px-3 py-1 rounded-md bg-emerald-500/20 text-emerald-400 text-xs font-bold border border-emerald-500/30">
                  ZERO DATA LEAKAGE
                </span>
              </div>

              {/* FLOW ANIMATION FEDERATED */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
                {["Hospital A", "Hospital B", "Hospital C"].map((hosp, idx) => (
                  <div key={idx} className="p-4 rounded-xl bg-slate-950 border border-emerald-500/40 text-center space-y-2 relative">
                    <div className="absolute top-2 right-2 px-1.5 py-0.5 rounded bg-emerald-500/20 text-[9px] text-emerald-400 font-bold">
                      LOCAL
                    </div>
                    <Database className="w-6 h-6 text-emerald-400 mx-auto" />
                    <strong className="block text-sm text-slate-200">{hosp}</strong>
                    <span className="text-[10px] text-slate-400 block font-mono">Raw Scans Locked Local</span>
                    <div className="pt-1 border-t border-slate-800 text-[10px] text-cyan-400 font-bold flex items-center justify-center gap-1">
                      <Lock className="w-3 h-3 text-cyan-400" />
                      Encrypted Model Δw Only
                    </div>
                  </div>
                ))}

                {/* FEDERATED AGGREGATOR */}
                <div className="p-5 rounded-xl bg-slate-950 border border-cyan-500 text-center space-y-2 relative shadow-xl shadow-cyan-950/50">
                  <Server className="w-8 h-8 text-cyan-400 mx-auto" />
                  <strong className="block text-sm text-white">Flower Federated Server</strong>
                  <p className="text-[11px] text-cyan-300">SecAgg+ & TenSEAL Parameter Aggregation Engine</p>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-emerald-950/20 border border-emerald-900/50 text-xs text-emerald-300 flex items-start gap-3">
                <ShieldCheck className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <strong className="font-semibold block mb-1">Guaranteed Clinical Privacy:</strong>
                  FedMed guarantees strict data localization while achieving predictive accuracy matching centralized baselines.
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};

import React, { useState } from "react";
import { Building2, Database, ShieldAlert, Cpu, Lock, ChevronDown, ChevronUp } from "lucide-react";
import { HOSPITALS_DATA, type HospitalNodeInfo } from "../../data/demoData";

export const HospitalNodesSection: React.FC = () => {
  const [expandedHospitalId, setExpandedHospitalId] = useState<string | null>("hospital_a");

  const toggleExpand = (id: string) => {
    setExpandedHospitalId(expandedHospitalId === id ? null : id);
  };

  return (
    <section id="hospital-nodes" className="py-16 px-6 bg-slate-950 border-b border-slate-800">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* HEADER */}
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-semibold uppercase tracking-wider">
            <Building2 className="w-3.5 h-3.5" />
            SECTION 3 — FEDERATED HOSPITAL NODES
          </div>
          <h2 className="text-3xl font-bold text-white tracking-tight">
            Distributed Healthcare Node Architecture
          </h2>
          <p className="text-slate-400 text-sm max-w-2xl mx-auto">
            Each hospital site operates as an autonomous node executing local training passes on its internal MRI dataset. Raw scans never pass through the firewall.
          </p>
        </div>

        {/* HOSPITAL CARDS GRID */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {HOSPITALS_DATA.map((hosp: HospitalNodeInfo) => {
            const isExpanded = expandedHospitalId === hosp.id;
            return (
              <div
                key={hosp.id}
                className={`rounded-2xl border transition-all duration-300 overflow-hidden ${
                  isExpanded
                    ? "bg-slate-900 border-cyan-500/80 shadow-xl shadow-cyan-500/10"
                    : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
                }`}
              >
                {/* CARD HEADER */}
                <div className="p-6 space-y-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
                        <span className="text-xs font-mono uppercase text-cyan-400 font-bold">{hosp.id}</span>
                      </div>
                      <h3 className="text-lg font-bold text-white mt-1">{hosp.name}</h3>
                      <p className="text-xs text-slate-400 font-mono">Port: {hosp.port} • Host: {hosp.host}</p>
                    </div>

                    <span className="px-2.5 py-1 rounded-md text-[10px] font-extrabold uppercase bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      {hosp.status}
                    </span>
                  </div>

                  {/* METRICS */}
                  <div className="grid grid-cols-3 gap-2 p-3 rounded-xl bg-slate-950 border border-slate-800 text-center">
                    <div>
                      <span className="block text-[10px] text-slate-500 font-bold uppercase">Samples</span>
                      <strong className="text-sm text-white">{hosp.samples}</strong>
                    </div>
                    <div>
                      <span className="block text-[10px] text-slate-500 font-bold uppercase">Local Loss</span>
                      <strong className="text-sm text-red-400">{hosp.loss}</strong>
                    </div>
                    <div>
                      <span className="block text-[10px] text-slate-500 font-bold uppercase">Local Dice</span>
                      <strong className="text-sm text-emerald-400">{hosp.dice}</strong>
                    </div>
                  </div>

                  {/* EXPAND BUTTON */}
                  <button
                    onClick={() => toggleExpand(hosp.id)}
                    className="w-full flex items-center justify-between text-xs font-semibold text-cyan-400 hover:text-cyan-300 pt-2 border-t border-slate-800/80"
                  >
                    <span>{isExpanded ? "Hide Internal Node Workflow" : "Inspect On-Premise Workflow"}</span>
                    {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                </div>

                {/* EXPANDED INNER WORKFLOW */}
                {isExpanded && (
                  <div className="p-6 bg-slate-950 border-t border-slate-800 space-y-4 text-xs">
                    <div className="p-3 rounded-lg bg-emerald-950/20 border border-emerald-900/40 text-emerald-300 flex items-center gap-2">
                      <ShieldAlert className="w-4 h-4 text-emerald-400 shrink-0" />
                      <span>Data Isolation Perimeter Active — No raw MRI leaves this node.</span>
                    </div>

                    <div className="space-y-2">
                      <span className="text-[10px] uppercase font-bold text-slate-400 block">Local Node Execution Sequence</span>
                      <div className="space-y-1.5 font-mono text-[11px]">
                        <div className="p-2 rounded bg-slate-900 text-slate-300 flex items-center gap-2">
                          <Database className="w-3.5 h-3.5 text-cyan-400" />
                          <span>1. Local MRI Dataset ({hosp.datasetName})</span>
                        </div>
                        <div className="p-2 rounded bg-slate-900 text-slate-300 flex items-center gap-2">
                          <Cpu className="w-3.5 h-3.5 text-blue-400" />
                          <span>2. PyTorch 3D U-Net Local Optimization</span>
                        </div>
                        <div className="p-2 rounded bg-slate-900 text-slate-300 flex items-center gap-2">
                          <Lock className="w-3.5 h-3.5 text-indigo-400" />
                          <span>3. Compute Model Update Δw & Encrypt</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
